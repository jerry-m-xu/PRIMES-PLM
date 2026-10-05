#!/usr/bin/env python3
"""How much the patch isometry score moves under changes that should not matter.

    python data_scripts/label_noise.py                     # 1000 test protein pairs
    python data_scripts/label_noise.py --pairs 200         # quicker
    python data_scripts/label_noise.py --summarise-only    # re-read the CSV it wrote

For a sample of held-out residue pairs, the structure label is recomputed
under perturbations a structural model should be indifferent to:

  noise_<s>   every CA moved by isotropic Gaussian noise of RMS s angstrom,
              about the coordinate error of a homology model; the patch is
              re-selected on the moved coordinates, so membership can change
  k31, k33    one neighbour fewer or more in each patch
  init        the GW solver started from a random coupling instead of the
              uniform one, which exposes its local optima

Noise and init run twice with independent draws (r1, r2).

What the numbers mean. Let y be the label in fgw_scores.csv. A model that is
stable under sub-angstrom changes, as any useful structural model should be,
cannot follow the part of y that such changes alter. Its error is at least
the label's variance under them, sigma^2, so

    best achievable R^2  ~  1 - sigma^2 / Var(y)

sigma^2 is estimated two ways: Var(r1 - r2) / 2 from the two independent
draws, the primary estimate, and Var(y - r1) against the stored label. Both
are reported overall and per pair type, next to Var(y), so they can be set
against the teacher's and student's test R^2 directly. For the student,
which never sees coordinates, structural uncertainty is irreducible outright.

The run first recomputes every label unperturbed and compares it with the
CSV, so a mismatch in solver settings shows up before anything else.
"""

import argparse
import json
import os
import sys
import time

# one process per core; threaded BLAS on 32x32 matrices is pure overhead
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import log, pdb_path  # noqa: E402
from fgw import (  # noqa: E402
    EPS,
    EPS_START,
    GW_LABEL_SCALE,
    INNER_ITER,
    OUTER_ITER,
    compute_structure_gw,
    fgw_cost_matrix,
    gw_term,
    similarity_from_distortion,
    sinkhorn,
    structure_matrices,
)
from patches import K_NEIGHBORS, knn_indices  # noqa: E402
from splits import row_split  # noqa: E402

DATA_DIR = "/jet/home/jxu23/OCEANDIR"
FGW_CSV = os.path.join(DATA_DIR, "fgw_scores.csv")
PDB_DIR = os.path.join(DATA_DIR, "pdbs")
PACK_DIR = os.path.join(DATA_DIR, "protein_pack")
OUTPUT_CSV = os.path.join(DATA_DIR, "label_noise.csv")

NOISE_LEVELS = (0.1, 0.25, 0.5)  # RMS displacement per CA, angstrom
PATCH_SIZES = (K_NEIGHBORS - 1, K_NEIGHBORS + 1)
REPLICATES = 2
SEED = 0
PAIR_TYPES = ("aligned", "shifted", "random")
USECOLS = ["tm_data_row", "id1", "id2", "pair_type", "residue_idx1", "residue_idx2", "gw_raw"]


def conditions():
    """(column, kind, parameter, replicate) for every recomputation."""
    out = [("reference", "reference", None, 0)]
    for sigma in NOISE_LEVELS:
        for r in range(1, REPLICATES + 1):
            out.append((f"noise_{sigma:g}_r{r}", "noise", sigma, r))
    for k in PATCH_SIZES:
        out.append((f"k{k}", "patch_size", k, 0))
    for r in range(1, REPLICATES + 1):
        out.append((f"init_r{r}", "init", None, r))
    return out


def random_coupling(n, m, rng, sweeps=50):
    """A random positive coupling with uniform marginals (iterative proportional fitting)."""
    T = rng.random((n, m)) + 1e-3
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    for _ in range(sweeps):
        T *= (a / T.sum(axis=1))[:, None]
        T *= (b / T.sum(axis=0))[None, :]
    return T


def structure_gw_from(X, Y, T):
    """fgw.compute_structure_gw, but from a given initial coupling."""
    C1, C2 = structure_matrices(X, Y)
    a, b = np.full(len(C1), 1.0 / len(C1)), np.full(len(C2), 1.0 / len(C2))
    for step in range(OUTER_ITER):
        fraction = step / max(OUTER_ITER - 1, 1)
        eps_step = EPS_START * (EPS / EPS_START) ** fraction
        T = sinkhorn(fgw_cost_matrix(C1, C2, T), a, b, eps=eps_step, n_iter=INNER_ITER)
    return gw_term(C1, C2, T)


_PACK = None


def _init_worker(pack_dir):
    global _PACK
    from protein_pack import open_pack

    _PACK = open_pack(pack_dir)


def load_coords(protein_id):
    """CA coordinates as fgw_data.py saw them (float32 from parse_pdb, packed or not)."""
    packed = _PACK.get(protein_id) if _PACK is not None else None
    if packed is not None:
        return packed[0]
    from parse_pdb import parse_pdb

    coords, _ = parse_pdb(pdb_path(PDB_DIR, protein_id))
    return coords


def score_group(task):
    """Every condition for one protein pair's residue pairs."""
    key, id1, id2, residues = task
    coords1, coords2 = load_coords(id1), load_coords(id2)
    results = {}
    for column, kind, parameter, replicate in conditions():
        # one stream per (pair, condition, replicate): reruns are reproducible
        rng = np.random.default_rng([SEED, int(key), len(results), replicate])
        c1, c2, k = coords1, coords2, K_NEIGHBORS
        if kind == "noise":
            step = parameter / np.sqrt(3.0)  # per axis, so the 3D RMS is `parameter`
            c1 = coords1 + rng.normal(0.0, step, coords1.shape)
            c2 = coords2 + rng.normal(0.0, step, coords2.shape)
        elif kind == "patch_size":
            k = parameter

        values = []
        for i, j in residues:
            X = c1[knn_indices(c1, i, k)]
            Y = c2[knn_indices(c2, j, k)]
            if kind == "init":
                values.append(structure_gw_from(X, Y, random_coupling(len(X), len(Y), rng)))
            else:
                values.append(compute_structure_gw(X, Y))
        results[column] = values
    return key, results


def sample_groups(csv_path, split, count, chunk_size=200000):
    """The first `count` protein pairs of a split, whole.

    fgw_scores.csv follows the source parquet, whose pairs are shuffled, so
    the first ones are a random sample and the file need not be read to the end.
    """
    groups, order = {}, []
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunk_size):
        for key, group in chunk.groupby("tm_data_row", sort=False):
            if key not in groups:
                if len(order) >= count:
                    continue
                if row_split(group["id1"].iloc[0], group["id2"].iloc[0]) != split:
                    continue
                order.append(key)
                groups[key] = [group]
            else:  # a group split across two chunks
                groups[key].append(group)
        # rows of a pair are contiguous, so only the chunk's last pair can continue
        if len(order) >= count and order[-1] != chunk["tm_data_row"].iloc[-1]:
            break
    return [pd.concat(groups[key]) for key in order]


def run(args):
    groups = sample_groups(args.csv, args.split, args.pairs)
    rows = sum(len(g) for g in groups)
    columns = [c for c, *_ in conditions()]
    log(
        f"{len(groups)} {args.split} protein pairs, {rows} residue pairs, "
        f"{len(columns)} recomputations each, {args.workers} workers"
    )

    tasks = [
        (
            int(g["tm_data_row"].iloc[0]),
            g["id1"].iloc[0],
            g["id2"].iloc[0],
            list(zip(g["residue_idx1"].astype(int), g["residue_idx2"].astype(int))),
        )
        for g in groups
    ]
    by_key = {}
    start = time.time()
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=_init_worker, initargs=(args.pack_dir,)) as pool:
            for done, (key, result) in enumerate(pool.imap_unordered(score_group, tasks), 1):
                by_key[key] = result
                if done % 50 == 0:
                    log(f"  {done}/{len(tasks)} protein pairs ({time.time() - start:.0f} s)")
    else:
        _init_worker(args.pack_dir)
        for done, task in enumerate(tasks, 1):
            key, result = score_group(task)
            by_key[key] = result
            if done % 50 == 0:
                log(f"  {done}/{len(tasks)} protein pairs ({time.time() - start:.0f} s)")

    out = []
    for g in groups:
        key = int(g["tm_data_row"].iloc[0])
        frame = g[USECOLS].rename(columns={"gw_raw": "gw_csv"}).reset_index(drop=True)
        for column in columns:
            frame[f"gw_{column}"] = by_key[key][column]
        out.append(frame)
    table = pd.concat(out, ignore_index=True)
    table.to_csv(args.output, index=False)
    log(f"wrote {args.output} ({time.time() - start:.0f} s)")
    return table


def label(raw):
    return similarity_from_distortion(raw, GW_LABEL_SCALE)


def noise_stats(y, a, b=None):
    """Spread of a perturbed label a (and an independent draw b) around y."""
    stats = {
        "pearson_vs_csv": float(np.corrcoef(y, a)[0, 1]) if np.std(a) > 0 else float("nan"),
        "bias_vs_csv": float(np.mean(a - y)),
        "var_vs_csv": float(np.var(y - a)),
    }
    sigma2 = stats["var_vs_csv"]
    if b is not None:
        stats["pearson_between_draws"] = float(np.corrcoef(a, b)[0, 1])
        sigma2 = float(np.var(a - b) / 2.0)
        stats["var_between_draws_half"] = sigma2
    var_y = float(np.var(y))
    stats["sigma2"] = sigma2
    stats["r2_ceiling"] = float(1.0 - sigma2 / var_y) if var_y > 0 else float("nan")
    return stats


def summarise(table, output):
    y = label(table["gw_csv"].to_numpy())
    reference = label(table["gw_reference"].to_numpy())
    mismatch = np.abs(reference - y)
    log("")
    log(
        f"reproduction: max |label difference| {mismatch.max():.2e}, "
        f"{(mismatch > 1e-4).mean():.1%} of pairs differ by more than 1e-4"
    )
    if mismatch.max() > 1e-3:
        log(
            "WARNING: the unperturbed recomputation does not reproduce fgw_scores.csv; "
            "check that fgw.py's solver settings match the ones the CSV was made with"
        )

    families = [(f"noise {s:g} A", f"noise_{s:g}") for s in NOISE_LEVELS]
    families += [(f"patch size {k}", f"k{k}") for k in PATCH_SIZES]
    families += [("solver init", "init")]

    groups = [("all", np.ones(len(table), dtype=bool))]
    groups += [(t, (table["pair_type"] == t).to_numpy()) for t in PAIR_TYPES]

    summary = {"n": int(len(table)), "label_scale": GW_LABEL_SCALE, "groups": {}}
    for group_name, mask in groups:
        if mask.sum() < 3:
            continue
        yg = y[mask]
        log("")
        log(f"  {group_name}: n={mask.sum()}  label mean={yg.mean():.3f}  Var(y)={yg.var():.5f}")
        log(
            f"  {'perturbation':<16}{'r vs csv':>10}{'r draws':>9}{'bias':>9}"
            f"{'sigma^2':>10}{'R2 ceiling':>12}"
        )
        entry = {"n": int(mask.sum()), "var_y": float(yg.var()), "perturbations": {}}
        for name, prefix in families:
            if f"gw_{prefix}" in table:  # single-run conditions
                a, b = label(table[f"gw_{prefix}"].to_numpy())[mask], None
            else:
                a = label(table[f"gw_{prefix}_r1"].to_numpy())[mask]
                b = label(table[f"gw_{prefix}_r2"].to_numpy())[mask]
            stats = noise_stats(yg, a, b)
            entry["perturbations"][name] = stats
            draws = stats.get("pearson_between_draws", float("nan"))
            log(
                f"  {name:<16}{stats['pearson_vs_csv']:>10.4f}{draws:>9.4f}"
                f"{stats['bias_vs_csv']:>+9.4f}{stats['sigma2']:>10.5f}"
                f"{stats['r2_ceiling']:>12.4f}"
            )
        summary["groups"][group_name] = entry

    log("")
    log("  sigma^2 is Var(r1 - r2) / 2 where there are two draws, else Var(csv - perturbed).")
    log("  Compare 'R2 ceiling' with the models' test R2 for the same group.")

    path = os.path.splitext(output)[0] + "_summary.json"
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    log(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=FGW_CSV)
    parser.add_argument("--pack-dir", default=PACK_DIR)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--pairs", type=int, default=1000, help="protein pairs to sample")
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    )
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.output) if args.summarise_only else run(args)
    summarise(table, args.output)


if __name__ == "__main__":
    main()
