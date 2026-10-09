#!/usr/bin/env python3
"""Time every step of structural comparison: TM-align against the student.

    python data_scripts/benchmark_speed.py                       # 35M and 650M inputs
    python data_scripts/benchmark_speed.py --pairs 100 --esm 35M
    python data_scripts/benchmark_speed.py --device cpu          # inference on CPU

Run it on a GPU node (sbatch with train.sh's resources, or an interactive
GPU session). It samples test protein pairs and measures, per item:

  tm_align          one TM-align run per protein pair (tmtools, one CPU core)
  pis_label         one patch isometry score per residue pair (one CPU core)
  embed_<model>     embedding one protein: ESM-2 forward plus the student encoder
  compare_pooled    cosine of two pooled embeddings, O(d)
  compare_global    the student's TM-score prediction: residue similarity
                    matrix, its coverage statistics and the global head
  compare_local     the full calibrated residue similarity matrix
  foldseek_pair     with --foldseek BIN: one exhaustive all-against-all Foldseek
                    search of the sampled structures (first chain), per pair
  mmseqs_pair       with --mmseqs BIN: the same with MMseqs2 on the sequences

External tools run on --threads threads (default 1, like TM-align) with
prefiltering switched off, so every pair is scored, as for TM-align and the
student; their time includes building the databases. TM-Vec is timed by
baselines/tmvec_predict.py, which writes its own timing file.

Timings do not depend on trained weights, so the student is built fresh with
each model's input width and no checkpoint is needed. GPU timings are
synchronised; fast operations are repeated and averaged. Writes a JSON file
for manuscript/scripts/speed_figure.py and prints the numbers the paper's
Cost section needs.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(REPO_ROOT, "student_model"), os.path.join(REPO_ROOT, "teacher_model")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import DATA_DIR, ESM_MODELS, log, pdb_path  # noqa: E402

FGW_CSV = os.path.join(DATA_DIR, "fgw_scores.csv")
PDB_DIR = os.path.join(DATA_DIR, "pdbs")
OUTPUT = os.path.join(DATA_DIR, "benchmark_speed.json")
FAST_REPEATS = 50  # repetitions averaged for sub-millisecond operations


def summary(seconds):
    values = sorted(seconds)
    return {
        "n": len(values),
        "median_s": statistics.median(values),
        "mean_s": statistics.fmean(values),
        "p90_s": values[int(0.9 * (len(values) - 1))],
        "total_s": sum(values),
    }


def sample_pairs(count):
    """The first `count` test protein pairs, with their labelled residue pairs."""
    from pair_data import iter_pair_groups

    pairs = []
    for group in iter_pair_groups(FGW_CSV, split="test", max_groups=count):
        pairs.append((
            group["id1"].iloc[0], group["id2"].iloc[0],
            list(zip(group["residue_idx1"].astype(int), group["residue_idx2"].astype(int))),
        ))
    return pairs


def load_proteins(ids):
    from parse_pdb import parse_pdb

    proteins = {}
    for protein_id in ids:
        coords, sequence = parse_pdb(pdb_path(PDB_DIR, protein_id))
        proteins[protein_id] = (np.asarray(coords, dtype=np.float64), sequence)
    return proteins


def time_tm_align(pairs, proteins):
    from tmtools import tm_align

    times = []
    for id1, id2, _ in pairs:
        (c1, s1), (c2, s2) = proteins[id1], proteins[id2]
        start = time.perf_counter()
        tm_align(c1, c2, s1, s2)
        times.append(time.perf_counter() - start)
    return summary(times)


def time_pis(pairs, proteins, limit):
    from fgw import compute_structure_gw
    from patches import knn_indices

    times = []
    for id1, id2, residues in pairs:
        c1, c2 = proteins[id1][0], proteins[id2][0]
        for i, j in residues:
            if len(times) >= limit:
                return summary(times)
            start = time.perf_counter()
            compute_structure_gw(c1[knn_indices(c1, i)], c2[knn_indices(c2, j)])
            times.append(time.perf_counter() - start)
    return summary(times)


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_embedding(tag, sequences, device):
    """ESM-2 plus the student encoder per protein; returns (stats, embeddings)."""
    from embed_esm2 import get_esm_embeddings, load_esm
    from student_model import SequenceStudent

    model, _, converter = load_esm(ESM_MODELS[tag][0], device=str(device))
    student = SequenceStudent(input_dim=ESM_MODELS[tag][1], dropout=0.0, max_length=1024).to(device).eval()

    def embed(sequence):
        features = torch.from_numpy(get_esm_embeddings(sequence, model, converter, device=str(device)))
        features = features.to(device=device, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad():
            return student(features, mask)[0], student

    for sequence in sequences[:3]:  # warm-up: CUDA kernels, allocator
        embed(sequence)
    times, embeddings = [], []
    for sequence in sequences:
        synchronise(device)
        start = time.perf_counter()
        z, _ = embed(sequence)
        synchronise(device)
        times.append(time.perf_counter() - start)
        embeddings.append(z)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary(times), embeddings, student


def time_comparisons(embeddings, student, device):
    """Per protein pair, averaged over FAST_REPEATS calls each."""
    from student_model import similarity_statistics

    def repeat(fn):
        fn()
        synchronise(device)
        start = time.perf_counter()
        for _ in range(FAST_REPEATS):
            fn()
        synchronise(device)
        return (time.perf_counter() - start) / FAST_REPEATS

    pooled, global_, local = [], [], []
    with torch.no_grad():
        for z1, z2 in embeddings:
            g1 = torch.nn.functional.normalize(z1.mean(0), dim=-1)
            g2 = torch.nn.functional.normalize(z2.mean(0), dim=-1)
            pooled.append(repeat(lambda: (g1 * g2).sum()))

            m1 = torch.ones(1, len(z1), dtype=torch.bool, device=device)
            m2 = torch.ones(1, len(z2), dtype=torch.bool, device=device)
            lengths1 = torch.tensor([len(z1)], device=device)
            lengths2 = torch.tensor([len(z2)], device=device)

            def global_head():
                matrix = (z1 @ z2.T).unsqueeze(0)
                stats = similarity_statistics(matrix, m1, m2)
                return student.global_head(g1[None], g2[None], stats, lengths1, lengths2)

            global_.append(repeat(global_head))
            local.append(repeat(lambda: student.calibration(z1 @ z2.T)))
    return summary(pooled), summary(global_), summary(local)


def write_first_chains(ids, directory):
    """Each protein's first chain as its own PDB file, the chain the pipeline reads."""
    from Bio.PDB import PDBIO, PDBParser, Select

    class FirstChain(Select):
        def __init__(self, chain):
            self.chain = chain

        def accept_model(self, model):
            return model.id == 0

        def accept_chain(self, chain):
            return chain.id == self.chain

    parser, writer = PDBParser(QUIET=True), PDBIO()
    os.makedirs(directory, exist_ok=True)
    for protein_id in ids:
        structure = parser.get_structure(protein_id, pdb_path(PDB_DIR, protein_id))
        first = next(structure[0].get_chains()).id
        writer.set_structure(structure)
        writer.save(os.path.join(directory, f"{protein_id}.pdb"), FirstChain(first))


def time_external(command, n, label):
    """Wall time of one all-against-all run, per pair; None if the tool fails."""
    import subprocess

    log(f"timing {label}: {' '.join(command)}")
    start = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if done.returncode != 0:
        log(f"WARNING: {label} failed (exit {done.returncode}); skipped. {done.stderr.strip()[-400:]}")
        return None
    per_pair = elapsed / (n * n)
    return {"n": n * n, "median_s": per_pair, "mean_s": per_pair, "p90_s": per_pair, "total_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=200, help="test protein pairs to time")
    parser.add_argument("--pis-limit", type=int, default=500, help="residue pairs to label")
    parser.add_argument("--esm", action="append", choices=sorted(ESM_MODELS),
                        help="student input model(s); default 35M and 650M")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=OUTPUT)
    parser.add_argument("--foldseek", default=None, help="path to the foldseek binary")
    parser.add_argument("--mmseqs", default=None, help="path to the mmseqs binary")
    parser.add_argument("--threads", type=int, default=1, help="threads for the external tools")
    args = parser.parse_args()
    tags = args.esm or ["35M", "650M"]
    device = torch.device(args.device)

    pairs = sample_pairs(args.pairs)
    ids = sorted({p for id1, id2, _ in pairs for p in (id1, id2)})
    log(f"{len(pairs)} test protein pairs, {len(ids)} proteins")
    proteins = load_proteins(ids)
    lengths = [len(proteins[p][1]) for p in ids]

    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "pairs": len(pairs),
        "proteins": len(ids),
        "length_median": statistics.median(lengths),
        "length_mean": statistics.fmean(lengths),
        "steps": {},
    }

    log("timing TM-align (one CPU core)")
    result["steps"]["tm_align"] = time_tm_align(pairs, proteins)
    log("timing PIS labels (one CPU core)")
    result["steps"]["pis_label"] = time_pis(pairs, proteins, args.pis_limit)

    index = {p: k for k, p in enumerate(ids)}
    for tag in tags:
        log(f"timing ESM-2 {tag} + student on {result['device']}")
        stats, embeddings, student = time_embedding(tag, [proteins[p][1] for p in ids], device)
        result["steps"][f"embed_{tag}"] = stats
        if tag == tags[0]:  # comparisons cost the same whatever produced the embeddings
            pair_embeddings = [(embeddings[index[a]], embeddings[index[b]]) for a, b, _ in pairs]
            pooled, global_, local = time_comparisons(pair_embeddings, student, device)
            result["steps"]["compare_pooled"] = pooled
            result["steps"]["compare_global"] = global_
            result["steps"]["compare_local"] = local

    if args.foldseek or args.mmseqs:
        import tempfile

        work = tempfile.mkdtemp(prefix="speed_", dir=os.path.dirname(args.out) or ".")
        threads = str(args.threads)
        if args.foldseek:
            structures = os.path.join(work, "structures")
            write_first_chains(ids, structures)
            stats = time_external(
                [args.foldseek, "easy-search", structures, structures, os.path.join(work, "foldseek.m8"),
                 os.path.join(work, "foldseek_tmp"), "--exhaustive-search", "1", "-e", "inf",
                 "--threads", threads], len(ids), "Foldseek")
            if stats:
                result["steps"]["foldseek_pair"] = stats
        if args.mmseqs:
            fasta = os.path.join(work, "sequences.fasta")
            with open(fasta, "w") as handle:
                for protein_id in ids:
                    handle.write(f">{protein_id}\n{proteins[protein_id][1]}\n")
            stats = time_external(
                [args.mmseqs, "easy-search", fasta, fasta, os.path.join(work, "mmseqs.m8"),
                 os.path.join(work, "mmseqs_tmp"), "--exhaustive-search", "-e", "inf",
                 "--threads", threads], len(ids), "MMseqs2")
            if stats:
                result["steps"]["mmseqs_pair"] = stats
        result["external_threads"] = args.threads
        log(f"external tool outputs kept in {work}")

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    log("")
    log(f"device {result['device']}; proteins of median length {result['length_median']:.0f}")
    for step, stats in result["steps"].items():
        log(f"  {step:<16} median {stats['median_s'] * 1e3:10.3f} ms   mean {stats['mean_s'] * 1e3:10.3f} ms   (n={stats['n']})")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
