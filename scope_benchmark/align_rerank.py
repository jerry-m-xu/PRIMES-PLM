#!/usr/bin/env python3
"""Re-rank SCOPe40 hits by aligning residue similarity matrices.

    python scope_benchmark/align_rerank.py --checkpoint $D/student_checkpoints_dense_650M/student_best.pt
    python scope_benchmark/align_rerank.py --checkpoint ... --teacher $D/teacher_checkpoints/teacher_best.pt

Run after search_student.py, on a GPU node, with numba installed (pip install
numba). The student's tm score and pooled cosine ignore the order of
residues: a target with the right pieces in the wrong order scores as well as
a homologue. Here each query's best --top hits are scored again by
Smith-Waterman local alignment over a residue-by-residue similarity matrix,
which counts only matches in the same order along both chains, as BLAST does
over BLOSUM62:

  <name>_sw         the student's residue cosines (its PIS estimate),
                    re-ranking the student's tm hits
  esm2_<model>_sw   ESM-2's residue cosines, re-ranking ESM-2's pooled hits:
                    the same method without the student
  <name>_esm2_sw    the average of the two matrices, on the student's tm hits
  <name>_sw_tm      <name>_sw blended with the tm score
  esm2_<model>_sw_on_<name>
                    ESM-2's residue cosines on the student's tm hits: against
                    esm2_<model>_sw it shows what the candidate list is worth,
                    against <name>_sw and <name>_esm2_sw what the matrices are
  teacher_sw_on_<name>
                    with --teacher, the teacher's residue cosines, from the
                    structures (teacher_residues.py), on the student's tm hits:
                    the ceiling for what the student could learn from it

Each pair's matrix is standardised, every entry by its row and by its column
(the mean of the two z-scores), so one set of alignment settings suits either
matrix; a constant (the shift) is then subtracted so that unrelated residues
score below zero. The shift, the gap-open and gap-extend penalties, and the
blend's weight are chosen by grid search on --tune-queries queries that have
a close training homologue (train_identity at or above --max-train-identity),
so the held-out queries below it play no part in tuning; their numbers are
the ones to report. The numbers over all queries include the tuning queries.

Each output file, hits/<variant>.tsv.gz, lists a query's re-ranked hits first
and then the rest of its original list in the original order; the score
column holds the alignment score for the first and the original score after.
ESM-2's residue embeddings (about 5 GB in float16) are cached in
hits/esm2_<model>_residues.pt, the teacher's in hits/teacher_residues.pt, and
the chosen settings are written to hits/<name>_align_params.json.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scope40 import FASTA, HITS, SCOPE_DIR, load_domains, read_fasta  # noqa: E402
from common import ESM_MODELS, log  # noqa: E402
from evaluate_search import eligible_queries, evaluate, print_table, query_set  # noqa: E402
from search_student import cached_embeddings, embed, esm_features, load_student  # noqa: E402
from teacher_residues import teacher_residues  # noqa: E402

try:
    import numba
except ImportError:  # main() asks for it; the module still imports for --help and tests
    numba = None

# the grid searched on the tuning queries, in units of the standardised matrix
SHIFTS = (0.25, 0.5, 1.0, 1.5)
GAP_OPENS = (1.0, 2.0, 4.0)
GAP_EXTENDS = (0.25, 1.0)
BLEND_WEIGHTS = (0.25, 0.5, 0.75)  # of the alignment score; the rest is tm
TUNING_LEVEL = "superfamily"  # the level whose mean sensitivity picks the settings


def jit(**options):
    return numba.njit(**options) if numba else (lambda function: function)


prange = numba.prange if numba else range


@jit(parallel=True, cache=True)
def standardise(S, starts, lengths):
    """Each target's block of S ([query residues, all targets' residues]), with every
    entry z-scored along its row and along its column and the two averaged."""
    Z = np.empty_like(S)
    m = S.shape[0]
    for k in prange(len(starts)):
        a, n = starts[k], lengths[k]
        row_mean, row_sd = np.zeros(m), np.zeros(m)
        col_mean, col_sd = np.zeros(n), np.zeros(n)
        for i in range(m):
            for j in range(n):
                # float64, as numba's float() would not: a float32 square loses a flat row's variance
                v = np.float64(S[i, a + j])
                row_mean[i] += v
                row_sd[i] += v * v
                col_mean[j] += v
                col_sd[j] += v * v
        for i in range(m):
            row_mean[i] /= n
            row_sd[i] = max(row_sd[i] / n - row_mean[i] ** 2, 1e-12) ** 0.5
        for j in range(n):
            col_mean[j] /= m
            col_sd[j] = max(col_sd[j] / m - col_mean[j] ** 2, 1e-12) ** 0.5
        for i in range(m):
            for j in range(n):
                v = np.float64(S[i, a + j])
                Z[i, a + j] = 0.5 * ((v - row_mean[i]) / row_sd[i] + (v - col_mean[j]) / col_sd[j])
    return Z


@jit(parallel=True, cache=True)
def align(Z, starts, lengths, shift, gap_open, gap_extend):
    """Each target block's best local alignment score (Smith-Waterman, affine gaps).

    A cell scores Z - shift; a gap of g residues costs gap_open + (g - 1) * gap_extend.
    """
    m = Z.shape[0]
    scores = np.empty(len(starts), np.float32)
    for k in prange(len(starts)):
        a, n = starts[k], lengths[k]
        H = np.zeros(n + 1, np.float32)  # the previous query residue's row; H[0] is the border
        E = np.full(n + 1, -1e30, np.float32)  # ending with query residues against a gap
        best = 0.0
        for i in range(m):
            diag, left, gap = 0.0, 0.0, -1e30  # gap: ending with target residues against a gap
            for j in range(n):
                up = H[j + 1]
                e = max(up - gap_open, E[j + 1] - gap_extend)
                gap = max(left - gap_open, gap - gap_extend)
                h = max(0.0, diag + Z[i, a + j] - shift, e, gap)
                diag = up
                H[j + 1] = h
                E[j + 1] = e
                left = h
                if h > best:
                    best = h
        scores[k] = best
    return scores


class Residues:
    """Every domain's residue embeddings, unit length, in one tensor on the device."""

    def __init__(self, embeddings, device):
        # float16 halves ESM-2's 5 GB on a GPU; a CPU multiplies float32 far faster
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.lengths = torch.tensor([len(z) for z in embeddings])
        self.offsets = torch.cumsum(self.lengths, 0) - self.lengths
        self.data = torch.cat([F.normalize(z.float(), dim=-1).to(dtype) for z in embeddings]).to(device)

    def similarity(self, query, targets):
        """(cosines [query length, the targets' lengths summed] as float32 numpy,
        each target's first column, each target's length)."""
        targets = torch.tensor(targets)  # a copy: the hit lists' arrays are read-only
        lengths = self.lengths[targets]
        starts = torch.cumsum(lengths, 0) - lengths
        within = torch.arange(int(lengths.sum())) - torch.repeat_interleave(starts, lengths)
        rows = torch.repeat_interleave(self.offsets[targets], lengths) + within
        zq = self.data[self.offsets[query]: self.offsets[query] + self.lengths[query]]
        zt = self.data.index_select(0, rows.to(self.data.device))
        S = (zq @ zt.T).float().cpu().numpy()
        return S, starts.numpy(), lengths.numpy()


@torch.no_grad()
def esm_residues(path, tag, domains, sequences, device):
    """ESM-2's per-residue embeddings of every domain, float16 on the CPU, cached at path."""
    stamp = {"model": tag, "domains": list(domains)}
    if os.path.exists(path):
        cache = torch.load(path, weights_only=False)
        if all(cache.get(key) == value for key, value in stamp.items()):
            log(f"reusing the ESM-2 residues in {path}")
            return cache["residues"]
    from embed_esm2 import load_esm

    model, _, converter = load_esm(ESM_MODELS[tag][0], device=str(device))
    residues = []
    for k, domain in enumerate(domains):
        residues.append(esm_features(sequences[domain], model, converter, device)[0].half().cpu())
        if (k + 1) % 1000 == 0:
            log(f"  ESM-2 residues {k + 1:,}/{len(domains):,}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.save({**stamp, "residues": residues}, path)
    return residues


class Ranked:
    """A hit file's ranked lists as domain indices, with its score column."""

    def __init__(self, path, index):
        hits = pd.read_csv(path, sep="\t", header=None, names=["query", "target", "score"],
                           dtype={"query": str, "target": str, "score": np.float32})
        query, target = hits["query"].map(index), hits["target"].map(index)
        if query.isna().any() or target.isna().any():
            raise SystemExit(f"{path} names domains outside domains.tsv")
        self.target = target.to_numpy(np.int64)
        self.score = hits["score"].to_numpy()
        query = query.to_numpy(np.int64)
        firsts = np.flatnonzero(np.r_[True, query[1:] != query[:-1]])  # each query's hits are contiguous
        ends = np.r_[firsts[1:], len(query)]
        self.blocks = {int(query[a]): (int(a), int(b)) for a, b in zip(firsts, ends)}

    def top(self, query, k):
        """(the first k targets, their scores)."""
        a, b = self.blocks[query]
        return self.target[a:min(a + k, b)], self.score[a:min(a + k, b)]

    def rest(self, query, k):
        """(the targets after the first k, their scores)."""
        a, b = self.blocks[query]
        return self.target[a + k:b], self.score[a + k:b]

    def reorder(self, query, scores):
        """The query's targets, the first len(scores) sorted by scores, best first."""
        top, _ = self.top(query, len(scores))
        rest, _ = self.rest(query, len(scores))
        return np.concatenate([top[np.argsort(-scores, kind="stable")], rest])


def per_query_z(values):
    return (values - values.mean()) / (values.std() + 1e-12)


def mean_sensitivity(ranked, scores_by_query, names, domains):
    """Mean sensitivity at each level over the queries in scores_by_query, their top
    hits ordered by those scores."""
    rows_q, rows_t = [], []
    for query, scores in scores_by_query.items():
        order = ranked.reorder(query, scores)
        rows_q.append(np.full(len(order), query))
        rows_t.append(order)
    hits = pd.DataFrame({"query": names[np.concatenate(rows_q)], "target": names[np.concatenate(rows_t)]})
    return evaluate(hits, domains, set(names[list(scores_by_query)]))[0].mean()


def write_ranked(handle, ranked, query, scores, names):
    top, _ = ranked.top(query, len(scores))
    order = np.argsort(-scores, kind="stable")
    rest, rest_scores = ranked.rest(query, len(scores))
    q = names[query]
    handle.write("".join(f"{q}\t{names[t]}\t{s:.4f}\n"
                         for t, s in zip(top[order].tolist(), scores[order].tolist())))
    handle.write("".join(f"{q}\t{names[t]}\t{s:.5f}\n" for t, s in zip(rest.tolist(), rest_scores.tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="the student search_student.py ran")
    parser.add_argument("--teacher", default=None, help="teacher checkpoint; adds teacher_sw_on_<name>")
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--name", default="student", help="search_student.py's --name")
    parser.add_argument("--top", type=int, default=500, help="hits re-ranked per query")
    parser.add_argument("--tune-queries", type=int, default=500,
                        help="queries with a close training homologue used to choose the settings")
    parser.add_argument("--max-train-identity", type=float, default=0.3,
                        help="queries below this identity to training are held out of tuning and reported")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if numba is None:
        raise SystemExit("align_rerank.py needs numba: pip install numba")
    if hasattr(os, "sched_getaffinity"):  # numba would otherwise start a thread per core of the node
        numba.set_num_threads(min(numba.config.NUMBA_NUM_THREADS, len(os.sched_getaffinity(0))))
    device = torch.device(args.device)
    hits_dir = os.path.join(args.dir, HITS)

    table = load_domains(args.dir)
    sequences = read_fasta(os.path.join(args.dir, FASTA))
    domains = [d for d in table["domain"] if d in sequences]
    names = np.asarray(domains)
    index = {d: i for i, d in enumerate(domains)}
    indexed = table.set_index("domain")
    student, tag, _ = load_student(args.checkpoint, device)
    sources = {"student": os.path.join(hits_dir, f"{args.name}_tm.tsv.gz"),
               "esm2": os.path.join(hits_dir, f"esm2_{tag}_pooled.tsv.gz")}
    for path in sources.values():
        if not os.path.exists(path):
            raise SystemExit(f"no {path}; run search_student.py first")

    start = time.perf_counter()
    embeddings, _ = cached_embeddings(
        os.path.join(hits_dir, f"{args.name}_embeddings.pt"), args.checkpoint, domains,
        lambda: embed(student, tag, [sequences[d] for d in domains], device))
    residues ={"student": Residues(embeddings, device)}
    del embeddings
    residues["esm2"] = Residues(esm_residues(os.path.join(hits_dir, f"esm2_{tag}_residues.pt"),
                                             tag, domains, sequences, device), device)
    if args.teacher:
        residues["teacher"] = Residues(teacher_residues(os.path.join(hits_dir, "teacher_residues.pt"),
                                                        args.teacher, args.dir, domains, sequences, device),
                                       device)
    # the matrices on the student's candidates share its blocks, so every domain needs one length throughout
    for source, embedded in residues.items():
        if not torch.equal(embedded.lengths, residues["student"].lengths):
            raise SystemExit(f"the {source} residues do not match the student's in length")
    ranked = {source: Ranked(path, index) for source, path in sources.items()}
    log(f"embeddings and hit lists loaded in {time.perf_counter() - start:.0f} s; "
        f"numba on {numba.get_num_threads()} threads")

    # variant: (the hit list it re-ranks, output name)
    variants = {"student": ("student", f"{args.name}_sw"),
                "esm2": ("esm2", f"esm2_{tag}_sw"),
                "mixed": ("student", f"{args.name}_esm2_sw"),
                "esm2_same": ("student", f"esm2_{tag}_sw_on_{args.name}")}
    if args.teacher:
        variants["teacher"] = ("student", f"teacher_sw_on_{args.name}")
    blend_name = f"{args.name}_sw_tm"

    def matrices(query):
        """{variant: (standardised matrix, each target's first column, lengths)} for one query."""
        targets, _ = ranked["student"].top(query, args.top)
        S, starts, lengths = residues["student"].similarity(query, targets)
        student_z = standardise(S, starts, lengths)
        esm_z = standardise(residues["esm2"].similarity(query, targets)[0], starts, lengths)
        targets_esm, _ = ranked["esm2"].top(query, args.top)
        S, starts_esm, lengths_esm = residues["esm2"].similarity(query, targets_esm)
        out = {"student": (student_z, starts, lengths),
               "esm2": (standardise(S, starts_esm, lengths_esm), starts_esm, lengths_esm),
               "mixed": (0.5 * (student_z + esm_z), starts, lengths),
               "esm2_same": (esm_z, starts, lengths)}
        if "teacher" in residues:
            out["teacher"] = (standardise(residues["teacher"].similarity(query, targets)[0], starts, lengths),
                              starts, lengths)
        return out

    # 1. choose the settings on queries with a close training homologue
    _, eligible = eligible_queries(indexed)
    held_out = query_set(indexed, args.max_train_identity)
    pool = sorted(d for d in indexed.index[eligible] if d not in held_out and d in index
                  and index[d] in ranked["student"].blocks and index[d] in ranked["esm2"].blocks)
    if not pool:
        raise SystemExit("no eligible queries at or above --max-train-identity to tune on")
    rng = np.random.default_rng(0)
    tune = sorted(index[d] for d in rng.choice(pool, size=min(args.tune_queries, len(pool)), replace=False))
    configs = [(shift, gap_open, gap_extend) for shift in SHIFTS for gap_open in GAP_OPENS
               for gap_extend in GAP_EXTENDS if gap_extend <= gap_open]
    log(f"\ntuning {len(configs)} settings per variant on {len(tune)} queries with a close training homologue")
    start = time.perf_counter()
    tuning = {variant: {config: {} for config in configs} for variant in variants}
    for n, query in enumerate(tune):
        for variant, (Z, starts, lengths) in matrices(query).items():
            for config in configs:
                tuning[variant][config][query] = align(Z, starts, lengths, *config)
        if (n + 1) % 100 == 0:
            log(f"  aligned {n + 1}/{len(tune)} tuning queries ({time.perf_counter() - start:.0f} s)")

    tune_names = set(names[tune])
    chosen, report = {}, {}
    for variant, (source, output) in variants.items():
        before = evaluate(sources[source], indexed, tune_names)[0].mean()
        results = {config: mean_sensitivity(ranked[source], tuning[variant][config], names, indexed)
                   for config in configs}
        chosen[variant] = max(configs, key=lambda c: (results[c][TUNING_LEVEL], results[c]["family"]))
        report[output] = {f"shift={c[0]} open={c[1]} extend={c[2]}": results[c].round(4).to_dict()
                          for c in configs}
        log(f"\n{output}: before re-ranking family {before['family']:.3f}, superfamily "
            f"{before['superfamily']:.3f}, fold {before['fold']:.3f}; best settings:")
        for c in sorted(configs, key=lambda c: -results[c][TUNING_LEVEL])[:5]:
            log(f"  shift {c[0]:<4} open {c[1]:<3} extend {c[2]:<4}  family {results[c]['family']:.3f}  "
                f"superfamily {results[c]['superfamily']:.3f}  fold {results[c]['fold']:.3f}")

    tm = {query: ranked["student"].top(query, args.top)[1] for query in tune}
    sw = tuning["student"][chosen["student"]]
    blends = {w: mean_sensitivity(ranked["student"], {q: w * per_query_z(sw[q]) + (1 - w) * per_query_z(tm[q])
                                                      for q in tune}, names, indexed)
              for w in BLEND_WEIGHTS}
    weight = max(BLEND_WEIGHTS, key=lambda w: (blends[w][TUNING_LEVEL], blends[w]["family"]))
    report[blend_name] = {f"weight={w}": blends[w].round(4).to_dict() for w in BLEND_WEIGHTS}
    log(f"\n{blend_name}: alignment weight {weight} (superfamily "
        + ", ".join(f"{w}: {blends[w]['superfamily']:.3f}" for w in BLEND_WEIGHTS) + ")")
    tune_seconds = time.perf_counter() - start

    # 2. re-rank every query with the chosen settings
    queries = [q for q in range(len(domains)) if q in ranked["student"].blocks and q in ranked["esm2"].blocks]
    log(f"\nre-ranking the top {args.top} hits of {len(queries):,} queries")
    start = time.perf_counter()
    paths = {output: os.path.join(hits_dir, f"{output}.tsv.gz") for _, output in variants.values()}
    paths[blend_name] = os.path.join(hits_dir, f"{blend_name}.tsv.gz")
    handles = {output: gzip.open(path + ".part", "wt", compresslevel=4) for output, path in paths.items()}
    for n, query in enumerate(queries):
        scores = {}
        for variant, (Z, starts, lengths) in matrices(query).items():
            source, output = variants[variant]
            scores[variant] = align(Z, starts, lengths, *chosen[variant])
            write_ranked(handles[output], ranked[source], query, scores[variant], names)
        blend = (weight * per_query_z(scores["student"])
                 + (1 - weight) * per_query_z(ranked["student"].top(query, args.top)[1]))
        write_ranked(handles[blend_name], ranked["student"], query, blend.astype(np.float32), names)
        if (n + 1) % 1000 == 0:
            log(f"  {n + 1:,}/{len(queries):,} ({time.perf_counter() - start:.0f} s)")
    for output, handle in handles.items():
        handle.close()
        os.replace(paths[output] + ".part", paths[output])
    rerank_seconds = time.perf_counter() - start
    log(f"wrote {', '.join(paths.values())}")

    with open(os.path.join(hits_dir, f"{args.name}_align_params.json"), "w") as handle:
        json.dump({"top": args.top, "tune_queries": len(tune), "max_train_identity": args.max_train_identity,
                   "tuning_level": TUNING_LEVEL,
                   "chosen": {variants[v][1]: dict(zip(("shift", "gap_open", "gap_extend"), chosen[v]))
                              for v in variants},
                   "blend_weight": weight, "tuning": report}, handle, indent=2)
    timing_path = os.path.join(hits_dir, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    timing[f"{args.name}_align"] = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "threads": numba.get_num_threads(), "top": args.top, "queries": len(queries),
        "tune_s": tune_seconds, "rerank_s": rerank_seconds,
        "note": "rerank_s covers every variant, matrices included",
    }
    with open(timing_path, "w") as handle:
        json.dump(timing, handle, indent=2)

    names_by_file = {f"{args.name}_tm": sources["student"], f"esm2_{tag}_pooled": sources["esm2"], **paths}
    log("\nall queries (they include the tuning queries)")
    print_table({name: evaluate(path, indexed)[0] for name, path in names_by_file.items()})
    log(f"\nheld out of tuning: queries whose closest training protein is below "
        f"{args.max_train_identity:.0%} identity")
    print_table({name: evaluate(path, indexed, held_out)[0] for name, path in names_by_file.items()})


if __name__ == "__main__":
    main()
