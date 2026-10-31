#!/usr/bin/env python3
"""Search SCOPe40 all-against-all with the student, then evaluate the search.

    python scope_benchmark/search_student.py --checkpoint $D/student_checkpoints_dense_650M/student_best.pt
    python scope_benchmark/search_student.py --checkpoint ... --name student_epoch3 --max-cells 4e8

Run it on a GPU node after prepare_scope.py. Every domain is embedded once
(ESM-2, then the student encoder); every ordered pair of domains is then
scored in blocks of similar length, so padding stays small:

  tm       the global head's TM-score normalised by the query's length,
           the head's first output, as in training (TM_P with P the query)
  tm_max   the larger of its two normalisations
  pooled   the cosine of the two pooled embeddings: the cheap screening score

and, as the baseline the student has to beat, the cosine of ESM-2's own
mean-pooled embeddings: the student's input, without the student.

For each score, the best --top hits of every query (2000, the depth the
benchmark gives Foldseek and MMseqs2) go to hits/<name>_<score>.tsv.gz, and
the baseline's to hits/esm2_<model>_pooled.tsv.gz, best first; then
evaluate_search.py's sensitivities are printed. Wall times go to
hits/timing.json.

The inputs follow training: the ESM-2 model the checkpoint was trained on, its
last layer, rounded through float16 as precompute_esm.py stores it, and at
most MAX_SEQ_LENGTH residues; the few longer domains are truncated. The
student's embeddings and ESM-2's pooled ones are cached in
hits/<name>_embeddings.pt with the checkpoint's path and modification time,
so a rerun skips ESM-2.
"""

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(REPO_ROOT, "student_model")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scope40 import FASTA, HITS, SCOPE_DIR, load_domains, read_fasta  # noqa: E402
from common import ESM_MODELS, log  # noqa: E402
from evaluate_search import evaluate, print_table, query_set  # noqa: E402
from student_model import SequenceStudent, similarity_statistics  # noqa: E402

MAX_SEQ_LENGTH = 1000  # train_student.MAX_SEQ_LENGTH and precompute_esm.MAX_SEQUENCE_LENGTH


def load_student(path, device):
    """(student, ESM-2 model name, epoch), built as test_student.load_student builds it.

    Not imported from there: test_student imports train_student, whose signal
    handlers turn Ctrl-C into a request to checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    student = SequenceStudent(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=0.0,
        max_length=config["max_length"],
        use_tm_head=config.get("use_tm_head", True),
    ).to(device)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    tag = config.get("student_esm") or next(
        tag for tag, (_, width) in ESM_MODELS.items() if width == config["input_dim"])
    return student, tag, checkpoint.get("epoch")


def esm_features(sequence, model, converter, device):
    """ESM-2's last layer for one sequence as the student sees it, [1, L, E] float32 on device."""
    from embed_esm2 import get_esm_embeddings

    features = get_esm_embeddings(sequence[:MAX_SEQ_LENGTH], model, converter, device=str(device))
    return torch.from_numpy(features.astype(np.float16).astype(np.float32)).to(device)[None]


@torch.no_grad()
def embed(student, tag, sequences, device):
    """(each sequence's per-residue student embeddings, [L, D] float32 on the CPU;
    ESM-2's mean-pooled embeddings, normalised, [N, E] on the CPU)."""
    from embed_esm2 import load_esm

    model, _, converter = load_esm(ESM_MODELS[tag][0], device=str(device))
    embeddings, esm_pooled = [], []
    for k, sequence in enumerate(sequences):
        features = esm_features(sequence, model, converter, device)
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        embeddings.append(student(features, mask)[0].cpu())
        esm_pooled.append(F.normalize(features[0].mean(0), dim=-1).cpu())
        if (k + 1) % 1000 == 0:
            log(f"  embedded {k + 1:,}/{len(sequences):,}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, torch.stack(esm_pooled)


def cached_embeddings(path, checkpoint, domains, compute):
    """compute()'s (student embeddings, ESM-2 pooled), from the cache when it matches."""
    stamp = {"checkpoint": os.path.abspath(checkpoint), "mtime": os.path.getmtime(checkpoint),
             "domains": list(domains)}
    if os.path.exists(path):
        cache = torch.load(path, weights_only=False)
        # caches written before the ESM-2 baseline lack esm_pooled
        if all(cache.get(key) == value for key, value in stamp.items()) and "esm_pooled" in cache:
            log(f"reusing the embeddings in {path}")
            return cache["embeddings"], cache["esm_pooled"]
    embeddings, esm_pooled = compute()
    torch.save({**stamp, "embeddings": embeddings, "esm_pooled": esm_pooled}, path)
    return embeddings, esm_pooled


@torch.no_grad()
def score_pairs(student, embeddings, device, block_size, max_cells):
    """[N, N] scores of every ordered pair, the query in the row, on the CPU.

    Returns {"pooled": cosine of pooled embeddings, "tm_query": the global
    head's TM-score normalised by the query, "tm_target": by the target}; the
    TM-scores only when the checkpoint has a global head.
    """
    n, dim = len(embeddings), embeddings[0].shape[1]
    lengths = torch.tensor([len(z) for z in embeddings])
    # the global head pools over every residue, as forward_pair's masked_mean does
    pooled = torch.stack([F.normalize(z.mean(0), dim=-1) for z in embeddings]).to(device)
    scores = {"pooled": (pooled @ pooled.T).cpu()}
    if student.global_head is None:
        return scores

    blocks = []
    order = torch.argsort(lengths)
    for start in range(0, n, block_size):
        index = order[start:start + block_size]
        width = int(lengths[index].max())
        z = torch.zeros(len(index), width, dim)
        for row, protein in enumerate(index.tolist()):
            z[row, : lengths[protein]] = embeddings[protein]
        mask = torch.arange(width)[None] < lengths[index][:, None]
        blocks.append((index, index.to(device), z.to(device), mask.to(device)))
    lengths = lengths.to(device)

    tm_query, tm_target = torch.empty(n, n), torch.empty(n, n)
    for k, (q_index, q_device, zq, mq) in enumerate(blocks):
        for t_index, t_device, zt, mt in blocks:
            m, width, bt = zq.shape[1], zt.shape[1], len(t_index)
            step = max(1, int(max_cells) // (bt * m * width))
            for start in range(0, len(q_index), step):
                rows = slice(start, start + step)
                qi, c = q_device[rows], len(q_device[rows])
                similarity = torch.einsum("qmd,tnd->qtmn", zq[rows], zt).reshape(c * bt, m, width)
                stats = similarity_statistics(similarity,
                                              mq[rows, None, :].expand(c, bt, m).reshape(c * bt, m),
                                              mt[None].expand(c, bt, width).reshape(c * bt, width))
                tm = student.global_head(
                    pooled[qi][:, None].expand(c, bt, dim).reshape(c * bt, dim),
                    pooled[t_device][None].expand(c, bt, dim).reshape(c * bt, dim),
                    stats,
                    lengths[qi][:, None].expand(c, bt).reshape(-1),
                    lengths[t_device][None].expand(c, bt).reshape(-1),
                ).view(c, bt, 2).cpu()
                cells = (q_index[rows][:, None], t_index[None, :])
                tm_query[cells] = tm[..., 0]
                tm_target[cells] = tm[..., 1]
        if (k + 1) % 20 == 0 or k + 1 == len(blocks):
            log(f"  scored {min((k + 1) * block_size, n):,}/{n:,} queries")
    scores["tm_query"], scores["tm_target"] = tm_query, tm_target
    return scores


def write_hits(scores, domains, path, top):
    """Each query's best `top` targets, best first, self excluded: query<TAB>target<TAB>score."""
    names = np.asarray(domains)
    k = min(top, len(names) - 1)
    with gzip.open(path, "wt", compresslevel=4) as handle:
        for start in range(0, len(names), 1024):
            block = scores[start:start + 1024].clone()
            rows = torch.arange(len(block))
            block[rows, rows + start] = -float("inf")
            values, index = torch.topk(block, k, dim=1)
            index = index.numpy()
            for row in range(len(block)):
                query = names[start + row]
                handle.write("".join(f"{query}\t{target}\t{value:.5f}\n"
                                     for target, value in zip(names[index[row]], values[row].tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--name", default="student", help="prefix of the output files")
    parser.add_argument("--top", type=int, default=2000, help="hits kept per query")
    parser.add_argument("--block-size", type=int, default=64, help="proteins per length block")
    parser.add_argument("--max-cells", type=float, default=2e8,
                        help="similarity-matrix entries scored at once; lower it if the GPU runs out of memory")
    parser.add_argument("--max-train-identity", type=float, default=None,
                        help="also report on queries whose closest training protein is below this identity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    hits_dir = os.path.join(args.dir, HITS)
    os.makedirs(hits_dir, exist_ok=True)

    table = load_domains(args.dir)
    sequences = read_fasta(os.path.join(args.dir, FASTA))
    domains = [d for d in table["domain"] if d in sequences]
    student, tag, epoch = load_student(args.checkpoint, device)
    log(f"student {args.checkpoint} (epoch {epoch}, ESM-2 {tag}); {len(domains):,} domains on {device}")
    truncated = sum(len(sequences[d]) > MAX_SEQ_LENGTH for d in domains)
    if truncated:
        log(f"{truncated} domains longer than {MAX_SEQ_LENGTH} residues are truncated")

    start = time.perf_counter()
    embeddings, esm_pooled = cached_embeddings(
        os.path.join(hits_dir, f"{args.name}_embeddings.pt"), args.checkpoint, domains,
        lambda: embed(student, tag, [sequences[d] for d in domains], device))
    embed_seconds = time.perf_counter() - start
    log(f"scoring {len(domains) ** 2:,} ordered pairs")
    start = time.perf_counter()
    scores = score_pairs(student, embeddings, device, args.block_size, args.max_cells)
    score_seconds = time.perf_counter() - start

    variants = {f"{args.name}_pooled": scores["pooled"]}
    if "tm_query" in scores:
        variants[f"{args.name}_tm"] = scores["tm_query"]
        variants[f"{args.name}_tm_max"] = torch.maximum(scores["tm_query"], scores["tm_target"])
    esm_pooled = esm_pooled.to(device)
    variants[f"esm2_{tag}_pooled"] = (esm_pooled @ esm_pooled.T).cpu()
    paths = {}
    for name, matrix in variants.items():
        paths[name] = os.path.join(hits_dir, f"{name}.tsv.gz")
        write_hits(matrix, domains, paths[name], args.top)
    log(f"wrote {', '.join(paths.values())}")

    timing_path = os.path.join(hits_dir, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    timing[args.name] = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "domains": len(domains), "embed_s": embed_seconds, "score_s": score_seconds,
        "note": "embed_s is ~0 when the cached embeddings were reused",
    }
    with open(timing_path, "w") as handle:
        json.dump(timing, handle, indent=2)

    indexed = table.set_index("domain")
    log("")
    print_table({name: evaluate(path, indexed)[0] for name, path in paths.items()})
    queries = query_set(indexed, args.max_train_identity)
    if queries is not None:
        log(f"\nqueries whose closest training protein is below {args.max_train_identity:.0%} identity")
        print_table({name: evaluate(path, indexed, queries)[0] for name, path in paths.items()})


if __name__ == "__main__":
    main()
