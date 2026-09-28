#!/usr/bin/env python3
"""Load protein pairs exactly as the trainers do, and report what fails.

    python data_scripts/check_loader.py                # 500 train + 500 val pairs
    python data_scripts/check_loader.py --pairs 2000
    python data_scripts/check_loader.py --workers 0    # in-process: full error histogram
    python data_scripts/check_loader.py --no-pack      # the old per-file path, for comparison

The trainers skip any pair that fails to load, so a broken dataset trains
quietly on whatever survives. This runs the same loader the student uses
(the superset of the teacher's: patches, sequences, and the extra
distillation residues), through the same DataLoader and worker count the
trainers use, and prints a failure summary per split. It also times loading,
which is the floor on how fast an epoch can go.

Exit code 0 when every split loads with at most 1% failures, 1 otherwise.
Run it from the repository root on a compute node with as many CPUs as the
training job (train.sh asks for 5), not on a login node.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import log, log_failures  # noqa: E402
from pair_data import (  # noqa: E402
    ProteinDataCache,
    ProteinPairDataset,
    iter_group_buffers,
    loader_workers,
    pair_loader,
)

DATA_DIR = "/jet/home/jxu23/OCEANDIR"
MAX_FAILURE_RATE = 0.01
EXTRA_DISTILL_RESIDUES = 16  # train_student.py's value
BATCH_SIZE = 4  # both trainers' protein pairs per step
TRAIN_PAIRS = 656_434  # the current train split, for the epoch estimate


def check_split(csv_path, cache, split, pairs, workers):
    """Load the first `pairs` pairs of one split. Returns (delivered, failed, seconds)."""
    try:
        groups = next(iter_group_buffers(csv_path, split=split, buffer_size=pairs))
    except StopIteration:
        log(f"{split}: no pairs in this split")
        return 0, 0, 0.0

    if cache.pack is not None:
        ids = {g["id1"].iloc[0] for g in groups} | {g["id2"].iloc[0] for g in groups}
        in_pack = sum(protein_id in cache.pack for protein_id in ids)
        log(f"{split}: {in_pack:,} of {len(ids):,} proteins in this sample are in the pack")

    dataset = ProteinPairDataset(
        groups,
        cache,
        include_sequence=True,
        extra_distill_residues=EXTRA_DISTILL_RESIDUES,
    )
    loader = pair_loader(dataset, BATCH_SIZE, shuffle=False, workers=workers)

    start = time.time()
    delivered, first = 0, None
    for batch in loader:
        if batch is None:
            continue
        delivered += batch["tm"].shape[0]
        if first is None:
            first = batch
    seconds = time.time() - start

    if first is not None:
        shapes = ", ".join(
            f"{key} {tuple(first[key].shape)}"
            for key in ("patch_features1", "features1", "pair_mask")
            if key in first
        )
        log(f"{split}: collated a batch of {first['tm'].shape[0]}: {shapes}")

    failed = len(dataset) - delivered
    log(
        f"{split}: {delivered:,} of {len(dataset):,} pairs loaded in {seconds:.0f} s "
        f"({len(dataset) / max(seconds, 1e-9):.1f} pairs/s, {workers} workers, cold cache)"
    )
    if dataset.errors:
        log_failures(dataset.errors, len(dataset))
    elif failed:
        log(f"  failures: {failed} of {len(dataset)} (each logged above by the worker that hit it;"
            f" --workers 0 gives a histogram)")
    else:
        log("  no failures")
    return delivered, failed, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--csv", default=None, help="default: <data-dir>/fgw_scores.csv")
    parser.add_argument("--pack-dir", default=None, help="default: <data-dir>/protein_pack")
    parser.add_argument("--no-pack", action="store_true", help="read PDB and embedding files directly")
    parser.add_argument("--pairs", type=int, default=500, help="pairs per split")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument(
        "--workers", type=int, default=loader_workers(),
        help="loader processes, as in training (default: this job's CPUs minus one)",
    )
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.data_dir, "fgw_scores.csv")
    for path in (csv_path, os.path.join(args.data_dir, "pdbs"), os.path.join(args.data_dir, "embeddings")):
        if not os.path.exists(path):
            log(f"not found: {path}")
            return 1

    log(f"csv: {csv_path}")
    pack_dir = None if args.no_pack else (args.pack_dir or os.path.join(args.data_dir, "protein_pack"))
    cache = ProteinDataCache(
        os.path.join(args.data_dir, "pdbs"),
        os.path.join(args.data_dir, "embeddings"),
        max_size=512,
        pack_dir=pack_dir,
    )

    ok = True
    total_attempted, total_seconds = 0, 0.0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        log("")
        delivered, failed, seconds = check_split(csv_path, cache, split, args.pairs, args.workers)
        attempted = delivered + failed
        if attempted == 0 or failed / attempted > MAX_FAILURE_RATE:
            ok = False
        total_attempted += attempted
        total_seconds += seconds

    log("")
    if total_attempted:
        rate = total_attempted / max(total_seconds, 1e-9)
        log(
            f"loading rate {rate:.1f} pairs/s with {args.workers} workers: one pass over "
            f"{TRAIN_PAIRS:,} training pairs would spend about {TRAIN_PAIRS / rate / 3600:.1f} h "
            f"on data loading"
        )
    log("OK: the data is ready" if ok else f"FAILED: more than {MAX_FAILURE_RATE:.0%} of pairs did not load")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
