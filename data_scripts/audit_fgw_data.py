#!/usr/bin/env python3
"""Report what fgw_scores.csv actually contains."""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fgw import FGW_LABEL_SCALE, GW_LABEL_SCALE, similarity_from_distortion  # noqa: E402
from splits import row_split, split_summary  # noqa: E402

DEFAULT_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"
USECOLS = [
    "tm_data_row",
    "source_row",
    "id1",
    "id2",
    "gw_raw",
    "fgw_raw",
    "tm_term",
    "pair_type",
    "tm_score_norm1",
    "tm_score_norm2",
]


def log(msg):
    print(msg, flush=True)


def section(title):
    log("")
    log(title)
    log("-" * len(title))


def contiguous_runs(keys):
    """Number of separate contiguous blocks each key occupies."""
    runs = {}
    previous = object()
    for key in keys:
        if key != previous:
            runs[key] = runs.get(key, 0) + 1
            previous = key
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        log(f"not found: {args.csv}")
        return 1

    size_gb = os.path.getsize(args.csv) / 1e9
    log(f"file   : {args.csv}")
    log(f"size   : {size_gb:.2f} GB")

    header = pd.read_csv(args.csv, nrows=0)
    available = [c for c in USECOLS if c in header.columns]
    missing = [c for c in USECOLS if c not in header.columns]

    frames = [
        chunk
        for chunk in pd.read_csv(
            args.csv, usecols=available, chunksize=args.chunk_size
        )
    ]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    section("volume")
    log(f"  rows                      {len(data):,}")
    if missing:
        log(f"  MISSING COLUMNS           {missing}")
    if data.empty:
        return 1

    keys = data["tm_data_row"].to_numpy()
    unique_pairs = len(np.unique(keys))
    log(f"  protein pairs             {unique_pairs:,}")
    log(f"  rows per pair (mean)      {len(data) / max(unique_pairs, 1):.1f}")
    log(f"  chunks at 5000 rows       {-(-len(data) // 5000):,}")

    proteins = pd.unique(pd.concat([data["id1"], data["id2"]], ignore_index=True))
    log(f"  distinct proteins         {len(proteins):,}")

    section("integrity")
    runs = contiguous_runs(keys)
    duplicated = {k: n for k, n in runs.items() if n > 1}
    if duplicated:
        log(f"  DUPLICATED pairs          {len(duplicated):,}")
        log("    these were written by more than one run and will be")
        log("    over-weighted in training. Example keys: "
            f"{sorted(duplicated)[:5]}")
    else:
        log("  duplicated pairs          0")

    counts = data.groupby("tm_data_row", sort=False).size()
    tiny = counts[counts == 1]
    log(f"  pairs with 1 row          {len(tiny):,}"
        f"{'   <- possible truncation' if len(tiny) else ''}")
    log(f"  rows per pair min/med/max {counts.min()}/{int(counts.median())}/{counts.max()}")

    if "pair_type" in data.columns:
        counts = data["pair_type"].astype(str).value_counts()
        log("  pair types                " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))
        if "aligned" in counts and counts.get("shifted", 0) + counts.get("random", 0) == 0:
            log("    <- no negatives: this file predates off-path sampling in fgw_data.py")

    for column in ("gw_raw", "fgw_raw", "tm_term", "tm_score_norm1", "tm_score_norm2"):
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        bad = int(values.isna().sum())
        log(
            f"  {column:<25} "
            f"range [{values.min():.3f}, {values.max():.3f}]"
            f"{f'   NaN: {bad:,}' if bad else ''}"
        )

    if "source_row" in data.columns:
        source = pd.to_numeric(data["source_row"], errors="coerce").dropna()
        if len(source):
            log(f"  source_row range          [{int(source.min())}, {int(source.max())}]")

    section("labels (exp(-raw / scale), the targets pair_data.py builds)")
    log(f"  {'target':<10}{'pair type':<10}{'scale':<8}{'5%':>8}{'25%':>8}{'50%':>8}{'75%':>8}{'95%':>8}")
    pair_types = (
        data["pair_type"].astype(str) if "pair_type" in data.columns
        else pd.Series(["all"] * len(data), index=data.index)
    )
    for column, scale, name in (
        ("gw_raw", GW_LABEL_SCALE, "structure"),
        ("fgw_raw", FGW_LABEL_SCALE, "composite"),
        ("tm_term", None, "tm_term"),
    ):
        if column not in data.columns:
            continue
        for pair_type in pair_types.unique():
            raw = pd.to_numeric(
                data.loc[pair_types == pair_type, column], errors="coerce"
            ).dropna().to_numpy()
            if not len(raw):
                continue
            label = raw if scale is None else similarity_from_distortion(raw, scale)
            percentiles = np.percentile(label, [5, 25, 50, 75, 95])
            log(
                f"  {name:<10}{pair_type:<10}{str(scale if scale else '-'):<8}"
                + "".join(f"{p:8.3f}" for p in percentiles)
            )
    log("  A usable target spreads over (0, 1]. If the median sits near 0 or 1,")
    log("  change the scale in fgw.py; the CSV does not need regenerating.")

    section(f"split ({split_summary()})")
    pair_ids = data.drop_duplicates("tm_data_row")[["id1", "id2"]]
    assignments = [row_split(a, b) for a, b in zip(pair_ids["id1"], pair_ids["id2"])]

    for name in ("train", "val", "test", "discard"):
        count = assignments.count(name)
        share = 100 * count / max(len(assignments), 1)
        log(f"  {name:<10} {count:>8,} pairs  ({share:5.1f}%)")

    val = assignments.count("val")
    test = assignments.count("test")
    if min(val, test) < 100:
        log("")
        log("  WARNING: val/test are small. Protein-disjoint splitting needs BOTH")
        log("  proteins of a pair in the same bucket, so they shrink quadratically.")
        log("  Raise VAL_BUCKETS / TEST_BUCKETS in splits.py if these are too thin.")

    manifest = f"{args.csv}.runs.jsonl"
    section("provenance")
    if os.path.exists(manifest):
        log(f"  run manifest: {manifest}")
        with open(manifest) as handle:
            for line in handle:
                log(f"    {line.rstrip()}")
    else:
        log("  no run manifest -- this file predates run_pipeline.py.")
        log("  Runs from now on will record their window here.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
