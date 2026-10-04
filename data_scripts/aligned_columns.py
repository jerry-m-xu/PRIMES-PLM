#!/usr/bin/env python3
"""Every TM-align-aligned residue pair of every protein pair, in a compact store.

    python data_scripts/aligned_columns.py          # reads tm_scores.csv, a few minutes

fgw_scores.csv keeps every 32nd aligned column, because each one needs a
Gromov-Wasserstein solve. The student's dense teacher-scored pairs need no
label, only the alignment, so they can use every column. tm_scores.csv
already holds it as the seqxA/seqM/seqyA strings; this script turns them into
runs of consecutive aligned pairs, which is all a TM-align path is between
gaps, and stores those:

    rows.npy      int64 [n]        tm_scores "row" (= fgw_scores tm_data_row), sorted
    offsets.npy   int64 [n + 1]    row k's runs are segments[offsets[k]:offsets[k + 1]]
    segments.npy  int32 [S, 3]     (i0, j0, length): residues i0 + t of protein 1
                                   and j0 + t of protein 2 are aligned, t < length

The arrays are memory-mapped by the loader, so every worker shares one copy.
Which columns count as aligned is iter_aligned_residue_pairs()'s rule, the
same one fgw_data.py samples its positives from.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import log  # noqa: E402

DATA_DIR = "/jet/home/jxu23/OCEANDIR"
TM_DATA_CSV = os.path.join(DATA_DIR, "tm_scores.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "aligned_columns")
CSV_CHUNK_SIZE = 20000

GAP = ord("-")
# fgw_data.SEQM_VALUES: every column where both residues are present
SEQM_BYTES = np.frombuffer(b":. ", dtype=np.uint8)


def aligned_pairs(seqxA, seqM, seqyA):
    """(idx1, idx2) arrays of the aligned residue pairs, in alignment order.

    The vectorised form of fgw_data.iter_aligned_residue_pairs().
    """
    a = np.frombuffer(seqxA.encode(), dtype=np.uint8)
    m = np.frombuffer(seqM.encode(), dtype=np.uint8)
    b = np.frombuffer(seqyA.encode(), dtype=np.uint8)
    if not (len(a) == len(m) == len(b)):
        raise ValueError("seqxA, seqM, and seqyA must have the same length")

    present1 = a != GAP
    present2 = b != GAP
    keep = present1 & present2 & np.isin(m, SEQM_BYTES)
    idx1 = np.cumsum(present1) - 1
    idx2 = np.cumsum(present2) - 1
    return idx1[keep], idx2[keep]


def to_segments(idx1, idx2):
    """Runs where both indices advance by one, as (i0, j0, length) rows."""
    if len(idx1) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    breaks = np.flatnonzero((np.diff(idx1) != 1) | (np.diff(idx2) != 1)) + 1
    starts = np.concatenate([[0], breaks])
    lengths = np.diff(np.concatenate([starts, [len(idx1)]]))
    return np.stack([idx1[starts], idx2[starts], lengths], axis=1).astype(np.int32)


def segments_to_pairs(segments):
    """Inverse of to_segments()."""
    if len(segments) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    lengths = segments[:, 2].astype(np.int64)
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    idx1 = np.repeat(segments[:, 0].astype(np.int64), lengths) + within
    idx2 = np.repeat(segments[:, 1].astype(np.int64), lengths) + within
    return idx1, idx2


class AlignedColumns:
    """Read side: row -> (idx1, idx2) of every aligned residue pair."""

    def __init__(self, directory):
        self.directory = directory
        self.rows = np.load(os.path.join(directory, "rows.npy"), mmap_mode="r")
        self.offsets = np.load(os.path.join(directory, "offsets.npy"), mmap_mode="r")
        self.segments = np.load(os.path.join(directory, "segments.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def get(self, row):
        """The aligned pairs of tm_scores row `row`, or None if it is not stored."""
        k = int(np.searchsorted(self.rows, row))
        if k >= len(self.rows) or int(self.rows[k]) != int(row):
            return None
        start, end = int(self.offsets[k]), int(self.offsets[k + 1])
        return segments_to_pairs(np.asarray(self.segments[start:end]))


def open_aligned_columns(directory):
    """AlignedColumns for a built store, or None when there is none."""
    if directory and os.path.exists(os.path.join(directory, "rows.npy")):
        return AlignedColumns(directory)
    return None


def build(csv_path, output_dir):
    by_row = {}
    seen = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["row", "seqxA", "seqM", "seqyA"],
        chunksize=CSV_CHUNK_SIZE,
        keep_default_na=False,
        dtype={"seqxA": str, "seqM": str, "seqyA": str},
    ):
        for row, seqxA, seqM, seqyA in zip(
            chunk["row"].to_numpy(), chunk["seqxA"], chunk["seqM"], chunk["seqyA"]
        ):
            seen += 1
            row = int(row)
            # overlapping shard runs duplicated some source rows; the first
            # copy is the one fgw_data.py scored
            if row in by_row:
                continue
            by_row[row] = to_segments(*aligned_pairs(seqxA, seqM, seqyA))
        log(f"  {seen} rows read, {len(by_row)} distinct")

    rows = np.array(sorted(by_row), dtype=np.int64)
    counts = np.array([len(by_row[row]) for row in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    segments = (
        np.concatenate([by_row[row] for row in rows])
        if len(rows)
        else np.zeros((0, 3), dtype=np.int32)
    )

    os.makedirs(output_dir, exist_ok=True)
    for name, array in (("rows", rows), ("offsets", offsets), ("segments", segments)):
        tmp = os.path.join(output_dir, f"{name}.tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, os.path.join(output_dir, f"{name}.npy"))

    columns = int(segments[:, 2].sum()) if len(segments) else 0
    log(
        f"wrote {output_dir}: {len(rows)} protein pairs, {len(segments)} runs, "
        f"{columns} aligned columns ({columns / max(len(rows), 1):.0f} per pair)"
    )
    return rows, segments


def verify(csv_path, output_dir, samples=200, seed=0):
    """Check a random sample against fgw_data.py's own parser."""
    from fgw_data import iter_aligned_residue_pairs

    store = AlignedColumns(output_dir)
    rng = np.random.default_rng(seed)
    wanted = set(int(row) for row in rng.choice(store.rows, size=min(samples, len(store)), replace=False))
    checked = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=["row", "seqxA", "seqM", "seqyA"],
        chunksize=CSV_CHUNK_SIZE,
        keep_default_na=False,
        dtype={"seqxA": str, "seqM": str, "seqyA": str},
    ):
        for row, seqxA, seqM, seqyA in zip(
            chunk["row"].to_numpy(), chunk["seqxA"], chunk["seqM"], chunk["seqyA"]
        ):
            row = int(row)
            if row not in wanted:
                continue
            wanted.discard(row)
            expected = [(i, j) for _, i, j, _, _, _ in iter_aligned_residue_pairs(seqxA, seqM, seqyA)]
            idx1, idx2 = store.get(row)
            if list(zip(idx1.tolist(), idx2.tolist())) != expected:
                raise AssertionError(f"row {row}: stored alignment differs from fgw_data's")
            checked += 1
        if not wanted:
            break
    log(f"verified {checked} protein pairs against iter_aligned_residue_pairs()")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=TM_DATA_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    log(f"reading {args.csv}")
    build(args.csv, args.output_dir)
    if not args.no_verify:
        verify(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
