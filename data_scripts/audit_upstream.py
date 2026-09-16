#!/usr/bin/env python3
"""Check that stage 1 (embeddings) and stage 2 (tm_scores.csv) are sound.

    python data_scripts/audit_upstream.py

Run before regenerating fgw_scores.csv. Stage 3 consumes both of these, and a
defect here shows up there as skipped pairs rather than as an error, so it is
worth knowing first.

The embedding length check parses PDBs, which is slow, so it runs on a random
sample (--sample). It is the check that catches multi-model PDBs, whose stored
embeddings were built from a concatenated sequence.
"""

import argparse
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import embedding_path, log, pdb_path  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402

TM_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
MANIFEST_CSV = "/jet/home/jxu23/OCEANDIR/esm_manifest.csv"


def section(title):
    log("")
    log(title)
    log("-" * len(title))


def audit_tm_scores(path, chunk_size):
    section("stage 2: tm_scores.csv")
    if not os.path.exists(path):
        log(f"  MISSING: {path}")
        return None

    log(f"  file        {path}")
    log(f"  size        {os.path.getsize(path) / 1e9:.2f} GB")

    rows = 0
    source_rows = []
    proteins = set()
    tm_values = []
    bad_alignment = 0
    empty_alignment = 0
    bad_ids = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size, keep_default_na=False):
        rows += len(chunk)
        if "row" in chunk.columns:
            source_rows.append(pd.to_numeric(chunk["row"], errors="coerce"))
        proteins.update(chunk["id1"].astype(str))
        proteins.update(chunk["id2"].astype(str))
        tm_values.append(pd.to_numeric(chunk["tm_score_norm1"], errors="coerce"))

        bad_ids += int(
            (chunk["id1"].astype(str).str.strip() == "").sum()
            + (chunk["id2"].astype(str).str.strip() == "").sum()
        )

        # fgw_data raises if these three differ, dropping the pair
        lengths = pd.DataFrame(
            {
                "x": chunk["seqxA"].astype(str).str.len(),
                "m": chunk["seqM"].astype(str).str.len(),
                "y": chunk["seqyA"].astype(str).str.len(),
            }
        )
        bad_alignment += int(
            ((lengths["x"] != lengths["m"]) | (lengths["m"] != lengths["y"])).sum()
        )
        empty_alignment += int((lengths["x"] == 0).sum())

    tm = pd.concat(tm_values) if tm_values else pd.Series(dtype=float)
    log(f"  pairs       {rows:,}")
    log(f"  proteins    {len(proteins):,} distinct")

    if source_rows:
        source = pd.concat(source_rows).dropna().astype(int)
        duplicates = int(source.duplicated().sum())
        log(f"  source rows [{source.min():,}, {source.max():,}]")
        log(
            f"  duplicated  {duplicates:,}"
            f"{'   <- pairs written by more than one run' if duplicates else ''}"
        )
        covered = set(source.tolist())
        span = source.max() - source.min() + 1
        gaps = span - len(covered)
        log(f"  gaps in span {gaps:,} of {span:,} rows not present")
        log("               (expected: alignment failures are dropped)")
    else:
        log("  NO 'row' COLUMN -- source parquet rows are not recoverable")

    log(f"  tm_score    [{tm.min():.3f}, {tm.max():.3f}]  NaN: {int(tm.isna().sum()):,}")
    log(
        f"  alignment   {bad_alignment:,} rows with mismatched seqxA/seqM/seqyA"
        f"{'   <- stage 3 will drop these' if bad_alignment else ''}"
    )
    log(f"              {empty_alignment:,} rows with an empty alignment")
    if bad_ids:
        log(f"  blank ids   {bad_ids:,}")
    return proteins


def audit_embeddings(proteins, embedding_dir, pdb_dir, sample_size):
    section("stage 1: embeddings")
    if not os.path.isdir(embedding_dir):
        log(f"  MISSING: {embedding_dir}")
        return

    on_disk = {f[:-4] for f in os.listdir(embedding_dir) if f.endswith(".npy")}
    log(f"  .npy files  {len(on_disk):,}")

    if proteins is None:
        log("  (no tm_scores.csv, cannot check coverage)")
        return

    missing = proteins - on_disk
    log(f"  referenced  {len(proteins):,} proteins in tm_scores.csv")
    log(
        f"  MISSING     {len(missing):,}"
        f"{'   <- every pair using these will be dropped by stage 3' if missing else ''}"
    )
    if missing:
        log(f"    examples: {sorted(missing)[:5]}")
    extra = on_disk - proteins
    if extra:
        log(f"  unused      {len(extra):,} embeddings for proteins not in tm_scores.csv")

    present = sorted(proteins & on_disk)
    if not present:
        return

    sample = random.Random(0).sample(present, min(sample_size, len(present)))
    section(f"stage 1: sampled integrity ({len(sample)} proteins)")

    dtypes, dims, mismatched, unreadable = Counter(), Counter(), [], []
    for protein_id in sample:
        try:
            emb = np.load(embedding_path(embedding_dir, protein_id))
            dtypes[str(emb.dtype)] += 1
            dims[emb.shape[-1]] += 1
            coords, _ = parse_pdb(pdb_path(pdb_dir, protein_id))
            if len(coords) != len(emb):
                mismatched.append((protein_id, len(coords), len(emb)))
        except Exception as exc:
            unreadable.append((protein_id, type(exc).__name__))

    log(f"  dtype       {dict(dtypes)}")
    log(f"  dim         {dict(dims)}")
    log(
        f"  length mismatch (coords vs embedding): {len(mismatched)}"
        f"{'   <- these fail at training time' if mismatched else ''}"
    )
    for protein_id, n_coords, n_emb in mismatched[:5]:
        ratio = n_emb / n_coords if n_coords else 0
        note = "  (looks like a multi-model PDB)" if ratio > 1.5 else ""
        log(f"    {protein_id}: {n_coords} coords vs {n_emb} embeddings{note}")
    if mismatched:
        rate = len(mismatched) / len(sample)
        log(f"    sampled rate {rate:.1%} -> roughly {int(rate * len(present)):,} proteins")
    if unreadable:
        log(f"  unreadable  {len(unreadable)}: {unreadable[:5]}")


def audit_manifest(path):
    if not os.path.exists(path):
        return
    section("stage 1: esm_manifest.csv")
    manifest = pd.read_csv(path, keep_default_na=False)
    log(f"  rows        {len(manifest):,}")
    if "status" in manifest.columns:
        for status, count in manifest["status"].value_counts().items():
            log(f"    {status:<10} {count:,}")
        failed = manifest[manifest["status"] == "failed"]
        if len(failed) and "error" in failed.columns:
            reasons = failed["error"].astype(str).str.slice(0, 60).value_counts()
            log("  failure reasons:")
            for reason, count in reasons.head(5).items():
                log(f"    {count:>6}  {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tm-csv", default=TM_CSV)
    parser.add_argument("--pdb-dir", default=PDB_DIR)
    parser.add_argument("--embedding-dir", default=EMBEDDING_DIR)
    parser.add_argument("--manifest", default=MANIFEST_CSV)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()

    proteins = audit_tm_scores(args.tm_csv, args.chunk_size)
    audit_embeddings(proteins, args.embedding_dir, args.pdb_dir, args.sample)
    audit_manifest(args.manifest)
    log("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
