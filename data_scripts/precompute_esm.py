#!/usr/bin/env python3

import csv
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

from common import (
    build_parser,
    embedding_path,
    iter_parquet_rows,
    log_failures,
    pdb_path,
    write_progress,
    write_run_manifest,
)
from embed_esm2 import get_esm_embeddings, load_esm
from parse_pdb import parse_pdb

# global config
PARQUET_PATH = "/jet/home/jxu23/OCEANDIR/swiss_under_1000_320M.parquet"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
MANIFEST_CSV = "/jet/home/jxu23/OCEANDIR/esm_manifest.csv"
PROGRESS_FILE = "/jet/home/jxu23/OCEANDIR/precompute_esm_progress.txt"

START_ROW = 0
END_ROW = 100000

COL1 = "chain_1"
COL2 = "chain_2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DTYPE = np.float16
MAX_SEQUENCE_LENGTH = 1000


def valid_id(value) -> bool:
    protein_id = str(value).strip()
    return protein_id != "" and protein_id.lower() != "nan"


def iter_unique_protein_ids(parquet_path: str, start_row, end_row, progress=None):
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows

    start_row = 0 if start_row is None else start_row
    end_row = total_rows if end_row is None else min(end_row, total_rows)

    seen = set()

    for global_row_idx, row in iter_parquet_rows(
        parquet_file, [COL1, COL2], start_row, end_row
    ):
        if progress is not None:
            progress["last_row_processed"] = global_row_idx

        id1 = str(row[COL1]).strip()
        id2 = str(row[COL2]).strip()
        if not valid_id(id1) or not valid_id(id2):
            print(
                f"[row {global_row_idx}] invalid protein ID; skipping pair",
                file=sys.stderr,
            )
            continue

        for protein_id in (id1, id2):
            if protein_id in seen:
                continue
            seen.add(protein_id)
            yield protein_id


def manifest_fields():
    return [
        "id",
        "sequence_length",
        "embedding_shape",
        "embedding_dtype",
        "embedding_path",
        "status",
        "error",
    ]


def write_manifest_row(writer, protein_id, sequence_length, shape, status, error=""):
    writer.writerow(
        {
            "id": protein_id,
            "sequence_length": sequence_length,
            "embedding_shape": "x".join(str(dim) for dim in shape),
            "embedding_dtype": str(np.dtype(SAVE_DTYPE)),
            "embedding_path": embedding_path(EMBEDDING_DIR, protein_id),
            "status": status,
            "error": error,
        }
    )


def save_embedding(protein_id: str, embedding: np.ndarray):
    final_path = embedding_path(EMBEDDING_DIR, protein_id)
    tmp_path = f"{final_path}.tmp"

    embedding = embedding.astype(SAVE_DTYPE)
    with open(tmp_path, "wb") as handle:
        np.save(handle, embedding)

    os.replace(tmp_path, final_path)


def load_sequence(protein_id: str):
    path = pdb_path(PDB_DIR, protein_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    coords, sequence = parse_pdb(path)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {path}")

    return coords, sequence


def compute_and_save_embedding(protein_id: str, model, batch_converter):
    coords, sequence = load_sequence(protein_id)
    if MAX_SEQUENCE_LENGTH is not None and len(sequence) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}"
        )

    embedding = get_esm_embeddings(
        sequence,
        model,
        batch_converter,
        device=DEVICE,
    )

    if len(coords) != len(embedding):
        raise ValueError(
            f"Coordinate/embedding length mismatch: "
            f"{len(coords)} coords vs {len(embedding)} embeddings"
        )

    save_embedding(protein_id, embedding)
    return sequence, embedding.shape


def main():
    args = build_parser(
        "Precompute ESM-2 per-residue embeddings for a parquet row window.",
        START_ROW,
        END_ROW,
    ).parse_args()
    start_row, end_row = args.start_row, args.end_row

    if not os.path.exists(PARQUET_PATH):
        print(f"Error: parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    manifest_dir = os.path.dirname(MANIFEST_CSV)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    print(f"Parquet: {PARQUET_PATH}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    print(f"Rows: [{start_row}, {end_row if end_row is not None else 'EOF'})")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Device: {DEVICE}")
    print(f"Save dtype: {np.dtype(SAVE_DTYPE)}")

    model, _, batch_converter = load_esm(device=DEVICE)

    write_header = not os.path.exists(MANIFEST_CSV)
    processed = 0
    skipped_existing = 0
    failures = []
    progress = {"last_row_processed": None}

    with open(MANIFEST_CSV, "a", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields())
        if write_header:
            writer.writeheader()

        for protein_id in iter_unique_protein_ids(
            PARQUET_PATH, start_row, end_row, progress=progress
        ):
            final_path = embedding_path(EMBEDDING_DIR, protein_id)

            if os.path.exists(final_path) and not args.no_resume:
                skipped_existing += 1
                print(f"{protein_id}: exists, skipping")
                continue

            try:
                sequence, shape = compute_and_save_embedding(
                    protein_id,
                    model,
                    batch_converter,
                )
                processed += 1
                write_manifest_row(
                    writer,
                    protein_id,
                    len(sequence),
                    shape,
                    status="ok",
                )
                print(f"{protein_id}: saved {shape} -> {final_path}")
            except Exception as exc:
                if "out of memory" in str(exc).lower() and DEVICE == "cuda":
                    torch.cuda.empty_cache()

                failures.append(exc)
                write_manifest_row(
                    writer,
                    protein_id,
                    sequence_length=0,
                    shape=(),
                    status="failed",
                    error=str(exc),
                )
                print(f"{protein_id}: failed ({exc})", file=sys.stderr)

            manifest_file.flush()

    print("-" * 60)
    print(f"Done. Saved: {processed}, skipped existing: {skipped_existing}")
    log_failures(failures, processed + len(failures))
    print(f"Last row processed: {progress['last_row_processed']}")
    if progress["last_row_processed"] is not None:
        write_progress(PROGRESS_FILE, progress["last_row_processed"])
    print(f"Embeddings saved to: {os.path.abspath(EMBEDDING_DIR)}")
    print(f"Manifest saved to: {os.path.abspath(MANIFEST_CSV)}")
    run_manifest = write_run_manifest(
        MANIFEST_CSV,
        {
            "stage": "precompute_esm",
            "start_row": start_row,
            "end_row": end_row,
            "embeddings_written": processed,
            "skipped_existing": skipped_existing,
            "failed": len(failures),
            "last_row_processed": progress["last_row_processed"],
        },
    )
    print(f"Run recorded in: {run_manifest}")


if __name__ == "__main__":
    main()
