#!/usr/bin/env python3
"""Precompute ESM-2 per-residue embeddings.

    python data_scripts/precompute_esm.py --start-row 0 --end-row 1000000     # 35M, parquet window
    python data_scripts/precompute_esm.py --model 650M --part 0/4 \
        --proteins-in /jet/home/jxu23/OCEANDIR/fgw_scores.csv \
        --sequences-from /jet/home/jxu23/OCEANDIR/tm_scores.csv

--model picks the checkpoint (common.ESM_MODELS). 35M writes to embeddings/,
which the teacher, the labels and the pack depend on; any other model writes
to its own embeddings_<model>/ and refuses to touch embeddings/.

--proteins-in embeds every protein of a CSV with id1/id2 columns (fgw_scores.csv:
exactly the proteins training and evaluation read) instead of a parquet
window; --protein-list embeds the ids listed in a text file. --part K/N keeps
every N-th of them starting at K, so N jobs can split the work without
overlap. --sequences-from takes each sequence
from tm_scores.csv, which tm_data.py read with the same parse_pdb, instead of
parsing every PDB again (~170 ms each on Ocean); a sequence whose length
disagrees with the protein pack is re-read from its PDB.
"""

import csv
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

from common import (
    BASE_ESM,
    DATA_DIR,
    ESM_MODELS,
    build_parser,
    esm_dirs,
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

ESM_MODEL = BASE_ESM  # set from --model
SEQUENCES = {}  # protein id -> sequence, from --sequences-from
PACK_LENGTHS = {}  # protein id -> residue count in the 35M protein pack


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
    """(residue count the embedding must match, sequence)."""
    sequence = SEQUENCES.get(protein_id)
    expected = PACK_LENGTHS.get(protein_id)
    if sequence and (expected is None or expected == len(sequence)):
        return expected if expected is not None else len(sequence), sequence

    path = pdb_path(PDB_DIR, protein_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    coords, sequence = parse_pdb(path)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {path}")

    return len(coords), sequence


def load_sequences(csv_path, wanted=None):
    """{protein id: sequence} from tm_scores.csv, limited to `wanted` if given."""
    import pandas as pd

    sequences = {}
    for chunk in pd.read_csv(
        csv_path, usecols=["id1", "id2", "seq1", "seq2"], chunksize=200000,
        dtype=str, keep_default_na=False,
    ):
        for id_column, seq_column in (("id1", "seq1"), ("id2", "seq2")):
            for protein_id, sequence in zip(chunk[id_column].str.strip(), chunk[seq_column]):
                if wanted is None or protein_id in wanted:
                    sequences.setdefault(protein_id, sequence)
    return sequences


def read_protein_list(path):
    with open(path) as handle:
        ids = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return list(dict.fromkeys(ids))  # unique, in file order


def compute_and_save_embedding(protein_id: str, model, batch_converter):
    expected_length, sequence = load_sequence(protein_id)
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

    if expected_length != len(embedding):
        raise ValueError(
            f"Coordinate/embedding length mismatch: "
            f"{expected_length} coords vs {len(embedding)} embeddings"
        )

    save_embedding(protein_id, embedding)
    return sequence, embedding.shape


def main():
    global EMBEDDING_DIR, MANIFEST_CSV, ESM_MODEL, SEQUENCES, PACK_LENGTHS

    parser = build_parser(
        "Precompute ESM-2 per-residue embeddings for a parquet row window or a protein list.",
        START_ROW,
        END_ROW,
    )
    parser.add_argument("--model", default=BASE_ESM, choices=sorted(ESM_MODELS))
    parser.add_argument("--embedding-dir", default=None, help="default: by model (common.esm_dirs)")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--protein-list", default=None, help="one protein id per line")
    parser.add_argument("--proteins-in", default=None, help="every protein of this CSV's id1/id2")
    parser.add_argument("--part", default=None, help="K/N: every N-th protein from the K-th")
    parser.add_argument("--sequences-from", default=None, help="tm_scores.csv: skip PDB parsing")
    args = parser.parse_args()
    start_row, end_row = args.start_row, args.end_row

    ESM_MODEL = args.model
    base_embedding_dir, base_pack_dir = esm_dirs(BASE_ESM)
    EMBEDDING_DIR = args.embedding_dir or esm_dirs(ESM_MODEL)[0]
    if ESM_MODEL != BASE_ESM:
        MANIFEST_CSV = args.manifest or os.path.join(DATA_DIR, f"esm_manifest_{ESM_MODEL}.csv")
        if os.path.abspath(EMBEDDING_DIR) == os.path.abspath(base_embedding_dir):
            print(
                f"Error: {EMBEDDING_DIR} holds the {BASE_ESM} embeddings the pipeline "
                f"depends on; {ESM_MODEL} embeddings must go elsewhere",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.manifest:
        MANIFEST_CSV = args.manifest

    protein_ids = None
    if args.protein_list:
        protein_ids = read_protein_list(args.protein_list)
    elif args.proteins_in:
        from protein_pack import proteins_in_csv

        protein_ids = sorted(proteins_in_csv(args.proteins_in))
    if args.part:
        if protein_ids is None:
            print("Error: --part needs --protein-list or --proteins-in", file=sys.stderr)
            sys.exit(1)
        k, n = (int(value) for value in args.part.split("/"))
        protein_ids = protein_ids[k::n]
    if args.sequences_from:
        wanted = set(protein_ids) if protein_ids is not None else None
        SEQUENCES = load_sequences(args.sequences_from, wanted)
        print(f"Sequences: {len(SEQUENCES):,} from {args.sequences_from}")
        if os.path.isdir(base_pack_dir):
            from protein_pack import load_index

            PACK_LENGTHS = {pid: length for pid, (_, _, length) in load_index(base_pack_dir).items()}
            print(f"Lengths checked against {base_pack_dir} ({len(PACK_LENGTHS):,} proteins)")

    if protein_ids is None and not os.path.exists(PARQUET_PATH):
        print(f"Error: parquet file not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    manifest_dir = os.path.dirname(MANIFEST_CSV)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)

    print(f"Model: {ESM_MODELS[ESM_MODEL][0]} ({ESM_MODELS[ESM_MODEL][1]} dimensions)")
    if protein_ids is not None:
        source_name = args.protein_list or args.proteins_in
        part = f", part {args.part}" if args.part else ""
        print(f"Proteins: {len(protein_ids):,} from {source_name}{part}")
    else:
        print(f"Parquet: {PARQUET_PATH}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Manifest: {MANIFEST_CSV}")
    print(f"Progress file: {PROGRESS_FILE}")
    if protein_ids is None:
        print(f"Rows: [{start_row}, {end_row if end_row is not None else 'EOF'})")
    print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"Device: {DEVICE}")
    print(f"Save dtype: {np.dtype(SAVE_DTYPE)}")

    model, _, batch_converter = load_esm(ESM_MODELS[ESM_MODEL][0], device=DEVICE)

    write_header = not os.path.exists(MANIFEST_CSV)
    processed = 0
    skipped_existing = 0
    failures = []
    progress = {"last_row_processed": None}

    with open(MANIFEST_CSV, "a", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields())
        if write_header:
            writer.writeheader()

        source = (
            protein_ids
            if protein_ids is not None
            else iter_unique_protein_ids(PARQUET_PATH, start_row, end_row, progress=progress)
        )
        for protein_id in source:
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
    # only a parquet window has a row to resume from, and only the 35M stage tracks it
    if progress["last_row_processed"] is not None and ESM_MODEL == BASE_ESM:
        write_progress(PROGRESS_FILE, progress["last_row_processed"])
    print(f"Embeddings saved to: {os.path.abspath(EMBEDDING_DIR)}")
    print(f"Manifest saved to: {os.path.abspath(MANIFEST_CSV)}")
    run_manifest = write_run_manifest(
        MANIFEST_CSV,
        {
            "stage": "precompute_esm",
            "model": ESM_MODELS[ESM_MODEL][0],
            "proteins": args.protein_list or args.proteins_in,
            "part": args.part,
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
