#!/usr/bin/env python3
"""Local structural labels for TM-aligned protein pairs.

For every protein pair in tm_scores.csv, sample residue pairs and score each
with patch-level Gromov-Wasserstein (see fgw.py):

  aligned   every ALIGNED_RESIDUE_STRIDE-th TM-align-aligned residue pair
  shifted   the same residue of protein 1 matched SHIFT_RANGE residues away
            along protein 2 (a hard negative: its patch overlaps the true one)
  random    a uniformly random residue pair

NEGATIVES_PER_POSITIVE off-path pairs are written per aligned pair,
alternating shifted and random. Without them every label sits on TM-align's
path and nothing teaches a model that non-corresponding residues are
dissimilar.

Each row also carries the TM-align superposed CA distance and the per-pair
TM term 1 / (1 + (d / d0)^2). Unlike GW it is chirality-aware, and it
separates "displaced but locally intact" from "locally different".
"""

import os
import sys

import numpy as np
import pandas as pd

from common import (
    build_parser,
    embedding_path,
    log_failures,
    open_appending_writer,
    parse_superposition,
    pdb_path,
    read_done_rows,
    write_progress,
    write_run_manifest,
)
from fgw import (
    ALPHA,
    DIST_SCALE,
    EPS,
    INNER_ITER,
    OUTER_ITER,
    compute_fgw_from_features,
    compute_structure_gw,
)
from parse_pdb import parse_pdb
from patches import K_NEIGHBORS, knn_indices

# global config
TM_DATA_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
OUTPUT_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"
PROGRESS_FILE = "/jet/home/jxu23/OCEANDIR/fgw_data_progress.txt"

START_ROW = 0
END_ROW = 50000
CSV_CHUNK_SIZE = 1000
FLUSH_EVERY_ROWS = 10
ALIGNED_RESIDUE_STRIDE = 32  # K_NEIGHBORS now comes from patches.py
NEGATIVES_PER_POSITIVE = 1
SHIFT_RANGE = (4, 32)  # |offset| along protein 2 for a "shifted" negative
# solver settings (DIST_SCALE, EPS, ALPHA, iteration counts) live in fgw.py

PAIR_TYPES = ("aligned", "shifted", "random")
SEQM_VALUES = {":", ".", " "}
REQUIRED_TM_COLUMNS = (
    "id1", "id2", "tm_score_norm1", "tm_score_norm2",
    "seqxA", "seqM", "seqyA", "superposition",
)


def compute_local_labels(
    coords1: np.ndarray,
    coords2: np.ndarray,
    features1: np.ndarray,
    features2: np.ndarray,
):
    """Raw distortions for one residue pair: (gw_raw, fgw_raw, fgw_feature_term).

    gw_raw is the structure label's source: pure GW, features not consulted.
    fgw_raw is the fused objective at its own coupling, kept as a secondary
    score. Both are raw; pair_data.py maps them to exp(-raw / scale).
    """
    gw_raw = compute_structure_gw(
        coords1, coords2, eps=EPS, outer_iter=OUTER_ITER, inner_iter=INNER_ITER
    )
    fgw_raw, _, feature_term = compute_fgw_from_features(
        coords1,
        coords2,
        features1,
        features2,
        alpha=ALPHA,
        eps=EPS,
        outer_iter=OUTER_ITER,
        inner_iter=INNER_ITER,
        return_components=True,
    )
    return float(gw_raw), float(fgw_raw), float(feature_term)


def tm_d0(length):
    """TM-align's distance scale for a normalising length."""
    if length <= 21:
        return 0.5
    return max(0.5, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def tm_term(distance, d0):
    """One residue pair's contribution to TM-score, in (0, 1]."""
    return 1.0 / (1.0 + (distance / d0) ** 2)


def iter_aligned_residue_pairs(seqxA: str, seqM: str, seqyA: str):
    if not (len(seqxA) == len(seqM) == len(seqyA)):
        raise ValueError("seqxA, seqM, and seqyA must have the same length")

    idx1 = 0
    idx2 = 0

    for align_pos, (aa1, marker, aa2) in enumerate(zip(seqxA, seqM, seqyA)):
        residue_idx1 = idx1 if aa1 != "-" else None
        residue_idx2 = idx2 if aa2 != "-" else None

        if aa1 != "-":
            idx1 += 1
        if aa2 != "-":
            idx2 += 1

        if marker not in SEQM_VALUES:
            continue
        if residue_idx1 is None or residue_idx2 is None:
            continue

        yield align_pos, residue_idx1, residue_idx2, aa1, marker, aa2


def shifted_pair(residue_idx1, residue_idx2, length2, rng):
    """(i, j +/- shift), or None if the protein is too short for the shift."""
    shift = int(rng.integers(SHIFT_RANGE[0], SHIFT_RANGE[1] + 1))
    shift *= int(rng.choice((-1, 1)))
    for candidate in (residue_idx2 + shift, residue_idx2 - shift):
        if 0 <= candidate < length2:
            return residue_idx1, candidate
    return None


def random_pair(length1, length2, on_path, rng):
    pair = None
    for _ in range(10):
        pair = int(rng.integers(length1)), int(rng.integers(length2))
        if pair not in on_path:
            return pair
    return pair


def sample_residue_pairs(aligned_pairs, length1, length2, rng):
    """Strided positives plus off-path negatives.

    aligned_pairs: iter_aligned_residue_pairs() output, in alignment order.
    Returns (pair_type, align_pos, residue_idx1, residue_idx2, marker); the
    negatives carry align_pos -1 and marker "-".
    """
    on_path = {(i, j) for _, i, j, _, _, _ in aligned_pairs}
    sampled = []
    negatives = 0

    for k, (align_pos, i, j, _, marker, _) in enumerate(aligned_pairs):
        if k % ALIGNED_RESIDUE_STRIDE != 0:
            continue
        sampled.append(("aligned", align_pos, i, j, marker))

        for _ in range(NEGATIVES_PER_POSITIVE):
            kind = "shifted" if negatives % 2 == 0 else "random"
            pair = shifted_pair(i, j, length2, rng) if kind == "shifted" else None
            if pair is None:
                kind = "random"
                pair = random_pair(length1, length2, on_path, rng)
            sampled.append((kind, -1, pair[0], pair[1], "-"))
            negatives += 1

    return sampled


def load_protein_data(uniprot_id: str):
    pdb_file = pdb_path(PDB_DIR, uniprot_id)
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(pdb_file)

    embedding_file = embedding_path(EMBEDDING_DIR, uniprot_id)
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(embedding_file)

    coords, sequence = parse_pdb(pdb_file)
    if len(sequence) == 0:
        raise ValueError(f"No CA atoms found in {pdb_file}")

    embeddings = np.load(embedding_file).astype(np.float32)
    if len(coords) != len(embeddings):
        raise ValueError(
            f"Coordinate/embedding length mismatch for {uniprot_id}: "
            f"{len(coords)} coords vs {len(embeddings)} embeddings"
        )

    return coords, sequence, embeddings


def input_rows(csv_path: str, start_row, end_row, skip_source_rows=frozenset()):
    """Stream tm_scores.csv rows whose SOURCE parquet row is in the window."""
    start_row = 0 if start_row is None else start_row
    line_no = 0
    # tm_scores.csv itself contains duplicated source rows from overlapping
    # shard runs. skip_source_rows is read once at startup, so without this a
    # repeated input row is scored twice within a single run.
    emitted = set()

    for df_chunk in pd.read_csv(
        csv_path,
        chunksize=CSV_CHUNK_SIZE,
        keep_default_na=False,
    ):
        for _, row in df_chunk.iterrows():
            source_row = int(row["row"]) if "row" in row else line_no
            line_no += 1

            if source_row < start_row:
                continue
            if end_row is not None and source_row >= end_row:
                continue
            if source_row in skip_source_rows or source_row in emitted:
                continue

            emitted.add(source_row)
            yield source_row, row


def output_fields():
    return [
        "tm_data_row",
        "source_row",
        "id1",
        "id2",
        "tm_score_norm1",
        "tm_score_norm2",
        "pair_type",
        "align_pos",
        "residue_idx1",
        "residue_idx2",
        "aa1",
        "seqM",
        "aa2",
        "superposed_dist",
        "tm_term",
        "gw_raw",
        "fgw_raw",
        "fgw_feature_term",
        "neighborhood_size1",
        "neighborhood_size2",
    ]


def write_result(writer, row, input_row_idx, record):
    source_row = row["row"] if "row" in row else input_row_idx
    writer.writerow(
        {
            "tm_data_row": input_row_idx,
            "source_row": source_row,
            "id1": row["id1"],
            "id2": row["id2"],
            "tm_score_norm1": row["tm_score_norm1"],
            "tm_score_norm2": row["tm_score_norm2"],
            **record,
        }
    )


def process_tm_row(row, input_row_idx, writer):
    id1 = str(row["id1"]).strip()
    id2 = str(row["id2"]).strip()

    coords1, sequence1, embeddings1 = load_protein_data(id1)
    coords2, sequence2, embeddings2 = load_protein_data(id2)

    aligned_pairs = list(
        iter_aligned_residue_pairs(str(row["seqxA"]), str(row["seqM"]), str(row["seqyA"]))
    )
    source_row = int(row["row"]) if "row" in row else int(input_row_idx)
    rng = np.random.default_rng(source_row)
    residue_pairs = sample_residue_pairs(aligned_pairs, len(coords1), len(coords2), rng)

    translation, rotation = parse_superposition(row["superposition"])
    superposed1 = coords1.astype(np.float64) @ rotation.T + translation
    d0 = tm_d0(len(coords1))

    pending = []
    for pair_type, align_pos, residue_idx1, residue_idx2, marker in residue_pairs:
        indices1 = knn_indices(coords1, residue_idx1)
        indices2 = knn_indices(coords2, residue_idx2)

        gw_raw, fgw_raw, feature_term = compute_local_labels(
            coords1[indices1],
            coords2[indices2],
            embeddings1[indices1],
            embeddings2[indices2],
        )
        distance = float(np.linalg.norm(superposed1[residue_idx1] - coords2[residue_idx2]))

        pending.append(
            {
                "pair_type": pair_type,
                "align_pos": align_pos,
                "residue_idx1": residue_idx1,
                "residue_idx2": residue_idx2,
                "aa1": sequence1[residue_idx1],
                "seqM": marker,
                "aa2": sequence2[residue_idx2],
                "superposed_dist": distance,
                "tm_term": tm_term(distance, d0),
                "gw_raw": gw_raw,
                "fgw_raw": fgw_raw,
                "fgw_feature_term": feature_term,
                "neighborhood_size1": len(indices1),
                "neighborhood_size2": len(indices2),
            }
        )

    for record in pending:
        write_result(writer, row, input_row_idx, record)

    return len(pending)


def check_input_columns(csv_path):
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [name for name in REQUIRED_TM_COLUMNS if name not in header]
    if missing:
        print(
            f"Error: {csv_path} lacks columns {missing}. Regenerate it with the "
            f"current tm_data.py (it stores both TM normalisations and the "
            f"superposition).",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = build_parser(
        "Compute local FGW scores for TM-aligned pairs in a parquet row window.",
        START_ROW,
        END_ROW,
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="read already-completed rows from here instead of --output; lets "
             "array tasks share one resume set while writing separate files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write here instead of OUTPUT_CSV; give each array task its own "
             "file, since concurrent appends to one CSV interleave",
    )
    args = parser.parse_args()
    start_row, end_row = args.start_row, args.end_row
    output_csv = args.output or OUTPUT_CSV
    resume_csv = args.resume_from or output_csv

    if not os.path.exists(TM_DATA_CSV):
        print(f"Error: TM data CSV not found: {TM_DATA_CSV}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(PDB_DIR):
        print(f"Error: PDB directory not found: {PDB_DIR}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(EMBEDDING_DIR):
        print(
            f"Error: embedding directory not found: {EMBEDDING_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    check_input_columns(TM_DATA_CSV)

    print(f"TM data: {TM_DATA_CSV}")
    print(f"PDB dir: {PDB_DIR}")
    print(f"Embedding dir: {EMBEDDING_DIR}")
    print(f"Source rows: [{start_row}, {end_row if end_row is not None else 'EOF'})")
    print(f"CSV chunk size: {CSV_CHUNK_SIZE}")
    print(f"Flush every rows: {FLUSH_EVERY_ROWS}")
    print(f"k-NN size: {K_NEIGHBORS}")
    print(f"Aligned residue stride: {ALIGNED_RESIDUE_STRIDE}")
    print(f"Negatives per positive: {NEGATIVES_PER_POSITIVE} (shift range {SHIFT_RANGE})")
    print(f"Distance scale: {DIST_SCALE} A, eps: {EPS}, fused alpha: {ALPHA}")
    print(f"GW iterations: {OUTER_ITER} outer x {INNER_ITER} Sinkhorn")
    print(f"Output: {output_csv}")

    skip_source_rows = (
        frozenset() if args.no_resume else read_done_rows(resume_csv, "source_row")
    )
    if skip_source_rows:
        print(f"Resume: {len(skip_source_rows)} pairs already scored, skipping them")

    total_pairs = 0
    failures = []
    rows_since_flush = 0
    last_row_processed = None

    output_file, writer = open_appending_writer(output_csv, output_fields())
    with output_file:
        for input_row_idx, row in input_rows(
            TM_DATA_CSV, start_row, end_row, skip_source_rows
        ):
            try:
                scored_pairs = process_tm_row(
                    row,
                    input_row_idx,
                    writer,
                )
                total_pairs += scored_pairs
                print(
                    f"[row {input_row_idx}] {row['id1']} vs {row['id2']}: "
                    f"{scored_pairs} local labels"
                )
                rows_since_flush += 1
                if rows_since_flush >= FLUSH_EVERY_ROWS:
                    output_file.flush()
                    write_progress(PROGRESS_FILE, input_row_idx)
                    rows_since_flush = 0
            except Exception as exc:
                failures.append(exc)
                print(
                    f"[row {input_row_idx}] skipped ({exc})",
                    file=sys.stderr,
                )
            finally:
                last_row_processed = input_row_idx

        output_file.flush()
        if last_row_processed is not None:
            write_progress(PROGRESS_FILE, last_row_processed)

    print("-" * 60)
    print(f"Done. Local labels: {total_pairs}")
    log_failures(failures, total_pairs + len(failures))
    print(f"Results saved to: {os.path.abspath(output_csv)}")
    manifest = write_run_manifest(
        output_csv,
        {
            "stage": "fgw_data",
            "start_row": start_row,
            "end_row": end_row,
            "scores_written": total_pairs,
            "failed": len(failures),
            "resumed_skips": len(skip_source_rows),
            "last_row_processed": last_row_processed,
        },
    )
    print(f"Run recorded in: {manifest}")


if __name__ == "__main__":
    main()
