"""Small helpers shared across the data scripts."""

import argparse
import datetime
import json
import os


def pdb_path(pdb_dir, uniprot_id):
    return os.path.join(pdb_dir, f"{uniprot_id}.pdb")


def embedding_path(embedding_dir, uniprot_id):
    return os.path.join(embedding_dir, f"{uniprot_id}.npy")


def write_progress(progress_file, row_idx):
    """Record the last row processed, so a shard can be resumed."""
    progress_dir = os.path.dirname(progress_file)
    if progress_dir:
        os.makedirs(progress_dir, exist_ok=True)

    with open(progress_file, "w") as handle:
        handle.write(str(row_idx))


def read_done_rows(csv_path, column):
    """Source rows already present in an output file, for resume.

    Reads only the one column: tm_scores.csv carries full sequences, so
    parsing every field costs about a gigabyte of string work per startup.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    import pandas as pd

    try:
        values = pd.read_csv(csv_path, usecols=[column], keep_default_na=False)[column]
    except ValueError:  # column absent
        return set()

    values = pd.to_numeric(values, errors="coerce").dropna()
    return set(values.astype(int).tolist())


def log(msg):
    print(msg, flush=True)


def add_shard_args(parser, start_default, end_default):
    """The row window every stage shares."""
    parser.add_argument("--start-row", type=int, default=start_default)
    parser.add_argument(
        "--end-row",
        type=lambda v: None if v.lower() in ("none", "eof", "") else int(v),
        default=end_default,
        help="exclusive; 'none' for end of input",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="reprocess entries already present in the output (default: skip them)",
    )
    return parser


def build_parser(description, start_default, end_default):
    parser = argparse.ArgumentParser(description=description)
    return add_shard_args(parser, start_default, end_default)


def iter_row_groups(parquet_file, start_row, end_row):
    """Yield (row_group_index, first_global_row) for groups overlapping the window."""
    offset = 0
    for index in range(parquet_file.num_row_groups):
        num_rows = parquet_file.metadata.row_group(index).num_rows
        if offset + num_rows <= start_row:
            offset += num_rows
            continue
        if end_row is not None and offset >= end_row:
            return
        yield index, offset
        offset += num_rows


def iter_parquet_rows(parquet_file, columns, start_row, end_row, batch_size=65536):
    """Yield (global_row_index, {column: value}) for rows in [start_row, end_row).

    Streams Arrow batches rather than materialising a whole row group in
    pandas. The source parquet has 64M-row groups, so the old approach cost
    several GB per process, once per shard.
    """
    start_row = 0 if start_row is None else start_row

    for group_index, group_start in iter_row_groups(parquet_file, start_row, end_row):
        offset = group_start
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, row_groups=[group_index], columns=list(columns)
        ):
            batch_end = offset + batch.num_rows
            if batch_end <= start_row:
                offset = batch_end
                continue
            if end_row is not None and offset >= end_row:
                return

            values = batch.to_pydict()
            for local_index in range(batch.num_rows):
                global_row = offset + local_index
                if global_row < start_row:
                    continue
                if end_row is not None and global_row >= end_row:
                    return
                yield global_row, {name: values[name][local_index] for name in columns}
            offset = batch_end


def format_superposition(translation, rotation):
    """TM-align's superposition as 12 floats: t (3), then u row-major (9).

    Chain 1 maps onto chain 2 as y ~ u @ x + t.
    """
    values = [float(v) for v in translation] + [float(v) for row in rotation for v in row]
    if len(values) != 12:
        raise ValueError(f"expected 3 + 9 values, got {len(values)}")
    return " ".join(f"{v:.6f}" for v in values)


def parse_superposition(text):
    """Inverse of format_superposition: returns (translation (3,), rotation (3, 3))."""
    import numpy as np

    values = np.array(str(text).split(), dtype=np.float64)
    if values.size != 12:
        raise ValueError(f"superposition needs 12 numbers, got {values.size}: {text!r}")
    return values[:3], values[3:].reshape(3, 3)


def write_run_manifest(output_path, record):
    """Append one JSON line describing this run, beside its output."""
    manifest_path = f"{output_path}.runs.jsonl"
    record = dict(record)
    record.setdefault(
        "finished_at", datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    with open(manifest_path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return manifest_path


def summarise_failures(failures):
    """Turn a list of exceptions into a 'why did I lose rows' histogram."""
    counts = {}
    for exc in failures:
        counts[type(exc).__name__] = counts.get(type(exc).__name__, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def log_failures(failures, total):
    if not failures:
        log("  no failures")
        return
    log(f"  failures: {len(failures)} of {total}")
    for name, count in summarise_failures(failures).items():
        log(f"    {name:<28} {count}")


def open_appending_writer(csv_path, fieldnames):
    """Append-mode CSV writer that writes a header only for a genuinely new file."""
    import csv

    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    handle = open(csv_path, "a", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        handle.flush()
    return handle, writer
