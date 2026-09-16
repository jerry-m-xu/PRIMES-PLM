#!/usr/bin/env python3
"""Build the sequence-cluster map that splits.py uses to hold out homologues.

    python data_scripts/cluster_proteins.py            # after tm_data.py has run

Reads every protein and its sequence from tm_scores.csv, writes a FASTA, runs
MMseqs2 easy-cluster at MIN_SEQ_ID identity and MIN_COVERAGE coverage, and
copies the resulting representative<TAB>member table to splits.CLUSTER_TSV.
Adding proteins later only adds rows; an existing protein keeps its
representative as long as the FASTA is a superset, so splits stay stable.

MMseqs2 is an external binary: `conda install -c bioconda mmseqs2`, or the
cluster's module system.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import log  # noqa: E402
from splits import CLUSTER_TSV  # noqa: E402

TM_CSV = "/jet/home/jxu23/OCEANDIR/tm_scores.csv"
MIN_SEQ_ID = 0.3
MIN_COVERAGE = 0.8
CHUNK_SIZE = 20_000


def collect_sequences(csv_path):
    sequences = {}
    for chunk in pd.read_csv(
        csv_path,
        usecols=["id1", "id2", "seq1", "seq2"],
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        for id_col, seq_col in (("id1", "seq1"), ("id2", "seq2")):
            for protein_id, sequence in zip(chunk[id_col], chunk[seq_col]):
                protein_id = str(protein_id).strip()
                if protein_id and protein_id not in sequences and sequence:
                    sequences[protein_id] = str(sequence)
    return sequences


def write_fasta(sequences, path):
    with open(path, "w") as handle:
        for protein_id, sequence in sequences.items():
            handle.write(f">{protein_id}\n{sequence}\n")


def run_mmseqs(fasta, workdir, min_seq_id, coverage, threads):
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None:
        raise FileNotFoundError(
            "mmseqs not on PATH; install MMseqs2 (conda install -c bioconda mmseqs2)"
        )
    prefix = os.path.join(workdir, "clusters")
    command = [
        mmseqs, "easy-cluster", fasta, prefix, os.path.join(workdir, "tmp"),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
        "--threads", str(threads),
    ]
    log("  " + " ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return f"{prefix}_cluster.tsv"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tm-csv", default=TM_CSV)
    parser.add_argument("--output", default=CLUSTER_TSV)
    parser.add_argument("--min-seq-id", type=float, default=MIN_SEQ_ID)
    parser.add_argument("--coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    log(f"reading sequences from {args.tm_csv}")
    sequences = collect_sequences(args.tm_csv)
    log(f"  {len(sequences):,} proteins")

    with tempfile.TemporaryDirectory(prefix="mmseqs_") as workdir:
        fasta = os.path.join(workdir, "proteins.fasta")
        write_fasta(sequences, fasta)
        log(f"clustering at {args.min_seq_id:.0%} identity, {args.coverage:.0%} coverage")
        cluster_tsv = run_mmseqs(fasta, workdir, args.min_seq_id, args.coverage, args.threads)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(cluster_tsv, args.output + ".tmp")
        os.replace(args.output + ".tmp", args.output)

    representatives = set()
    members = 0
    with open(args.output) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                representatives.add(parts[0])
                members += 1
    log(f"wrote {args.output}: {members:,} proteins in {len(representatives):,} clusters")
    log(f"  mean cluster size {members / max(len(representatives), 1):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
