#!/usr/bin/env python3
"""Foldseek and MMseqs2 searches of SCOPe40, with the benchmark's settings.

    python scope_benchmark/search_baselines.py --foldseek $(which foldseek) --mmseqs $(which mmseqs) --threads 8

Run after prepare_scope.py. The settings are those of foldseek-analysis,
scopbenchmark/scripts/runFoldseek.sh and runMMseqs.sh:

  Foldseek   easy-search on structures/, -s 9.5 --max-seqs 2000 -e 10
  MMseqs2    easy-search on domains.fasta, -s 7.5 --max-seqs 2000 -e 10000

Foldseek reads the structures and is the structure-based reference; MMseqs2
reads the same sequences as the student. Writes hits/foldseek.m8 and
hits/mmseqs.m8, adds their wall times to hits/timing.json, and prints
evaluate_search.py's sensitivities.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scope40 import FASTA, HITS, SCOPE_DIR, STRUCTURES, has_several_models, load_domains  # noqa: E402
from common import log  # noqa: E402
from evaluate_search import evaluate, print_table, query_set  # noqa: E402


def run(command):
    """Wall time of a command; exits with its error output if it fails."""
    log(" ".join(command))
    start = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"{command[0]} failed (exit {done.returncode}):\n{done.stderr[-2000:]}")
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--foldseek", default=None, help="foldseek binary")
    parser.add_argument("--mmseqs", default=None, help="mmseqs binary")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-train-identity", type=float, default=None,
                        help="also report on queries whose closest training protein is below this identity")
    args = parser.parse_args()
    if not args.foldseek and not args.mmseqs:
        parser.error("give --foldseek, --mmseqs or both")
    hits_dir = os.path.join(args.dir, HITS)
    os.makedirs(hits_dir, exist_ok=True)
    threads = ["--threads", str(args.threads)]

    jobs = {}
    if args.foldseek:
        structures = os.path.join(args.dir, STRUCTURES)
        ensembles = [name for name in sorted(os.listdir(structures))
                     if has_several_models(os.path.join(structures, name))]
        if ensembles:
            raise SystemExit(f"{len(ensembles):,} structures hold several models (e.g. {ensembles[0]}), and "
                             f"Foldseek would search each as a domain of its own; rerun prepare_scope.py, "
                             f"which keeps the first")
        jobs["foldseek"] = [args.foldseek, "easy-search", os.path.join(args.dir, STRUCTURES),
                            os.path.join(args.dir, STRUCTURES), None, None,
                            "-s", "9.5", "--max-seqs", "2000", "-e", "10", *threads]
    if args.mmseqs:
        jobs["mmseqs"] = [args.mmseqs, "easy-search", os.path.join(args.dir, FASTA),
                          os.path.join(args.dir, FASTA), None, None,
                          "-s", "7.5", "--max-seqs", "2000", "-e", "10000", *threads]

    timing_path = os.path.join(hits_dir, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    paths = {}
    for name, command in jobs.items():
        paths[name] = os.path.join(hits_dir, f"{name}.m8")
        work = tempfile.mkdtemp(prefix=f"{name}_", dir=hits_dir)
        command[4:6] = [paths[name], work]
        timing[name] = {"threads": args.threads, "search_s": run(command)}
        shutil.rmtree(work)
    with open(timing_path, "w") as handle:
        json.dump(timing, handle, indent=2)

    indexed = load_domains(args.dir).set_index("domain")
    log("")
    print_table({name: evaluate(path, indexed)[0] for name, path in paths.items()})
    queries = query_set(indexed, args.max_train_identity)
    if queries is not None:
        log(f"\nqueries whose closest training protein is below {args.max_train_identity:.0%} identity")
        print_table({name: evaluate(path, indexed, queries)[0] for name, path in paths.items()})


if __name__ == "__main__":
    main()
