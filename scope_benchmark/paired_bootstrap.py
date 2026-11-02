#!/usr/bin/env python3
"""Whether one method's SCOPe40 sensitivity beats another's by more than chance.

    python scope_benchmark/paired_bootstrap.py $D/scope40/per_query_heldout.tsv
    python scope_benchmark/paired_bootstrap.py per_query_heldout.tsv --pair student_esm2_sw esm2_650M_sw

Reads a per-query file from evaluate_search.py --per-query. Both methods were
scored on the same queries, so the queries are resampled together: each of
--resamples draws takes the queries with replacement and averages A - B over
them. The 95 % interval runs from the 2.5th to the 97.5th percentile of those
averages; one that excludes 0 (marked *) is a difference the queries support.
"wins" counts the queries where each method scores higher.

Without --pair, the comparisons that tell the stage-0 story, for the
student's hit-file prefix --name and the ESM-2 model the file holds:

  <name>_esm2_sw            - esm2_<model>_sw            the best method against ESM-2 alone
  <name>_esm2_sw            - esm2_<model>_sw_on_<name>  the student's matrix added, same candidates
  esm2_<model>_sw_on_<name> - esm2_<model>_sw            the student's candidates against ESM-2's
  <name>_sw                 - esm2_<model>_sw_on_<name>  the student's matrix against ESM-2's
  teacher_sw_on_<name>      - <name>_sw                  the teacher's matrix against the student's
  <name>_tm                 - esm2_<model>_pooled        the global scores, before alignment

Pairs with a method missing from the file are skipped with a note.
"""

import argparse

import numpy as np
import pandas as pd

LEVELS = ("family", "superfamily", "fold")


def methods(table):
    """The methods in a per-query file, whose columns are <method>_<level>."""
    return sorted({column.rsplit("_", 1)[0] for column in table.columns if column.rsplit("_", 1)[-1] in LEVELS})


def standard_pairs(available, name):
    pooled = [m for m in available if m.startswith("esm2_") and m.endswith("_pooled")]
    esm = pooled[0][: -len("_pooled")] if pooled else "esm2_650M"
    return [
        (f"{name}_esm2_sw", f"{esm}_sw", "the best method against ESM-2 alone"),
        (f"{name}_esm2_sw", f"{esm}_sw_on_{name}", "the student's matrix added, same candidates"),
        (f"{esm}_sw_on_{name}", f"{esm}_sw", "the student's candidates against ESM-2's"),
        (f"{name}_sw", f"{esm}_sw_on_{name}", "the student's matrix against ESM-2's"),
        (f"teacher_sw_on_{name}", f"{name}_sw", "the teacher's matrix against the student's"),
        (f"{name}_tm", f"{esm}_pooled", "the global scores, before alignment"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("per_query", help="evaluate_search.py --per-query output")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="compare A - B; repeatable")
    parser.add_argument("--name", default="student", help="search_student.py's --name, for the default pairs")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = pd.read_csv(args.per_query, sep="\t", index_col="domain")
    available = methods(table)
    pairs = [(a, b, "") for a, b in args.pair] if args.pair else standard_pairs(available, args.name)
    # one set of draws for every comparison, so all of them see the same resampled queries
    draws = np.random.default_rng(args.seed).integers(0, len(table), size=(args.resamples, len(table)),
                                                     dtype=np.int32)
    print(f"{len(table):,} queries from {args.per_query}, {args.resamples:,} resamples")
    for a, b, about in pairs:
        missing = [m for m in (a, b) if m not in available]
        if missing:
            print(f"\nskipped {a} - {b}: {', '.join(missing)} not in the file")
            continue
        print(f"\n{a} - {b}" + (f"  ({about})" if about else ""))
        print(f"  {'level':<11}  {'A':>6}  {'B':>6}  {'A - B':>7}  {'95% interval':>17}  {'wins A/B':>9}")
        for level in LEVELS:
            x, y = table[f"{a}_{level}"].to_numpy(), table[f"{b}_{level}"].to_numpy()
            difference = x - y
            low, high = np.percentile(difference[draws].mean(axis=1), [2.5, 97.5])
            mark = "*" if low > 0 or high < 0 else " "
            print(f"  {level:<11}  {x.mean():>6.3f}  {y.mean():>6.3f}  {difference.mean():>+7.3f}  "
                  f"[{low:+.3f}, {high:+.3f}]{mark}  {int((x > y).sum()):>4}/{int((x < y).sum()):<4}")


if __name__ == "__main__":
    main()
