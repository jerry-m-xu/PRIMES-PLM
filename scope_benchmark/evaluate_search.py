#!/usr/bin/env python3
"""Sensitivity up to the first false positive on SCOPe40, for ranked hit lists.

    python scope_benchmark/evaluate_search.py $D/scope40/hits/*.m8 $D/scope40/hits/*.tsv.gz
    python scope_benchmark/evaluate_search.py student=hits/student_tm.tsv.gz foldseek=hits/foldseek.m8
    python scope_benchmark/evaluate_search.py hits/* --max-train-identity 0.3 --per-query per_query.tsv

Any tab-separated file whose first two columns are query and target works:
Foldseek's and MMseqs2's output (.m8), and search_student.py's (.tsv.gz).
Each query's hits must be in rank order, best first, as all three write them.

This is Foldseek's measure (foldseek-analysis, scopbenchmark/scripts/
bench.noselfhit.awk), reproduced so that the numbers compare with the
Foldseek paper's:
  - Hits are read in file order. Self hits are skipped.
  - A query's first hit to another fold is its first false positive, and so is
    a hit to a domain outside the benchmark; nothing after it counts.
  - Before it, a hit in the query's family is a family true positive; in its
    superfamily but another family, a superfamily one; in its fold but another
    superfamily, a fold one.
  - Each level's sensitivity is the true positives found divided by the
    awk script's denominator. Only queries with something to find at every
    level count: a family with another member, a superfamily with another
    family, a fold with another superfamily. A query without hits scores zero.
The reported numbers are means over those queries.

--max-train-identity keeps only queries whose closest training protein is
below that identity (prepare_scope.py measures it). The database stays
whole: a search database holds known proteins, so only the query needs to be
unseen.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scope40 import SCOPE_DIR, load_domains  # noqa: E402

LEVELS = ("family", "superfamily", "fold")
EXTENSIONS = (".gz", ".tsv", ".m8", ".ent", ".pdb", ".cif")


def method_name(path):
    name = os.path.basename(path)
    for extension in EXTENSIONS:
        if name.endswith(extension):
            name = name[: -len(extension)]
    return name


def domain_names(names, known):
    """{hit name: domain id}. Foldseek can add a path, an extension or a _<chain> suffix."""
    mapping = {}
    for name in names:
        base = method_name(name)
        if base not in known and "_" in base and base.rsplit("_", 1)[0] in known:
            base = base.rsplit("_", 1)[0]
        mapping[name] = base
    return mapping


def eligible_queries(domains):
    """The awk script's per-domain counts, and which domains count as queries."""
    counts = {level: domains.groupby(level)[level].transform("size") for level in LEVELS}
    eligible = ((counts["family"] > 1)
                & (counts["superfamily"] - counts["family"] > 0)
                & (counts["fold"] - counts["superfamily"] > 0))
    return counts, eligible


def read_hits(path):
    """A hit file's first two columns, as query and target, in file order."""
    return pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], names=["query", "target"],
                       dtype=str, keep_default_na=False)


def evaluate(hits, domains, queries=None):
    """Per-query sensitivity at each level, indexed by domain; also the share of unknown targets.

    hits: a hit file, or a DataFrame of query and target in rank order.
    domains: load_domains() indexed by domain. queries: optional set of domains
    to report on; the rest still count as database entries.
    """
    hits = read_hits(hits) if isinstance(hits, (str, os.PathLike)) else hits[["query", "target"]].copy()
    known = set(domains.index)
    names = domain_names(pd.unique(pd.concat([hits["query"], hits["target"]])), known)
    hits["query"] = hits["query"].map(names)
    hits["target"] = hits["target"].map(names)
    hits = hits[hits["query"].isin(known)]
    # the first, best line per pair: Foldseek splits multi-chain files into one entry per chain
    hits = hits.drop_duplicates(["query", "target"])
    hits = hits[hits["query"] != hits["target"]].reset_index(drop=True)

    unknown = ~hits["target"].isin(known)
    same = {level: (hits["query"].map(domains[level]) == hits["target"].map(domains[level])).to_numpy()
            for level in LEVELS}
    false_positive = unknown.to_numpy() | ~same["fold"]
    before_first_fp = pd.Series(false_positive).groupby(hits["query"]).cumsum().to_numpy() == 0
    found = pd.DataFrame({
        "family": before_first_fp & same["family"],
        "superfamily": before_first_fp & ~same["family"] & same["superfamily"],
        "fold": before_first_fp & ~same["superfamily"] & same["fold"],
    }).groupby(hits["query"]).sum()

    counts, eligible = eligible_queries(domains)
    if queries is not None:
        eligible &= domains.index.isin(queries)
    found = found.reindex(domains.index[eligible], fill_value=0)
    # the awk script's denominators, kept for comparability: the superfamily and
    # fold ones are one more than the true positives possible, so even a perfect
    # ranking scores just below 1 at those levels
    possible = {
        "family": counts["family"] - 1,
        "superfamily": counts["superfamily"] - counts["family"] + 1,
        "fold": counts["fold"] - counts["superfamily"] + 1,
    }
    result = pd.DataFrame({level: found[level] / possible[level][eligible] for level in LEVELS})
    return result, float(unknown.mean()) if len(hits) else 0.0


def query_set(domains, max_train_identity):
    if max_train_identity is None:
        return None
    if "train_identity" not in domains or domains["train_identity"].isna().all():
        raise SystemExit("domains.tsv has no train_identity; rerun prepare_scope.py with --mmseqs")
    return set(domains.index[domains["train_identity"] < max_train_identity])


def print_table(results):
    """results: {method: per-query DataFrame}."""
    width = max(len(name) for name in results)
    print(f"{'method':<{width}}  {'queries':>7}  {'family':>7}  {'superfamily':>11}  {'fold':>7}")
    for name, result in results.items():
        means = result.mean()
        print(f"{name:<{width}}  {len(result):>7,}  {means['family']:>7.3f}  "
              f"{means['superfamily']:>11.3f}  {means['fold']:>7.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hits", nargs="+", help="hit files, as path or name=path")
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--max-train-identity", type=float, default=None,
                        help="report on queries whose closest training protein is below this identity")
    parser.add_argument("--per-query", default=None, help="write every query's sensitivities here")
    args = parser.parse_args()

    domains = load_domains(args.dir).set_index("domain")
    queries = query_set(domains, args.max_train_identity)
    results = {}
    for item in args.hits:
        name, _, path = item.rpartition("=") if "=" in item else (method_name(item), "", item)
        result, unknown = evaluate(path, domains, queries)
        if unknown > 0.01:
            print(f"WARNING: {unknown:.1%} of {name}'s hits are to domains outside the benchmark "
                  f"and count as false positives; check that its names match domains.tsv", file=sys.stderr)
        results[name] = result

    if queries is not None:
        print(f"queries whose closest training protein is below {args.max_train_identity:.0%} identity")
    print_table(results)
    if args.per_query:
        table = pd.concat(results, axis=1)
        table.columns = [f"{name}_{level}" for name, level in table.columns]
        table.to_csv(args.per_query, sep="\t", index_label="domain")
        print(f"wrote {args.per_query}")


if __name__ == "__main__":
    main()
