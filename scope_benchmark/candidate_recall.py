#!/usr/bin/env python3
"""How many of each query's true hits a ranked list holds within its first K.

    python scope_benchmark/candidate_recall.py $D/scope40/hits/student_tm.tsv.gz \
        $D/scope40/hits/esm2_650M_pooled.tsv.gz --max-train-identity 0.3 --union

align_rerank.py re-orders only each query's first --top hits (500), so a true
hit further down gains nothing from aligning well. For each list and each K,
this reports the share of the query's family members, and of the members of
its superfamily in other families and of its fold in other superfamilies (the
levels evaluate_search.py counts), found anywhere in the first K hits,
averaged over the eligible queries. --union adds the lists merged: the
candidates a re-ranker would see if it took the first K of each.

Unlike evaluate_search.py's sensitivity, the order within the first K and any
false positives do not matter, and the denominators are the true counts.
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scope40 import SCOPE_DIR, load_domains  # noqa: E402
from evaluate_search import LEVELS, domain_names, eligible_queries, method_name, query_set, read_hits  # noqa: E402


def ranked_hits(path, domains, queries):
    """The hits of the given queries as domains, self hits dropped, with each hit's rank from 0."""
    hits = read_hits(path)
    known = set(domains.index)
    names = domain_names(pd.unique(pd.concat([hits["query"], hits["target"]])), known)
    hits["query"] = hits["query"].map(names)
    hits = hits[hits["query"].isin(queries)].copy()
    hits["target"] = hits["target"].map(names)
    hits = hits.drop_duplicates(["query", "target"])
    hits = hits[hits["query"] != hits["target"]].reset_index(drop=True)
    hits["rank"] = hits.groupby("query", sort=False).cumcount()
    return hits


def recall(hits, domains, queries, possible):
    """Mean share of each query's true hits among the given hits, per level."""
    same = {level: (hits["query"].map(domains[level]) == hits["target"].map(domains[level])).to_numpy()
            for level in LEVELS}
    found = pd.DataFrame({
        "family": same["family"],
        "superfamily": ~same["family"] & same["superfamily"],
        "fold": ~same["superfamily"] & same["fold"],
    }).groupby(hits["query"].to_numpy()).sum().reindex(queries, fill_value=0)
    return pd.DataFrame({level: found[level] / possible[level][queries] for level in LEVELS}).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hits", nargs="+", help="hit files, as path or name=path")
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--top", default="100,500,2000", help="list depths K, comma-separated")
    parser.add_argument("--max-train-identity", type=float, default=None,
                        help="report on queries whose closest training protein is below this identity")
    parser.add_argument("--union", action="store_true", help="also the lists merged")
    args = parser.parse_args()

    domains = load_domains(args.dir).set_index("domain")
    counts, eligible = eligible_queries(domains)
    queries = list(domains.index[eligible])
    held_out = query_set(domains, args.max_train_identity)
    if held_out is not None:
        queries = [q for q in queries if q in held_out]
    possible = {"family": counts["family"] - 1,
                "superfamily": counts["superfamily"] - counts["family"],
                "fold": counts["fold"] - counts["superfamily"]}

    lists = {}
    for item in args.hits:
        name, _, path = item.rpartition("=") if "=" in item else (method_name(item), "", item)
        lists[name] = ranked_hits(path, domains, set(queries))
    if args.union and len(lists) > 1:
        lists["union"] = None

    if held_out is not None:
        print(f"queries whose closest training protein is below {args.max_train_identity:.0%} identity")
    width = max(len(name) for name in lists)
    print(f"{len(queries):,} queries; share of each level's true hits within the first K")
    print(f"{'list':<{width}}  {'K':>5}  {'family':>7}  {'superfamily':>11}  {'fold':>7}")
    for k in [int(depth) for depth in args.top.split(",")]:
        for name, hits in lists.items():
            if hits is None:
                top = pd.concat([h[h["rank"] < k] for h in lists.values() if h is not None])
                top = top.drop_duplicates(["query", "target"])
            else:
                top = hits[hits["rank"] < k]
            means = recall(top, domains, queries, possible)
            print(f"{name:<{width}}  {k:>5,}  {means['family']:>7.3f}  {means['superfamily']:>11.3f}  "
                  f"{means['fold']:>7.3f}")


if __name__ == "__main__":
    main()
