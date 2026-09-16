"""Deterministic train / val / test split, shared by every script.

Proteins are bucketed by a stable hash of their split key. The key is the
protein's sequence-cluster representative when CLUSTER_TSV exists (built by
cluster_proteins.py with MMseqs2), otherwise the UniProt id itself. Hashing
raw ids puts near-identical orthologues on both sides of the split, so the
id fallback is for smoke tests only and is announced once on stderr.

A pair is kept only when both proteins fall in the same split, so val and
test shrink quadratically with the bucket share; 10% buckets give a few
thousand held-out pairs per 200k source rows.
"""

import hashlib
import os
import sys

HOLDOUT_MODE = "protein"

# bucket ranges over 0-99
TEST_BUCKETS = 10  # buckets [0, 10)
VAL_BUCKETS = 10  # buckets [10, 20), train gets the remaining 80%

SPLITS = ("train", "val", "test")

# MMseqs2 createtsv output: representative<TAB>member, one member per line
CLUSTER_TSV = "/jet/home/jxu23/OCEANDIR/protein_clusters.tsv"

_CLUSTER_MAP = None  # loaded lazily; {} means "no map, hash ids"


def load_cluster_map(path=None):
    """{member id: representative id}, or {} when the file is absent."""
    path = CLUSTER_TSV if path is None else path
    if not path or not os.path.exists(path):
        return {}

    mapping = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mapping[parts[1].strip()] = parts[0].strip()
    return mapping


def cluster_map():
    global _CLUSTER_MAP
    if _CLUSTER_MAP is None:
        _CLUSTER_MAP = load_cluster_map()
        if not _CLUSTER_MAP:
            print(
                f"WARNING: no cluster map at {CLUSTER_TSV}; splitting by protein id, "
                f"which lets close homologues cross the split. Run "
                f"data_scripts/cluster_proteins.py to build it.",
                file=sys.stderr,
                flush=True,
            )
    return _CLUSTER_MAP


def split_key(protein_id):
    """The cluster representative when known, else the id itself."""
    return cluster_map().get(protein_id, protein_id)


def stable_bucket(key):
    """Deterministic 0-99 bucket."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def bucket_split(bucket):
    if bucket < TEST_BUCKETS:
        return "test"
    if bucket < TEST_BUCKETS + VAL_BUCKETS:
        return "val"
    return "train"


def protein_split(protein_id):
    return bucket_split(stable_bucket(split_key(protein_id)))


def row_split(id1, id2):
    """Assign a CSV row to 'train', 'val', 'test', or 'discard'."""
    if HOLDOUT_MODE == "pair":
        return bucket_split(stable_bucket(f"{id1}|{id2}"))

    split1 = protein_split(id1)
    split2 = protein_split(id2)
    return split1 if split1 == split2 else "discard"


def split_summary():
    unit = "clusters" if cluster_map() else "protein ids (NO cluster map)"
    return (
        f"{HOLDOUT_MODE} split by {unit}: test={TEST_BUCKETS}% val={VAL_BUCKETS}% "
        f"train={100 - TEST_BUCKETS - VAL_BUCKETS}%"
    )
