"""The SCOPe40 benchmark shared by the scripts in this folder.

The domains and their labels are those of Foldseek's SCOPe benchmark (van
Kempen et al., 2024; github.com/steineggerlab/foldseek-analysis, folder
scopbenchmark): the 11,211 domains of ASTRAL SCOPe 2.01 below 40 % sequence
identity, each with its SCOP class string (sccs). In a.1.1.1, a is the class,
a.1 the fold, a.1.1 the superfamily and a.1.1.1 the family.

SCOPE_DIR, written by prepare_scope.py:
  scop_lookup.fix.tsv   domain<TAB>sccs, as published
  structures/<domain>   one PDB-style file per domain, named as in the lookup
  domains.tsv           one row per domain: labels, length, training overlap
  domains.fasta         each domain's sequence, read from its structure
  hits/                 each method's ranked hits, one file per method
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (os.path.join(REPO_ROOT, "data_scripts"),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import DATA_DIR  # noqa: E402

SCOPE_DIR = os.path.join(DATA_DIR, "scope40")

LOOKUP_URL = ("https://raw.githubusercontent.com/steineggerlab/foldseek-analysis/main/"
              "scopbenchmark/data/scop_lookup.fix.tsv")
STRUCTURES_URL = "https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.01.tgz"

LOOKUP = "scop_lookup.fix.tsv"
STRUCTURES = "structures"
TABLE = "domains.tsv"
FASTA = "domains.fasta"
HITS = "hits"


def levels(sccs):
    """(family, superfamily, fold, class) of an sccs such as a.1.1.1."""
    parts = sccs.split(".")
    return sccs, ".".join(parts[:3]), ".".join(parts[:2]), parts[0]


def load_domains(scope_dir):
    """domains.tsv as a DataFrame, one row per domain."""
    import pandas as pd

    path = os.path.join(scope_dir, TABLE)
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run scope_benchmark/prepare_scope.py first")
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""],
                       dtype={"domain": str, "sccs": str, "train_hit": str})


def read_fasta(path):
    """{id: sequence}, in file order."""
    sequences, name = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                sequences[name] = ""
            elif name is not None:
                sequences[name] += line
    return sequences
