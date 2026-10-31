#!/usr/bin/env python3
"""Download and prepare the SCOPe40 benchmark (Foldseek's SCOPe 2.01 domain set).

    python scope_benchmark/prepare_scope.py --mmseqs $(which mmseqs) --threads 8
    python scope_benchmark/prepare_scope.py --skip-overlap      # no training-set check

Each step is skipped when its output exists; --force redoes them all.

1. Download Foldseek's scop_lookup.fix.tsv (11,211 domains) and SCOPe's
   PDB-style files of the 2.01 40 % subset (about 800 MB), and extract the
   lookup's domains to structures/. SCOPe names 51 multi-chain domains g...,
   which the lookup renames d...; every file takes the lookup's name, with no
   extension, as in the benchmark.
2. Read each domain's sequence and C-alpha count: the first model, every chain
   in file order (a multi-chain domain keeps all its chains), and every amino
   acid with a CA atom, modified ones as their parent (selenomethionine as M).
   data_scripts/parse_pdb.py reads one chain and skips HETATM residues, which
   suits SWISS-MODEL files but would cut gaps into crystal structures, where
   selenomethionine is common.
3. Measure overlap with the training data. MMseqs2 searches every domain
   against the training proteins (the proteins of tm_scores.csv in the train
   split of data_scripts/splits.py). train_identity is the highest sequence
   identity of a hit covering at least half of the domain with E <= 1e-3, and 0
   without one. evaluate_search.py --max-train-identity 0.3 then keeps the
   queries without a close homologue in training.

Writes domains.tsv and domains.fasta to --dir (default scope40.SCOPE_DIR).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scope40 import (  # noqa: E402
    FASTA, LOOKUP, LOOKUP_URL, SCOPE_DIR, STRUCTURES, STRUCTURES_URL, TABLE, levels, read_fasta,
)
from common import log  # noqa: E402

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
OVERLAP_EVALUE = 1e-3
OVERLAP_MIN_COVERAGE = 0.5  # of the domain


def download(url, path):
    if os.path.exists(path):
        return
    log(f"downloading {url}")
    partial = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "PRIMES-PLM SCOPe benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    os.replace(partial, path)


def read_lookup(path):
    """{domain: sccs}, in file order."""
    lookup = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                lookup[fields[0]] = fields[1]
    return lookup


def extract_structures(archive, lookup, directory):
    """Write each lookup domain's file to directory/<domain>; returns the domains missing."""
    os.makedirs(directory, exist_ok=True)
    written = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            name = name[:-len(".ent")] if name.endswith(".ent") else name
            if name in lookup:
                domain = name
            elif name.startswith("g") and "d" + name[1:] in lookup:
                domain = "d" + name[1:]  # the lookup's name for a multi-chain domain
            else:
                continue
            with tar.extractfile(member) as source, open(os.path.join(directory, domain), "wb") as target:
                shutil.copyfileobj(source, target)
            written.add(domain)
    return [domain for domain in lookup if domain not in written]


def read_domain(path):
    """(sequence, chain count) over every chain of the first model."""
    from Bio.Data.PDBData import protein_letters_3to1_extended
    from Bio.PDB import PDBParser

    models = list(PDBParser(QUIET=True).get_structure("domain", path))
    if not models:
        return "", 0
    sequence, chains = [], 0
    for chain in models[0]:
        before = len(sequence)
        for residue in chain:
            if "CA" not in residue:
                continue
            letter = protein_letters_3to1_extended.get(residue.resname)
            if letter in AMINO_ACIDS:
                sequence.append(letter)
        chains += len(sequence) > before
    return "".join(sequence), chains


def training_fasta(path):
    """Write the train-split proteins of tm_scores.csv; returns how many."""
    from cluster_proteins import TM_CSV, collect_sequences
    from splits import cluster_map, protein_split

    if not cluster_map():
        raise SystemExit("no cluster map, so no train split to compare with; "
                         "run data_scripts/cluster_proteins.py first")
    log(f"reading training proteins from {TM_CSV}")
    count = 0
    with open(path, "w") as handle:
        for protein_id, sequence in collect_sequences(TM_CSV).items():
            if protein_split(protein_id) == "train":
                handle.write(f">{protein_id}\n{sequence}\n")
                count += 1
    return count


def training_overlap(mmseqs, fasta, scope_dir, threads):
    """{domain: (identity, training protein)} of each domain's closest training protein."""
    work = tempfile.mkdtemp(prefix="overlap_", dir=scope_dir)
    train = os.path.join(work, "train.fasta")
    log(f"{training_fasta(train):,} training proteins")
    hits = os.path.join(work, "overlap.m8")
    command = [mmseqs, "easy-search", fasta, train, hits, os.path.join(work, "tmp"),
               "-s", "7.5", "-e", str(OVERLAP_EVALUE), "--threads", str(threads),
               "--format-output", "query,target,fident,qstart,qend,qlen"]
    log(" ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    closest = {}
    with open(hits) as handle:
        for line in handle:
            query, target, identity, start, end, length = line.split("\t")
            if (int(end) - int(start) + 1) / int(length) < OVERLAP_MIN_COVERAGE:
                continue
            if float(identity) > closest.get(query, (-1.0, ""))[0]:
                closest[query] = (float(identity), target)
    shutil.rmtree(work)
    return closest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=SCOPE_DIR)
    parser.add_argument("--mmseqs", default=None, help="mmseqs binary, for the training-overlap check")
    parser.add_argument("--skip-overlap", action="store_true", help="leave train_identity empty")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-archive", action="store_true", help="keep the 800 MB download")
    parser.add_argument("--force", action="store_true", help="redo every step")
    args = parser.parse_args()
    if not args.skip_overlap and not args.mmseqs:
        parser.error("the training-overlap check needs --mmseqs (or pass --skip-overlap)")
    os.makedirs(args.dir, exist_ok=True)

    lookup_path = os.path.join(args.dir, LOOKUP)
    structures = os.path.join(args.dir, STRUCTURES)
    fasta_path = os.path.join(args.dir, FASTA)
    table_path = os.path.join(args.dir, TABLE)
    if args.force:
        for path in (lookup_path, fasta_path, table_path):
            if os.path.exists(path):
                os.remove(path)
        shutil.rmtree(structures, ignore_errors=True)

    # 1. the domain list and the structures
    download(LOOKUP_URL, lookup_path)
    lookup = read_lookup(lookup_path)
    log(f"{len(lookup):,} domains in the lookup")
    if not os.path.isdir(structures):
        archive = os.path.join(args.dir, os.path.basename(STRUCTURES_URL))
        download(STRUCTURES_URL, archive)
        log(f"extracting the lookup's domains to {structures}")
        # into a temporary folder first, so an interrupted run is not mistaken for a finished one
        partial = structures + ".part"
        shutil.rmtree(partial, ignore_errors=True)
        missing = extract_structures(archive, lookup, partial)
        os.replace(partial, structures)
        if missing:
            log(f"WARNING: {len(missing)} domains not in the archive, e.g. {missing[:5]}")
        if not args.keep_archive:
            os.remove(archive)

    # 2. sequences from the structures
    if os.path.exists(fasta_path):
        sequences = read_fasta(fasta_path)
        chains = {}
    else:
        sequences, chains = {}, {}
        for k, domain in enumerate(lookup):
            path = os.path.join(structures, domain)
            if not os.path.exists(path):
                continue
            try:
                sequence, n_chains = read_domain(path)
            except Exception as error:  # one unreadable file should not end the run
                log(f"WARNING: cannot read {domain} ({error}); left out")
                continue
            if sequence:
                sequences[domain], chains[domain] = sequence, n_chains
            else:
                log(f"WARNING: no amino acids with a CA atom in {domain}; left out")
            if (k + 1) % 2000 == 0:
                log(f"  read {k + 1:,}/{len(lookup):,}")
        with open(fasta_path, "w") as handle:
            for domain, sequence in sequences.items():
                handle.write(f">{domain}\n{sequence}\n")
    log(f"{len(sequences):,} domains with a sequence")

    # 3. overlap with the training proteins
    import pandas as pd

    rows = []
    for domain, sequence in sequences.items():
        family, superfamily, fold, scop_class = levels(lookup[domain])
        rows.append({"domain": domain, "sccs": lookup[domain], "family": family,
                     "superfamily": superfamily, "fold": fold, "class": scop_class,
                     "length": len(sequence), "chains": chains.get(domain)})
    table = pd.DataFrame(rows)
    if os.path.exists(table_path):  # keep chain counts and overlap from an earlier run
        old = pd.read_csv(table_path, sep="\t", keep_default_na=False, na_values=[""],
                          dtype={"domain": str, "train_hit": str}).set_index("domain")
        for column in ("chains", "train_identity", "train_hit"):
            if column in old and (column not in table or table[column].isna().all()):
                table[column] = table["domain"].map(old[column])
    if not args.skip_overlap and ("train_identity" not in table or table["train_identity"].isna().all()):
        closest = training_overlap(args.mmseqs, fasta_path, args.dir, args.threads)
        table["train_identity"] = [closest.get(d, (0.0, ""))[0] for d in table["domain"]]
        table["train_hit"] = [closest.get(d, (0.0, ""))[1] for d in table["domain"]]
    table.to_csv(table_path, sep="\t", index=False)

    log("")
    log(f"wrote {table_path} and {fasta_path}")
    log(f"  domains {len(table):,}; families {table['family'].nunique():,}, superfamilies "
        f"{table['superfamily'].nunique():,}, folds {table['fold'].nunique():,}")
    log(f"  length median {table['length'].median():.0f}, max {table['length'].max()}; "
        f"multi-chain {int((table['chains'] > 1).sum()) if table['chains'].notna().any() else 'n/a'}")
    if "train_identity" in table and table["train_identity"].notna().any():
        for cut in (0.3, 0.5, 0.9):
            kept = (table["train_identity"] < cut).sum()
            log(f"  closest training protein below {cut:.0%} identity: {kept:,} domains "
                f"({kept / len(table):.0%})")


if __name__ == "__main__":
    main()
