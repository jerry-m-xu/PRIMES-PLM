"""Read one chain's CA trace and sequence from a PDB file.

One chain only. SWISS-MODEL repository files for oligomeric templates carry
several copies of the target protein as separate chains. Reading every chain
glued the copies into one sequence, which either tripped the 1000-residue
cap or, worse, passed under it with a doubled protein. The chain read is the
first one in file order that has a standard amino acid with a CA atom; for a
homo-oligomer every chain is the same protein, so the first is as good as any.

    python parse_pdb.py file.pdb     # chains, their lengths, and which one is used
"""

import sys
from collections import OrderedDict

import numpy as np
from Bio.PDB import PDBParser

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _first_model(pdb_path):
    """The first model, or an empty list when the file has no ATOM records."""
    structure = PDBParser(QUIET=True).get_structure("prot", pdb_path)
    models = list(structure)
    return models[0] if models else []


def _standard_residues(chain):
    """(CA coordinate, one-letter code) for each standard amino acid with a CA."""
    for residue in chain:
        if residue.id[0] != " ":  # HETATM: ligands, water, modified residues
            continue
        if "CA" not in residue:
            continue
        aa1 = AA3_TO_AA1.get(residue.resname)
        if aa1 is None:
            continue
        yield residue["CA"].coord, aa1


def chain_lengths(pdb_path):
    """Chain id -> standard CA residue count, in file order, first model only.

    More than one entry with a similar count means an oligomeric model. Empty
    for a file with no ATOM records.
    """
    return OrderedDict(
        (chain.id, sum(1 for _ in _standard_residues(chain)))
        for chain in _first_model(pdb_path)
    )


def parse_pdb(pdb_path, chain_id=None):
    """Coordinates [L, 3] (float32) and one-letter sequence of one chain.

    chain_id=None reads the first chain with any standard residue. An empty
    result means the file has no usable chain (or no ATOM records at all);
    callers raise on that.
    """
    model = _first_model(pdb_path)
    chains = [chain for chain in model if chain_id is None or chain.id == chain_id]
    if chain_id is not None and not chains:
        raise ValueError(f"chain {chain_id!r} not in {pdb_path}")

    for chain in chains:
        residues = list(_standard_residues(chain))
        if residues:
            coords = np.asarray([coord for coord, _ in residues], dtype=np.float32)
            return coords, "".join(aa1 for _, aa1 in residues)

    return np.zeros((0, 3), dtype=np.float32), ""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} file.pdb")
    path = sys.argv[1]
    lengths = chain_lengths(path)
    coords, sequence = parse_pdb(path)
    print("chains:", ", ".join(f"{cid}={n}" for cid, n in lengths.items()) or "none")
    print("used  :", len(sequence), "residues")
    print("seq   :", sequence[:100] + ("..." if len(sequence) > 100 else ""))
