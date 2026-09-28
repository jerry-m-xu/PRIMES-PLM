"""Delete embeddings that were computed from multi-chain PDB files.

Chains are contiguous in a PDB file, so the first ATOM record and the last
ATOM record of the first model carry different chain ids exactly when the
model has more than one chain. Reading the head and the tail of each file
is two small reads instead of the whole 200-300 KB, which is what makes this
finish in minutes rather than hours. Safe to rerun: deleted files are not
listed again.

    python3 data_scripts/purge_multichain_embeddings.py            # delete
    python3 data_scripts/purge_multichain_embeddings.py --dry-run  # only count
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

D = "/jet/home/jxu23/OCEANDIR"
TAIL_BYTES = 65536
DRY_RUN = "--dry-run" in sys.argv


def first_and_last_atom_chain(path):
    """(first chain id, last chain id) among ATOM records of the first model.

    Falls back to a full scan if the tail window holds no ATOM record
    (a very long ligand or water section after the protein).
    """
    with open(path, "rb") as handle:
        first = None
        for line in handle:
            if line.startswith(b"ATOM"):
                first = chr(line[21])
                break
            if line.startswith(b"ENDMDL"):
                break
        if first is None:
            return None, None

        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(size - TAIL_BYTES, 0))
        tail = handle.read().split(b"\n")

    # only the first model counts: cut the tail at the first ENDMDL if one is
    # in view, since anything after it belongs to a later model
    if any(line.startswith(b"ENDMDL") for line in tail):
        # the tail window straddles a model boundary or later models: do a
        # full first-model scan instead of guessing
        return first, _last_atom_chain_full_scan(path)

    for line in reversed(tail):
        if line.startswith(b"ATOM"):
            return first, chr(line[21])
    return first, _last_atom_chain_full_scan(path)


def _last_atom_chain_full_scan(path):
    last = None
    with open(path, "rb") as handle:
        for line in handle:
            if line.startswith(b"ENDMDL"):
                break
            if line.startswith(b"ATOM"):
                last = chr(line[21])
    return last


def check(name):
    pdb = f"{D}/pdbs/{name[:-4]}.pdb"
    if not os.path.exists(pdb):
        return name, "no pdb"
    try:
        first, last = first_and_last_atom_chain(pdb)
    except OSError as exc:
        return name, f"unreadable ({exc})"
    if first is None:
        return name, "no atoms"
    if first != last:
        if not DRY_RUN:
            os.remove(f"{D}/embeddings/{name}")
        return name, "multi-chain"
    return name, "single-chain"


def main():
    names = [n for n in os.listdir(f"{D}/embeddings") if n.endswith(".npy")]
    print(f"{len(names):,} embeddings to check{' (dry run)' if DRY_RUN else ''}", flush=True)

    counts = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for done, (name, status) in enumerate(pool.map(check, names, chunksize=256), start=1):
            counts[status] = counts.get(status, 0) + 1
            if done % 20000 == 0:
                print(f"  {done:,} checked, {counts.get('multi-chain', 0):,} multi-chain so far", flush=True)

    verb = "would remove" if DRY_RUN else "removed"
    print(f"{verb} {counts.get('multi-chain', 0):,} multi-chain embeddings")
    for status, count in sorted(counts.items()):
        print(f"  {status:<14} {count:,}")


if __name__ == "__main__":
    main()
