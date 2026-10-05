#!/usr/bin/env python3
"""Pack every protein's CA coordinates and ESM embedding into a few large files.

    python data_scripts/protein_pack.py               # proteins in fgw_scores.csv not yet packed
    python data_scripts/protein_pack.py --all         # every protein with an embedding
    python data_scripts/protein_pack.py --rebuild     # start over, e.g. after regenerating embeddings
    python data_scripts/protein_pack.py --model 650M  # another model's embeddings, same proteins

--model packs another ESM-2 model's embeddings (embeddings_<model>/) into its
own protein_pack_<model>/, for the student's input. Its coordinates are copied
from the 35M pack rather than parsed from the PDBs again.

Why: training reads two proteins per pair. From per-protein files on the
shared filesystem that cost ~170 ms to parse a PDB and ~40 ms to open one
embedding, almost all of it per-file latency rather than data. Here each
worker writes one shard (a coordinate file, an embedding file, and an index),
and the loader reads a protein as two slices of already-open files.

The coordinates come from parse_pdb and the embeddings are the stored float16
arrays, byte for byte, so training sees exactly what the per-file path gave.
A run verifies a random sample against the original files at the end.

Incremental and resumable: proteins already in the pack are skipped, and each
run writes new shards, never appending to old ones. A protein is indexed only
after its data is written, so a job killed mid-shard leaves only unreferenced
bytes. Proteins that cannot be packed (missing file, length mismatch) are
left out; the loader reads those from the original files, as before.

Layout of the pack directory:
    manifest.json
    shard_NNNNN.coords     float32 [rows, 3]
    shard_NNNNN.emb        float16 [rows, embedding_dim]
    shard_NNNNN.index.csv  id,row,length
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import BASE_ESM, ESM_MODELS, embedding_path, esm_dirs, log, pdb_path  # noqa: E402
from parse_pdb import parse_pdb  # noqa: E402

DATA_DIR = "/jet/home/jxu23/OCEANDIR"
PACK_DIR = os.path.join(DATA_DIR, "protein_pack")
EMBEDDING_DIM = 480

PACK_FORMAT = 1
MANIFEST = "manifest.json"
COORDS_DTYPE = np.dtype(np.float32)
EMBEDDING_DTYPE = np.dtype(np.float16)
SHARD_PATTERN = re.compile(r"^shard_(\d+)\.(coords|emb|index\.csv)$")


def shard_paths(pack_dir, shard):
    base = os.path.join(pack_dir, shard)
    return f"{base}.coords", f"{base}.emb", f"{base}.index.csv"


def load_index(pack_dir):
    """{protein id: (shard, row, length)} over every shard's index."""
    index = {}
    for name in sorted(os.listdir(pack_dir)):
        match = SHARD_PATTERN.match(name)
        if not match or match.group(2) != "index.csv":
            continue
        shard = name[: -len(".index.csv")]
        with open(os.path.join(pack_dir, name)) as handle:
            next(handle, None)  # header
            for line in handle:
                parts = line.rstrip("\n").split(",")
                if len(parts) == 3:
                    index[parts[0]] = (shard, int(parts[1]), int(parts[2]))
    return index


def _read_exact(fd, nbytes, offset):
    chunks, remaining = [], nbytes
    while remaining:
        chunk = os.pread(fd, remaining, offset + nbytes - remaining)
        if not chunk:
            raise EOFError(f"pack file ends before byte {offset + nbytes}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


class ProteinPack:
    """Read-only access to a packed directory.

    Files are opened once and read with pread, which never moves a shared
    file offset, so DataLoader workers forked from this process can use the
    inherited descriptors safely. Pickling (spawned workers) drops them and
    the receiving process reopens its own.
    """

    def __init__(self, pack_dir):
        with open(os.path.join(pack_dir, MANIFEST)) as handle:
            manifest = json.load(handle)
        if manifest.get("format") != PACK_FORMAT:
            raise ValueError(f"{pack_dir}: pack format {manifest.get('format')}, expected {PACK_FORMAT}")
        self.pack_dir = pack_dir
        self.embedding_dim = int(manifest["embedding_dim"])
        self.index = load_index(pack_dir)
        self._fds = {}

    def __len__(self):
        return len(self.index)

    def __contains__(self, protein_id):
        return protein_id in self.index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fds"] = {}
        return state

    def _fd(self, path):
        fd = self._fds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[path] = fd
        return fd

    def get(self, protein_id):
        """(coords float32 [L, 3], embeddings float16 [L, dim]), or None if not packed."""
        record = self.index.get(protein_id)
        if record is None:
            return None
        shard, row, length = record
        coords_path, emb_path, _ = shard_paths(self.pack_dir, shard)

        coords_bytes = 3 * COORDS_DTYPE.itemsize
        emb_bytes = self.embedding_dim * EMBEDDING_DTYPE.itemsize
        coords = np.frombuffer(
            _read_exact(self._fd(coords_path), length * coords_bytes, row * coords_bytes),
            dtype=COORDS_DTYPE,
        ).reshape(length, 3)
        embeddings = np.frombuffer(
            _read_exact(self._fd(emb_path), length * emb_bytes, row * emb_bytes),
            dtype=EMBEDDING_DTYPE,
        ).reshape(length, self.embedding_dim)
        return coords, embeddings


def open_pack(pack_dir):
    """The pack at pack_dir, or None (with a log line) when there is none."""
    if not pack_dir:
        return None
    if not os.path.exists(os.path.join(pack_dir, MANIFEST)):
        log(
            f"no protein pack at {pack_dir}: reading PDB and embedding files "
            f"directly, which is slow on shared storage (data_scripts/protein_pack.py builds it)"
        )
        return None
    pack = ProteinPack(pack_dir)
    log(f"protein pack: {len(pack):,} proteins from {pack_dir}")
    return pack


# ---------------------------------------------------------------- building


def next_shard_number(pack_dir):
    numbers = [
        int(match.group(1))
        for match in (SHARD_PATTERN.match(name) for name in os.listdir(pack_dir))
        if match
    ]
    return max(numbers) + 1 if numbers else 0


def write_manifest(pack_dir, embedding_dim, pdb_dir, embedding_dir):
    path = os.path.join(pack_dir, MANIFEST)
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        if existing.get("format") != PACK_FORMAT or int(existing["embedding_dim"]) != embedding_dim:
            raise ValueError(f"{path} describes a different pack; use --rebuild")
        return
    manifest = {
        "format": PACK_FORMAT,
        "embedding_dim": embedding_dim,
        "coords_dtype": str(COORDS_DTYPE),
        "embedding_dtype": str(EMBEDDING_DTYPE),
        "pdb_dir": pdb_dir,
        "embedding_dir": embedding_dir,
        "coordinates": "parse_pdb: CA of the first chain with standard residues, first model",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path + ".tmp", "w") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(path + ".tmp", path)


def read_protein(protein_id, pdb_dir, embedding_dir, embedding_dim, coords_pack=None):
    """(coords, embeddings), or (None, reason) when the protein cannot be packed.

    coords_pack, a ProteinPack, supplies the coordinates when it holds the
    protein; they are parse_pdb's, byte for byte, without the parse.
    """
    try:
        packed = coords_pack.get(protein_id) if coords_pack is not None else None
        if packed is not None:
            coords = packed[0]
        else:
            coords, _ = parse_pdb(pdb_path(pdb_dir, protein_id))
        embeddings = np.load(embedding_path(embedding_dir, protein_id))
    except Exception as exc:
        return None, type(exc).__name__
    if len(coords) == 0:
        return None, "no residues"
    if embeddings.dtype != EMBEDDING_DTYPE:
        return None, f"embedding dtype {embeddings.dtype}"  # packing would change values
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        return None, f"embedding shape {embeddings.shape}"
    if len(coords) != len(embeddings):
        return None, "length mismatch"
    return (np.ascontiguousarray(coords, dtype=COORDS_DTYPE), np.ascontiguousarray(embeddings)), None


def pack_shard(task):
    """Write one shard. Returns (shard, packed count, {skip reason: count})."""
    (number, protein_ids, pdb_dir, embedding_dir, pack_dir, embedding_dim, log_every,
     coords_pack_dir) = task
    coords_pack = ProteinPack(coords_pack_dir) if coords_pack_dir else None
    shard = f"shard_{number:05d}"
    coords_path, emb_path, index_path = shard_paths(pack_dir, shard)
    packed, skipped, row = 0, Counter(), 0
    start = time.time()

    with open(coords_path, "wb") as coords_file, open(emb_path, "wb") as emb_file, open(
        index_path, "w"
    ) as index_file:
        index_file.write("id,row,length\n")
        index_file.flush()
        for done, protein_id in enumerate(protein_ids, start=1):
            data, reason = read_protein(
                protein_id, pdb_dir, embedding_dir, embedding_dim, coords_pack
            )
            if data is None:
                skipped[reason] += 1
            else:
                coords, embeddings = data
                coords_file.write(coords.tobytes())
                emb_file.write(embeddings.tobytes())
                coords_file.flush()
                emb_file.flush()
                # indexed only once its bytes are written
                index_file.write(f"{protein_id},{row},{len(coords)}\n")
                index_file.flush()
                row += len(coords)
                packed += 1
            if log_every and done % log_every == 0:
                log(f"  {shard}: {done:,}/{len(protein_ids):,} ({done / (time.time() - start):.1f}/s)")
    return shard, packed, dict(skipped)


def pack_proteins(protein_ids, pdb_dir, embedding_dir, pack_dir, workers=1,
                  embedding_dim=EMBEDDING_DIM, log_every=2000, coords_pack_dir=None):
    """Pack the proteins not already in pack_dir. Returns (packed, {reason: count}, shards)."""
    os.makedirs(pack_dir, exist_ok=True)
    write_manifest(pack_dir, embedding_dim, pdb_dir, embedding_dir)
    existing = load_index(pack_dir)
    todo = sorted({p for p in protein_ids if p not in existing})
    log(f"{len(existing):,} proteins already packed, {len(todo):,} to pack")
    if not todo:
        return 0, {}, []

    num_shards = max(1, min(workers, len(todo)))
    first = next_shard_number(pack_dir)
    tasks = [
        (first + k, todo[k::num_shards], pdb_dir, embedding_dir, pack_dir, embedding_dim, log_every,
         coords_pack_dir)
        for k in range(num_shards)
    ]
    if workers > 1:
        import multiprocessing

        with multiprocessing.Pool(min(workers, num_shards)) as pool:
            results = pool.map(pack_shard, tasks, chunksize=1)
    else:
        results = [pack_shard(task) for task in tasks]

    packed, skipped = 0, Counter()
    for _, count, reasons in results:
        packed += count
        skipped.update(reasons)
    return packed, dict(skipped), [shard for shard, _, _ in results]


def verify(pack_dir, pdb_dir, embedding_dir, sample, seed=0):
    """Compare a random sample of packed proteins with the original files, exactly."""
    pack = ProteinPack(pack_dir)
    ids = random.Random(seed).sample(sorted(pack.index), min(sample, len(pack)))
    mismatched = []
    for protein_id in ids:
        coords, embeddings = pack.get(protein_id)
        reference_coords, _ = parse_pdb(pdb_path(pdb_dir, protein_id))
        reference_embeddings = np.load(embedding_path(embedding_dir, protein_id))
        if not (
            np.array_equal(coords, reference_coords)
            and np.array_equal(embeddings, reference_embeddings)
        ):
            mismatched.append(protein_id)
    return len(ids), mismatched


def remove_pack(pack_dir):
    removed = 0
    for name in os.listdir(pack_dir):
        if SHARD_PATTERN.match(name) or name in (MANIFEST, MANIFEST + ".tmp"):
            os.remove(os.path.join(pack_dir, name))
            removed += 1
    return removed


def proteins_in_csv(csv_path):
    import pandas as pd

    ids = set()
    for chunk in pd.read_csv(csv_path, usecols=["id1", "id2"], chunksize=500_000, dtype=str):
        ids.update(chunk["id1"].str.strip())
        ids.update(chunk["id2"].str.strip())
    ids.discard("")
    return ids


def default_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--pack-dir", default=None, help="default: <data-dir>/protein_pack")
    parser.add_argument("--csv", default=None, help="default: <data-dir>/fgw_scores.csv")
    parser.add_argument("--all", action="store_true", help="pack every protein with an embedding")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--model", default=BASE_ESM, choices=sorted(ESM_MODELS),
                        help="whose embeddings to pack; sets the embedding dir, pack dir and width")
    parser.add_argument("--protein-list", default=None,
                        help="pack only these proteins (one id per line)")
    parser.add_argument("--embedding-dim", type=int, default=None, help="default: by model")
    parser.add_argument("--verify", type=int, default=200, help="proteins to check exactly; 0 skips")
    parser.add_argument("--rebuild", action="store_true", help="delete the existing pack first")
    args = parser.parse_args()

    pdb_dir = os.path.join(args.data_dir, "pdbs")
    embedding_dir, default_pack_dir = esm_dirs(args.model, args.data_dir)
    pack_dir = args.pack_dir or default_pack_dir
    embedding_dim = args.embedding_dim or ESM_MODELS[args.model][1]
    csv_path = args.csv or os.path.join(args.data_dir, "fgw_scores.csv")
    # another model's pack copies its coordinates from the 35M pack
    base_pack_dir = esm_dirs(BASE_ESM, args.data_dir)[1]
    coords_pack_dir = None
    if os.path.abspath(pack_dir) != os.path.abspath(base_pack_dir) and os.path.exists(
        os.path.join(base_pack_dir, MANIFEST)
    ):
        coords_pack_dir = base_pack_dir
        log(f"coordinates from {base_pack_dir}")

    if args.rebuild and os.path.isdir(pack_dir):
        log(f"--rebuild: removed {remove_pack(pack_dir)} files from {pack_dir}")

    start = time.time()
    if args.protein_list:
        with open(args.protein_list) as handle:
            protein_ids = {line.strip() for line in handle if line.strip() and not line.startswith("#")}
        log(f"{len(protein_ids):,} proteins in {args.protein_list}")
    elif args.all:
        protein_ids = {name[:-4] for name in os.listdir(embedding_dir) if name.endswith(".npy")}
        log(f"{len(protein_ids):,} proteins with an embedding in {embedding_dir}")
    else:
        protein_ids = proteins_in_csv(csv_path)
        log(f"{len(protein_ids):,} proteins in {csv_path}")

    packed, skipped, shards = pack_proteins(
        protein_ids, pdb_dir, embedding_dir, pack_dir, workers=args.workers,
        embedding_dim=embedding_dim, coords_pack_dir=coords_pack_dir,
    )
    total = len(load_index(pack_dir))
    size = sum(
        os.path.getsize(os.path.join(pack_dir, name))
        for name in os.listdir(pack_dir)
        if SHARD_PATTERN.match(name)
    )
    log(f"packed {packed:,} proteins into {len(shards)} new shards in {(time.time() - start) / 60:.1f} min")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        log(f"  not packed: {reason:<30} {count:,}   (the loader reads these from the original files)")
    log(f"pack now holds {total:,} proteins, {size / 1e9:.1f} GB, in {pack_dir}")

    if args.verify:
        checked, mismatched = verify(pack_dir, pdb_dir, embedding_dir, args.verify)
        if mismatched:
            log(f"VERIFY FAILED: {len(mismatched)} of {checked} proteins differ from their files, "
                f"e.g. {mismatched[:5]}. Rebuild with --rebuild.")
            return 1
        log(f"verified {checked} random proteins: identical to their PDB and embedding files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
