"""Protein-pair batching, shared by the teacher and the student."""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from common import embedding_path, log, pdb_path
from fgw import FGW_LABEL_SCALE, GW_LABEL_SCALE, similarity_from_distortion
from parse_pdb import parse_pdb
from patches import K_NEIGHBORS, knn_indices, pad_patch
from protein_pack import open_pack
from splits import row_split

USECOLS = [
    "tm_data_row",
    "id1",
    "id2",
    "residue_idx1",
    "residue_idx2",
    "gw_raw",
    "fgw_raw",
    "tm_term",
    "pair_type",
    "tm_score_norm1",
    "tm_score_norm2",
]

# The CSV stores raw distortions; the labels are exp(-raw / scale), in (0, 1].
# "structure": pure-GW distortion of the two patches, features not consulted.
# "composite": the fused objective, which also rewards ESM agreement.
# "tm_term":   1 / (1 + (d / d0)^2) from TM-align's superposition; chirality-
#              aware, and low for residues that are locally intact but displaced.
# The scales live in fgw.py so the audit, this loader and the checkpoint
# configs agree; choose them from audit_fgw_data.py's label percentiles.
FGW_TARGETS = ("structure", "composite", "tm_term")

# fgw_data.py's pair_type column, as small integers for the batch
PAIR_TYPES = ("aligned", "shifted", "random")
PAIR_TYPE_CODES = {name: code for code, name in enumerate(PAIR_TYPES)}

# Dense teacher-scored pairs (student only). Each protein pair contributes a
# residue set R1 of protein 1 and R2 of protein 2, and every (r1, r2) in
# R1 x R2 is scored by the frozen teacher, so no label is needed. The cells
# fall into these classes, which the loss weights equally:
#   aligned  on the TM-align path
#   near     within max_shift residues of the path along either protein:
#            the hard negatives that sequence context makes look alike
#   far      everything else
DENSE_TYPES = ("aligned", "near", "far")


class DenseConfig:
    """How many residues a protein pair contributes to the dense pairs.

    anchors aligned pairs (i, j) put i in R1 and j in R2;
    shifts_per_anchor residues j +/- s, s in [1, max_shift], join R2 per anchor;
    random residues of each protein join R1 and R2.
    """

    def __init__(self, anchors=16, shifts_per_anchor=2, max_shift=32, random=8):
        self.anchors = anchors
        self.shifts_per_anchor = shifts_per_anchor
        self.max_shift = max_shift
        self.random = random


def sample_dense_residues(aligned1, aligned2, length1, length2, config, rng):
    """Residue sets R1, R2 and the class of every R1 x R2 cell.

    aligned1, aligned2: residue indices of the aligned pairs to draw anchors
    from (the full alignment when available, else the labelled positives).
    Returns (r1, r2, types), r1 and r2 sorted and unique, types an int8
    [len(r1), len(r2)] matrix of DENSE_TYPES codes.
    """
    aligned1 = np.asarray(aligned1, dtype=np.int64)
    aligned2 = np.asarray(aligned2, dtype=np.int64)

    pick = rng.choice(len(aligned1), size=min(config.anchors, len(aligned1)), replace=False)
    anchors1, anchors2 = aligned1[pick], aligned2[pick]

    # j +/- s, falling back to the other direction near a chain end, as
    # fgw_data.shifted_pair() does
    count = len(anchors2) * config.shifts_per_anchor
    centres = np.repeat(anchors2, config.shifts_per_anchor)
    steps = rng.integers(1, config.max_shift + 1, size=count) * rng.choice((-1, 1), size=count)
    shifted = centres + steps
    outside = (shifted < 0) | (shifted >= length2)
    shifted[outside] = centres[outside] - steps[outside]
    shifted = shifted[(shifted >= 0) & (shifted < length2)]

    random1 = rng.choice(length1, size=min(config.random, length1), replace=False)
    random2 = rng.choice(length2, size=min(config.random, length2), replace=False)

    r1 = np.unique(np.concatenate([anchors1, random1])).astype(np.int64)
    r2 = np.unique(np.concatenate([anchors2, shifted, random2])).astype(np.int64)

    # each residue's partner on the path, -1 if it has none
    partner_of_1 = np.full(length1, -1, dtype=np.int64)
    partner_of_1[aligned1] = aligned2
    partner_of_2 = np.full(length2, -1, dtype=np.int64)
    partner_of_2[aligned2] = aligned1
    p1 = partner_of_1[r1][:, None]  # partner in protein 2 of each R1 residue
    p2 = partner_of_2[r2][None, :]  # partner in protein 1 of each R2 residue
    offset2 = np.abs(r2[None, :] - p1)
    offset1 = np.abs(r1[:, None] - p2)

    on_path = (p1 >= 0) & (offset2 == 0)
    near = ((p1 >= 0) & (offset2 <= config.max_shift)) | (
        (p2 >= 0) & (offset1 <= config.max_shift)
    )
    types = np.full((len(r1), len(r2)), DENSE_TYPES.index("far"), dtype=np.int8)
    types[near] = DENSE_TYPES.index("near")
    types[on_path] = DENSE_TYPES.index("aligned")
    return r1, r2, types


def select_fgw_target(batch, mode="structure"):
    if mode == "structure":
        return batch["fgw_structure"]
    if mode == "composite":
        return batch["fgw"]
    if mode == "tm_term":
        return batch["tm_term"]
    raise ValueError(f"unknown FGW target {mode!r}, expected one of {FGW_TARGETS}")


def clip_unit(values, enabled=True):
    return values.clamp(0.0, 1.0) if enabled else values


def fgw_target(batch, mode="structure", clip=True):
    """The FGW target a model is trained against, clipped to [0, 1]."""
    return clip_unit(select_fgw_target(batch, mode), clip)


class ProteinDataCache:
    """LRU cache of (coords, embeddings) keyed by UniProt id.

    With pack_dir pointing at a built pack (protein_pack.py), proteins are
    read from it; anything not in the pack, or no pack at all, falls back to
    parsing the PDB and loading the embedding file. Both paths return the
    same arrays.
    """

    def __init__(self, pdb_dir, embedding_dir, max_size=32, pack_dir=None):
        self.pdb_dir = pdb_dir
        self.embedding_dir = embedding_dir
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.pack = open_pack(pack_dir)

    def get(self, uniprot_id):
        if uniprot_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(uniprot_id)
            return self.cache[uniprot_id]

        self.misses += 1
        packed = self.pack.get(uniprot_id) if self.pack is not None else None
        if packed is not None:
            coords, embeddings = packed
        else:
            coords, _ = parse_pdb(pdb_path(self.pdb_dir, uniprot_id))
            embeddings = np.load(embedding_path(self.embedding_dir, uniprot_id))
        embeddings = embeddings.astype(np.float32)

        if len(coords) != len(embeddings):
            raise ValueError(
                f"{uniprot_id}: {len(coords)} coords but "
                f"{len(embeddings)} embeddings"
            )

        value = (coords.astype(np.float32), embeddings)
        self.cache[uniprot_id] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return value

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def lookups(self):
        """Lookups made in this process; zero when loader workers did the loading."""
        return self.hits + self.misses


class EmbeddingStore:
    """Another ESM-2 model's per-residue embeddings, for the student's input only.

    The teacher's patch features stay the pipeline's own 35M embeddings from
    ProteinDataCache; this supplies what the student reads instead. Reads its
    pack when there is one (protein_pack.py --model ...), else the .npy files.
    """

    def __init__(self, embedding_dir, pack_dir=None, max_size=512):
        self.embedding_dir = embedding_dir
        self.max_size = max_size
        self.cache = OrderedDict()
        self.pack = open_pack(pack_dir)

    def get(self, uniprot_id):
        if uniprot_id in self.cache:
            self.cache.move_to_end(uniprot_id)
            return self.cache[uniprot_id]
        packed = self.pack.get(uniprot_id) if self.pack is not None else None
        if packed is not None:
            embeddings = packed[1]
        else:
            embeddings = np.load(embedding_path(self.embedding_dir, uniprot_id))
        value = embeddings.astype(np.float32)
        self.cache[uniprot_id] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return value


def loader_workers(limit=8):
    """Processes for building batches: this job's CPUs minus one for training."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cpus = os.cpu_count() or 1
    return max(0, min(limit, cpus - 1))


def seed_worker(worker_id):
    """Give each loader worker its own stream for the random extra residues.

    Forked workers otherwise inherit identical copies of the dataset's rng.
    """
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def pair_loader(dataset, batch_size, shuffle, workers=0):
    """The DataLoader every trainer and evaluator uses.

    workers > 0 builds batches in that many processes while the GPU trains.
    Pairs that fail to load are then logged by the worker that hit them, and
    the dataset's own errors list stays empty in this process: count failures
    as len(dataset) minus the pairs that arrived in batches.
    """
    extra = {"worker_init_fn": seed_worker, "prefetch_factor": 4} if workers else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pairs,
        num_workers=workers,
        **extra,
    )


def iter_pair_groups(
    csv_path, split="train", chunk_size=5000, max_groups=None, skip_groups=0
):
    """Yield one DataFrame per protein pair, whole.

    skip_groups fast-forwards past groups already processed, so a resumed run
    picks up mid-epoch instead of restarting it.
    """
    carry = None
    groups_yielded = 0
    matched = 0

    with pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunk_size) as reader:
        for chunk in reader:
            if carry is not None:
                chunk = pd.concat([carry, chunk], ignore_index=True)

            last_key = chunk["tm_data_row"].iloc[-1]
            is_last = chunk["tm_data_row"] == last_key
            carry = chunk[is_last]
            body = chunk[~is_last]

            for _, group in body.groupby("tm_data_row", sort=False):
                if split is not None:
                    if row_split(group["id1"].iloc[0], group["id2"].iloc[0]) != split:
                        continue
                matched += 1
                if matched <= skip_groups:
                    continue
                yield group
                groups_yielded += 1
                if max_groups is not None and groups_yielded >= max_groups:
                    return

    if carry is not None and len(carry) > 0:
        if max_groups is not None and groups_yielded >= max_groups:
            return
        if split is None or row_split(carry["id1"].iloc[0], carry["id2"].iloc[0]) == split:
            matched += 1
            if matched > skip_groups:
                yield carry


def iter_group_buffers(csv_path, split="train", buffer_size=256, **kwargs):
    """Batch the group stream into buffers, so a DataLoader can shuffle them."""
    buffer = []
    for group in iter_pair_groups(csv_path, split=split, **kwargs):
        buffer.append(group)
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


class ProteinPairDataset(Dataset):
    """One item = one protein pair."""

    def __init__(
        self,
        groups,
        cache,
        include_sequence=False,
        max_seq_length=1000,
        extra_distill_residues=0,
        dense=None,
        aligned_columns=None,
        student_features=None,
        skip_errors=True,
    ):
        self.groups = groups
        self.cache = cache
        self.include_sequence = include_sequence
        self.max_seq_length = max_seq_length
        self.extra_distill_residues = extra_distill_residues
        self.dense = dense  # DenseConfig, or None for no dense pairs
        self.aligned_columns = aligned_columns  # aligned_columns.AlignedColumns or None
        # EmbeddingStore for the student's input, or None to use the 35M features
        self.student_features = student_features
        self.rng = np.random.default_rng()
        self.skip_errors = skip_errors
        self.errors = []

    def __len__(self):
        return len(self.groups)

    def build_patches(self, coords, features, residue_indices):
        patch_coords, patch_features, patch_masks = [], [], []
        for residue_idx in residue_indices:
            neighbours = knn_indices(coords, int(residue_idx))
            padded_coords, padded_features, mask = pad_patch(
                coords[neighbours], features[neighbours]
            )
            patch_coords.append(padded_coords)
            patch_features.append(padded_features)
            patch_masks.append(mask)
        return (
            np.stack(patch_coords),
            np.stack(patch_features),
            np.stack(patch_masks),
        )

    def __getitem__(self, idx):
        if not self.skip_errors:
            return self.build_item(idx)
        try:
            return self.build_item(idx)
        except Exception as exc:
            self.errors.append(exc)
            log(f"  skipping protein pair {idx}: {exc}")
            return None

    def build_item(self, idx):
        group = self.groups[idx]
        id1 = group["id1"].iloc[0]
        id2 = group["id2"].iloc[0]

        coords1, features1 = self.cache.get(id1)
        coords2, features2 = self.cache.get(id2)

        residue_idx1 = group["residue_idx1"].to_numpy().astype(np.int64)
        residue_idx2 = group["residue_idx2"].to_numpy().astype(np.int64)

        if residue_idx1.max() >= len(coords1) or residue_idx2.max() >= len(coords2):
            raise IndexError(f"{id1}/{id2}: residue index outside protein length")

        pc1, pf1, pm1 = self.build_patches(coords1, features1, residue_idx1)
        pc2, pf2, pm2 = self.build_patches(coords2, features2, residue_idx2)

        item = {
            "patch_coords1": pc1,
            "patch_features1": pf1,
            "patch_mask1": pm1,
            "patch_coords2": pc2,
            "patch_features2": pf2,
            "patch_mask2": pm2,
            "fgw": similarity_from_distortion(
                group["fgw_raw"].to_numpy(), FGW_LABEL_SCALE
            ).astype(np.float32),
            "fgw_structure": similarity_from_distortion(
                group["gw_raw"].to_numpy(), GW_LABEL_SCALE
            ).astype(np.float32),
            "tm_term": group["tm_term"].to_numpy().astype(np.float32),
            "pair_type": np.array(
                [PAIR_TYPE_CODES[str(name)] for name in group["pair_type"]],
                dtype=np.int64,
            ),
            "tm": np.float32(group["tm_score_norm1"].iloc[0]),
            "tm2": np.float32(group["tm_score_norm2"].iloc[0]),
        }

        if self.include_sequence:
            if (
                len(features1) > self.max_seq_length
                or len(features2) > self.max_seq_length
            ):
                raise ValueError(
                    f"{id1}/{id2}: sequence longer than {self.max_seq_length}"
                )
            sequence1, sequence2 = features1, features2
            if self.student_features is not None:
                sequence1 = self.student_features.get(id1)
                sequence2 = self.student_features.get(id2)
                if len(sequence1) != len(coords1) or len(sequence2) != len(coords2):
                    raise ValueError(
                        f"{id1}/{id2}: student embeddings have {len(sequence1)}/"
                        f"{len(sequence2)} residues, structures {len(coords1)}/{len(coords2)}"
                    )
            item["features1"] = sequence1
            item["features2"] = sequence2
            item["residue_idx1"] = residue_idx1
            item["residue_idx2"] = residue_idx2

        if self.extra_distill_residues > 0:
            sides = (("1", coords1, features1), ("2", coords2, features2))
            for side, coords, features in sides:
                count = min(self.extra_distill_residues, len(coords))
                extra_idx = self.rng.choice(len(coords), size=count, replace=False)
                extra_idx = np.sort(extra_idx).astype(np.int64)
                pc, pf, pm = self.build_patches(coords, features, extra_idx)
                item[f"extra_residue_idx{side}"] = extra_idx
                item[f"extra_patch_coords{side}"] = pc
                item[f"extra_patch_features{side}"] = pf
                item[f"extra_patch_mask{side}"] = pm

        if self.dense is not None:
            aligned1, aligned2 = self.dense_anchors(group, residue_idx1, residue_idx2)
            r1, r2, types = sample_dense_residues(
                aligned1, aligned2, len(coords1), len(coords2), self.dense, self.rng
            )
            item["dense_type"] = types
            for side, residues, coords, features in (
                ("1", r1, coords1, features1),
                ("2", r2, coords2, features2),
            ):
                pc, pf, pm = self.build_patches(coords, features, residues)
                item[f"dense_residue_idx{side}"] = residues
                item[f"dense_patch_coords{side}"] = pc
                item[f"dense_patch_features{side}"] = pf
                item[f"dense_patch_mask{side}"] = pm

        return item

    def dense_anchors(self, group, residue_idx1, residue_idx2):
        """Aligned pairs to draw dense anchors from.

        Every TM-align column when the aligned-columns store has this pair,
        else the labelled positives (every 32nd column).
        """
        if self.aligned_columns is not None:
            aligned = self.aligned_columns.get(int(group["tm_data_row"].iloc[0]))
            if aligned is not None and len(aligned[0]) > 0:
                return aligned
        positives = group["pair_type"].to_numpy() == "aligned"
        return residue_idx1[positives], residue_idx2[positives]


def collate_pairs(items):
    """Pad to the batch's largest residue count (and sequence length)."""
    items = [item for item in items if item is not None]
    if not items:
        return None

    batch_size = len(items)
    max_residues = max(len(item["fgw"]) for item in items)
    feature_dim = items[0]["patch_features1"].shape[-1]
    include_sequence = "features1" in items[0]

    out = {
        "tm": torch.zeros(batch_size),
        "tm2": torch.zeros(batch_size),
        "fgw": torch.zeros(batch_size, max_residues),
        "fgw_structure": torch.zeros(batch_size, max_residues),
        "tm_term": torch.zeros(batch_size, max_residues),
        "pair_type": torch.full((batch_size, max_residues), -1, dtype=torch.long),
        "pair_mask": torch.zeros(batch_size, max_residues, dtype=torch.bool),
    }
    for side in ("1", "2"):
        out[f"patch_features{side}"] = torch.zeros(
            batch_size, max_residues, K_NEIGHBORS, feature_dim
        )
        out[f"patch_coords{side}"] = torch.zeros(
            batch_size, max_residues, K_NEIGHBORS, 3
        )
        out[f"patch_mask{side}"] = torch.zeros(
            batch_size, max_residues, K_NEIGHBORS, dtype=torch.bool
        )

    include_extra = "extra_patch_features1" in items[0]
    if include_extra:
        # protein lengths differ, so the two sides can carry different counts
        max_extra = max(
            max(len(item["extra_residue_idx1"]), len(item["extra_residue_idx2"]))
            for item in items
        )
        for side in ("1", "2"):
            out[f"extra_mask{side}"] = torch.zeros(
                batch_size, max_extra, dtype=torch.bool
            )
            out[f"extra_residue_idx{side}"] = torch.zeros(
                batch_size, max_extra, dtype=torch.long
            )
            out[f"extra_patch_features{side}"] = torch.zeros(
                batch_size, max_extra, K_NEIGHBORS, feature_dim
            )
            out[f"extra_patch_coords{side}"] = torch.zeros(
                batch_size, max_extra, K_NEIGHBORS, 3
            )
            out[f"extra_patch_mask{side}"] = torch.zeros(
                batch_size, max_extra, K_NEIGHBORS, dtype=torch.bool
            )

    include_dense = "dense_type" in items[0]
    if include_dense:
        max_dense = {
            side: max(len(item[f"dense_residue_idx{side}"]) for item in items)
            for side in ("1", "2")
        }
        out["dense_type"] = torch.full(
            (batch_size, max_dense["1"], max_dense["2"]), -1, dtype=torch.long
        )
        for side in ("1", "2"):
            count = max_dense[side]
            out[f"dense_mask{side}"] = torch.zeros(batch_size, count, dtype=torch.bool)
            out[f"dense_residue_idx{side}"] = torch.zeros(batch_size, count, dtype=torch.long)
            out[f"dense_patch_features{side}"] = torch.zeros(
                batch_size, count, K_NEIGHBORS, feature_dim
            )
            out[f"dense_patch_coords{side}"] = torch.zeros(batch_size, count, K_NEIGHBORS, 3)
            out[f"dense_patch_mask{side}"] = torch.zeros(
                batch_size, count, K_NEIGHBORS, dtype=torch.bool
            )

    if include_sequence:
        max_len = max(
            max(len(item["features1"]), len(item["features2"])) for item in items
        )
        # the student's input can come from a wider ESM-2 than the patches
        sequence_dim = items[0]["features1"].shape[-1]
        for side in ("1", "2"):
            out[f"features{side}"] = torch.zeros(batch_size, max_len, sequence_dim)
            out[f"seq_mask{side}"] = torch.zeros(
                batch_size, max_len, dtype=torch.bool
            )
            out[f"residue_idx{side}"] = torch.zeros(
                batch_size, max_residues, dtype=torch.long
            )

    for i, item in enumerate(items):
        num_residues = len(item["fgw"])
        out["tm"][i] = float(item["tm"])
        out["tm2"][i] = float(item["tm2"])
        out["fgw"][i, :num_residues] = torch.from_numpy(item["fgw"])
        out["fgw_structure"][i, :num_residues] = torch.from_numpy(
            item["fgw_structure"]
        )
        out["tm_term"][i, :num_residues] = torch.from_numpy(item["tm_term"])
        out["pair_type"][i, :num_residues] = torch.from_numpy(item["pair_type"])
        out["pair_mask"][i, :num_residues] = True

        for side in ("1", "2"):
            for key in ("patch_features", "patch_coords", "patch_mask"):
                out[f"{key}{side}"][i, :num_residues] = torch.from_numpy(
                    item[f"{key}{side}"]
                )
            if include_extra:
                count = len(item[f"extra_residue_idx{side}"])
                out[f"extra_mask{side}"][i, :count] = True
                out[f"extra_residue_idx{side}"][i, :count] = torch.from_numpy(
                    item[f"extra_residue_idx{side}"]
                )
                for key in ("extra_patch_features", "extra_patch_coords",
                            "extra_patch_mask"):
                    out[f"{key}{side}"][i, :count] = torch.from_numpy(
                        item[f"{key}{side}"]
                    )

            if include_dense:
                count = len(item[f"dense_residue_idx{side}"])
                out[f"dense_mask{side}"][i, :count] = True
                out[f"dense_residue_idx{side}"][i, :count] = torch.from_numpy(
                    item[f"dense_residue_idx{side}"]
                )
                for key in ("dense_patch_features", "dense_patch_coords",
                            "dense_patch_mask"):
                    out[f"{key}{side}"][i, :count] = torch.from_numpy(
                        item[f"{key}{side}"]
                    )

            if include_sequence:
                length = len(item[f"features{side}"])
                out[f"features{side}"][i, :length] = torch.from_numpy(
                    item[f"features{side}"]
                )
                out[f"seq_mask{side}"][i, :length] = True
                out[f"residue_idx{side}"][i, :num_residues] = torch.from_numpy(
                    item[f"residue_idx{side}"]
                )

        if include_dense:
            m1, m2 = item["dense_type"].shape
            out["dense_type"][i, :m1, :m2] = torch.from_numpy(item["dense_type"].astype(np.int64))

    return out


def masked_mse(predictions, targets, mask):
    valid = mask.float()
    return (((predictions - targets) ** 2) * valid).sum() / valid.sum().clamp(min=1.0)


def esm_baseline_similarity(batch):
    """Cosine similarity of the two centre residues' raw ESM vectors."""
    centre1 = torch.nn.functional.normalize(
        batch["patch_features1"][:, :, 0, :], dim=-1
    )
    centre2 = torch.nn.functional.normalize(
        batch["patch_features2"][:, :, 0, :], dim=-1
    )
    return (centre1 * centre2).sum(dim=-1)
