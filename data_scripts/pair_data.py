"""Protein-pair batching, shared by the teacher and the student."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import embedding_path, log, pdb_path
from fgw import FGW_LABEL_SCALE, GW_LABEL_SCALE, similarity_from_distortion
from parse_pdb import parse_pdb
from patches import K_NEIGHBORS, knn_indices, pad_patch
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
    """LRU cache of (coords, embeddings) keyed by UniProt id."""

    def __init__(self, pdb_dir, embedding_dir, max_size=32):
        self.pdb_dir = pdb_dir
        self.embedding_dir = embedding_dir
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, uniprot_id):
        if uniprot_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(uniprot_id)
            return self.cache[uniprot_id]

        self.misses += 1
        coords, _ = parse_pdb(pdb_path(self.pdb_dir, uniprot_id))
        embeddings = np.load(
            embedding_path(self.embedding_dir, uniprot_id)
        ).astype(np.float32)

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
        skip_errors=True,
    ):
        self.groups = groups
        self.cache = cache
        self.include_sequence = include_sequence
        self.max_seq_length = max_seq_length
        self.extra_distill_residues = extra_distill_residues
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
            item["features1"] = features1
            item["features2"] = features2
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

        return item


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

    if include_sequence:
        max_len = max(
            max(len(item["features1"]), len(item["features2"])) for item in items
        )
        for side in ("1", "2"):
            out[f"features{side}"] = torch.zeros(batch_size, max_len, feature_dim)
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

            if include_sequence:
                length = len(item[f"features{side}"])
                out[f"features{side}"][i, :length] = torch.from_numpy(
                    item[f"features{side}"]
                )
                out[f"seq_mask{side}"][i, :length] = True
                out[f"residue_idx{side}"][i, :num_residues] = torch.from_numpy(
                    item[f"residue_idx{side}"]
                )

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
