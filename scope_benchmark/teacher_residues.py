"""The teacher's residue embeddings of every SCOPe40 domain, read from its structure.

align_rerank.py --teacher aligns over them as a ceiling: the teacher sees the
coordinates the student has to infer, so its alignments show how much a
student distilled from it could reach.

A residue is encoded as in training (pair_data.ProteinPairDataset.build_patches):
the patch of its K_NEIGHBORS nearest residues by C-alpha distance, itself
first, with their C-alpha coordinates and ESM-2 35M features rounded through
float16, as precompute_esm.py stores them. Domains are read as prepare_scope.py
read them, so residue k is the k-th letter of domains.fasta, and are cut at
MAX_SEQ_LENGTH residues like the student's and ESM-2's.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(REPO_ROOT, "teacher_model")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scope40 import STRUCTURES, read_domain  # noqa: E402
from common import BASE_ESM, ESM_MODELS, log  # noqa: E402
from egnn_model import SiameseEGNNTeacher  # noqa: E402
from patches import K_NEIGHBORS  # noqa: E402
from search_student import MAX_SEQ_LENGTH, esm_features  # noqa: E402

PATCHES_PER_BATCH = 1024


def load_teacher(path, device):
    """The frozen teacher, built as train_student.load_teacher builds it."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["input_dim"] != ESM_MODELS[BASE_ESM][1]:
        raise SystemExit(f"{path} reads {config['input_dim']}-wide features, not ESM-2 {BASE_ESM}'s")
    teacher = SiameseEGNNTeacher(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        num_rbf=config.get("num_rbf", 16),
        rbf_max_distance=config.get("rbf_max_distance", 20.0),
        dropout=0.0,
        use_tm_head=config.get("use_tm_head", False),
    ).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    return teacher.eval()


def patch_indices(coords, k=K_NEIGHBORS):
    """[L, min(k, L)]: each residue's nearest residues, nearest first, ordered as patches.knn_indices orders them."""
    distances = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    return np.argsort(distances, axis=1)[:, :k]


@torch.no_grad()
def encode(teacher, coords, features):
    """One domain's residue embeddings, [L, output_dim]: coords [L, 3] numpy, features [L, E] on the device."""
    device = features.device
    neighbours = torch.from_numpy(patch_indices(coords)).to(device)
    length, nodes = neighbours.shape
    # padded out to K_NEIGHBORS nodes with zeros, as pad_patch pads short proteins
    patch_coords = torch.zeros(length, K_NEIGHBORS, 3, device=device)
    patch_features = torch.zeros(length, K_NEIGHBORS, features.shape[-1], device=device)
    mask = torch.zeros(length, K_NEIGHBORS, dtype=torch.bool, device=device)
    patch_coords[:, :nodes] = torch.from_numpy(coords).to(device)[neighbours]
    patch_features[:, :nodes] = features[neighbours]
    mask[:, :nodes] = True
    return torch.cat([
        teacher.encode_patches(patch_features[a:a + PATCHES_PER_BATCH], patch_coords[a:a + PATCHES_PER_BATCH],
                               mask[a:a + PATCHES_PER_BATCH])
        for a in range(0, length, PATCHES_PER_BATCH)
    ])


@torch.no_grad()
def teacher_residues(path, checkpoint, scope_dir, domains, sequences, device):
    """Every domain's teacher residue embeddings, float16 on the CPU, cached at path
    with the checkpoint's path and modification time."""
    stamp = {"checkpoint": os.path.abspath(checkpoint), "mtime": os.path.getmtime(checkpoint),
             "domains": list(domains)}
    if os.path.exists(path):
        cache = torch.load(path, weights_only=False)
        if all(cache.get(key) == value for key, value in stamp.items()):
            log(f"reusing the teacher's residues in {path}")
            return cache["residues"]
    from embed_esm2 import load_esm

    teacher = load_teacher(checkpoint, device)
    model, _, converter = load_esm(ESM_MODELS[BASE_ESM][0], device=str(device))
    residues = []
    for k, domain in enumerate(domains):
        sequence, coords, _ = read_domain(os.path.join(scope_dir, STRUCTURES, domain))
        if sequence != sequences[domain]:
            raise SystemExit(f"{domain}'s structure no longer reads as its sequence in domains.fasta; "
                             f"rerun prepare_scope.py --force")
        features = esm_features(sequence, model, converter, device)[0]
        residues.append(encode(teacher, coords[:MAX_SEQ_LENGTH], features).half().cpu())
        if (k + 1) % 1000 == 0:
            log(f"  teacher residues {k + 1:,}/{len(domains):,}")
    del teacher, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.save({**stamp, "residues": residues}, path)
    return residues
