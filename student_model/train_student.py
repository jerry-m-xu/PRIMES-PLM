#!/usr/bin/env python3
"""Distill the structural teacher into a sequence-only student."""

import os
import signal
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (
    HERE,
    os.path.join(REPO_ROOT, "teacher_model"),
    os.path.join(REPO_ROOT, "data_scripts"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from egnn_model import SiameseEGNNTeacher  # noqa: E402
from common import log  # noqa: E402
from common import log_failures  # noqa: E402
from fgw import FGW_LABEL_SCALE, GW_LABEL_SCALE  # noqa: E402
from losses import LossBalancer  # noqa: E402
from metrics import log_validation, summarise  # noqa: E402
from pair_data import (  # noqa: E402
    ProteinDataCache,
    ProteinPairDataset,
    clip_unit,
    collate_pairs,
    fgw_target,
    iter_group_buffers,
    masked_mse,
)
from splits import split_summary  # noqa: E402
from student_model import SequenceStudent, distillation_loss  # noqa: E402

# global config
FGW_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
TEACHER_CHECKPOINT = "/jet/home/jxu23/OCEANDIR/teacher_checkpoints/teacher_best.pt"
CHECKPOINT_DIR = "/jet/home/jxu23/OCEANDIR/student_checkpoints"
LATEST_CHECKPOINT_NAME = "student_latest.pt"

CSV_CHUNK_SIZE = 5000
GROUPS_PER_BUFFER = 256
MAX_GROUPS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7

BATCH_SIZE = 4  # protein pairs per step
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_EVERY_BATCHES = 20

MAX_SEQ_LENGTH = 1000
INPUT_DIM = 480
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 1024
DROPOUT = 0.1

CLIP_TARGETS = True
FGW_TARGET = "structure"  # see pair_data.FGW_TARGETS; "composite" rewards echoing ESM

EXTRA_DISTILL_RESIDUES = 16

# prior weights; under LOSS_BALANCE = "uncertainty" each term is also rescaled
# by a learned precision, so the ~0.01-scale label MSEs are not drowned by the
# ~0.3-scale distillation cosine (losses.py)
W_DISTILL = 1.0  # per-residue embedding distillation
W_GLOBAL = 0.5  # global embedding distillation (the teacher's fold-level view)
W_FGW = 1.0
W_TM = 0.2  # both TM normalisations, from the student's own global head
LOSS_BALANCE = "uncertainty"  # or "fixed"

PROTEIN_CACHE_SIZE = 512
CHECKPOINT_EVERY_BUFFERS = 10
VALIDATE_EVERY_BUFFERS = 50
VAL_MAX_GROUPS = 200

# resume the interrupted run in CHECKPOINT_DIR by default; set
# RESUME = False for a clean start after changing hyperparameters
RESUME = True
RESUME_PATH = None

CONFIG_FINGERPRINT = {
    "fgw_target": FGW_TARGET,
    "gw_label_scale": GW_LABEL_SCALE,
    "fgw_label_scale": FGW_LABEL_SCALE,
    "use_tm_head": W_TM > 0,
    "loss_balance": LOSS_BALANCE,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "ff_dim": FF_DIM,
    "teacher_checkpoint": TEACHER_CHECKPOINT,
}

STOP_REQUESTED = False


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log("received SIGTERM: checkpointing and exiting at the next buffer boundary")


signal.signal(signal.SIGTERM, _request_stop)
signal.signal(signal.SIGINT, _request_stop)


def resume_path():
    return RESUME_PATH or os.path.join(CHECKPOINT_DIR, LATEST_CHECKPOINT_NAME)


def load_resume_state(model, balancer, optimizer):
    """Continue a run that was cut short by a wall-clock limit.

    Returns (start_epoch, skip_groups, start_buffer, epoch_loss, epoch_examples,
    best_val). Resume is buffer-granular: at most one buffer of work is redone.
    A checkpoint written at an epoch boundary (epoch_complete) starts the
    next epoch from scratch rather than redoing the finished one.
    """
    path = resume_path()
    if not RESUME or not os.path.exists(path):
        return 1, 0, 0, 0.0, 0, float("inf")

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("balancer_state_dict") is not None:
        balancer.load_state_dict(checkpoint["balancer_state_dict"])
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    saved = checkpoint.get("config", {})
    for key, current in CONFIG_FINGERPRINT.items():
        if key in saved and saved[key] != current:
            log(
                f"WARNING: resuming a run trained with {key}={saved[key]!r} "
                f"but this script is configured for {current!r}"
            )

    epoch = checkpoint.get("epoch", 1)
    buffer_idx = checkpoint.get("buffer_idx", 0)
    epoch_loss = checkpoint.get("epoch_loss", 0.0)
    epoch_examples = checkpoint.get("epoch_examples", 0)
    if checkpoint.get("epoch_complete"):
        epoch, buffer_idx, epoch_loss, epoch_examples = epoch + 1, 0, 0.0, 0
        log(f"resuming {path}: epoch {epoch - 1} complete, starting epoch {epoch}")
    else:
        log(
            f"resuming {path}: epoch {epoch}, after buffer {buffer_idx} "
            f"({buffer_idx * GROUPS_PER_BUFFER} protein pairs already seen this epoch)"
        )
    return (
        epoch,
        buffer_idx * GROUPS_PER_BUFFER,
        buffer_idx,
        epoch_loss,
        epoch_examples,
        checkpoint.get("best_val", float("inf")),
    )


def make_dataset(groups, cache, extra_distill_residues=0):
    return ProteinPairDataset(
        groups,
        cache,
        include_sequence=True,
        max_seq_length=MAX_SEQ_LENGTH,
        extra_distill_residues=extra_distill_residues,
    )


def build_balancer():
    return LossBalancer(
        {"distill": W_DISTILL, "global": W_GLOBAL, "fgw": W_FGW, "tm": W_TM},
        mode=LOSS_BALANCE,
    )


@torch.no_grad()
def teacher_embeddings(teacher, batch, side, prefix=""):
    """Frozen teacher residue embeddings, shaped [B, K, output_dim]."""
    features = batch[f"{prefix}patch_features{side}"]
    coords = batch[f"{prefix}patch_coords{side}"]
    mask = batch[f"{prefix}patch_mask{side}"]

    batch_size, num_residues, k_neighbors, feature_dim = features.shape
    flat = teacher.encode_patches(
        features.reshape(-1, k_neighbors, feature_dim),
        coords.reshape(-1, k_neighbors, 3),
        mask.reshape(-1, k_neighbors),
    )
    return flat.view(batch_size, num_residues, -1)


def student_forward(student, batch):
    return student.forward_pair(
        batch["features1"],
        batch["seq_mask1"],
        batch["residue_idx1"],
        batch["features2"],
        batch["seq_mask2"],
        batch["residue_idx2"],
        pair_mask=batch["pair_mask"],
        extra_residue_idx1=batch.get("extra_residue_idx1"),
        extra_residue_idx2=batch.get("extra_residue_idx2"),
    )


def teacher_global(teacher, teacher_z, pair_mask):
    """The teacher's pooled fold-level embedding, over the sampled residues."""
    pooled = teacher._masked_mean(teacher_z, pair_mask)
    return torch.nn.functional.normalize(pooled, dim=-1)


def compute_losses(student_out, teacher_z1, teacher_z2, batch, balancer, teacher=None):
    pair_mask = batch["pair_mask"]

    distill = 0.5 * (
        distillation_loss(student_out["residue_z1"], teacher_z1, pair_mask)
        + distillation_loss(student_out["residue_z2"], teacher_z2, pair_mask)
    )

    global_distill = torch.zeros((), device=distill.device)
    if teacher is not None and W_GLOBAL > 0 and "global_sampled_z1" in student_out:
        for side, teacher_z in (("1", teacher_z1), ("2", teacher_z2)):
            reference = teacher_global(teacher, teacher_z, pair_mask)
            cosine = (student_out[f"global_sampled_z{side}"] * reference).sum(dim=-1)
            global_distill = global_distill + 0.5 * (1.0 - cosine).mean()

    # unlabelled residues: distillation only, no FGW/TM target exists for them
    if teacher is not None and "extra_z1" in student_out:
        extra_distill = torch.zeros((), device=distill.device)
        for side in ("1", "2"):
            reference = teacher_embeddings(teacher, batch, side, prefix="extra_")
            extra_distill = extra_distill + 0.5 * distillation_loss(
                student_out[f"extra_z{side}"], reference, batch[f"extra_mask{side}"]
            )
        distill = 0.5 * (distill + extra_distill)

    losses = {
        "distill": distill,
        "global": global_distill,
        "fgw": masked_mse(
            student_out["local_similarity"],
            fgw_target(batch, FGW_TARGET, CLIP_TARGETS),
            pair_mask,
        ),
    }

    if "tm_score_pred" in student_out:
        tm1 = torch.mean(
            (student_out["tm_score_pred"] - clip_unit(batch["tm"], CLIP_TARGETS)) ** 2
        )
        tm2 = torch.mean(
            (student_out["tm_score_pred2"] - clip_unit(batch["tm2"], CLIP_TARGETS)) ** 2
        )
        losses["tm"] = 0.5 * (tm1 + tm2)

    total = balancer(losses)
    return total, {name: float(value.detach()) for name, value in losses.items()}


def train_on_buffer(student, teacher, balancer, optimizer, groups, cache, epoch, buffer_idx):
    dataset = make_dataset(groups, cache, EXTRA_DISTILL_RESIDUES)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pairs,
    )

    student.train()
    totals = {"loss": 0.0, "distill": 0.0, "global": 0.0, "fgw": 0.0, "tm": 0.0}
    examples = 0
    skipped = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if batch is None:  # every pair in this batch failed to load
            skipped += 1
            continue
        try:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            teacher_z1 = teacher_embeddings(teacher, batch, "1")
            teacher_z2 = teacher_embeddings(teacher, batch, "2")

            student_out = student_forward(student, batch)
            loss, parts = compute_losses(
                student_out, teacher_z1, teacher_z2, batch, balancer, teacher=teacher
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(balancer.parameters()), max_norm=GRAD_CLIP
            )
            optimizer.step()

            n = batch["tm"].shape[0]
            examples += n
            totals["loss"] += loss.item() * n
            for key, value in parts.items():
                totals[key] += value * n

            if batch_idx % LOG_EVERY_BATCHES == 0:
                log(
                    f"epoch {epoch} buffer {buffer_idx} "
                    f"batch {batch_idx}/{len(loader)} "
                    f"loss={totals['loss'] / max(examples, 1):.6f} "
                    f"(distill={totals['distill'] / max(examples, 1):.4f} "
                    f"global={totals['global'] / max(examples, 1):.4f} "
                    f"fgw={totals['fgw'] / max(examples, 1):.4f} "
                    f"tm={totals['tm'] / max(examples, 1):.4f}) "
                    f"[{balancer.describe()}]"
                )
        except Exception as exc:
            skipped += 1
            log(f"skipped batch in epoch {epoch} buffer {buffer_idx}: {exc}")

    if skipped or dataset.errors:
        log(
            f"epoch {epoch} buffer {buffer_idx}: {skipped} batches skipped, "
            f"{len(dataset.errors)} pairs failed to load"
        )
        log_failures(dataset.errors, len(dataset))
    return totals["loss"] / max(examples, 1), examples


@torch.no_grad()
def validate(student, teacher, cache, max_groups=VAL_MAX_GROUPS):
    student.eval()
    fgw_pred, fgw_true = [], []
    tm_pred, tm_true, tm2_pred, tm2_true = [], [], [], []
    agreement = []

    for groups in iter_group_buffers(
        FGW_CSV,
        split="val",
        buffer_size=GROUPS_PER_BUFFER,
        chunk_size=CSV_CHUNK_SIZE,
        max_groups=max_groups,
    ):
        loader = DataLoader(
            make_dataset(groups, cache),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_pairs,
        )
        for batch in loader:
            if batch is None:
                continue
            try:
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                out = student_forward(student, batch)
                mask = batch["pair_mask"]

                fgw_pred.append(out["local_similarity"][mask].cpu().numpy())
                fgw_true.append(fgw_target(batch, FGW_TARGET, CLIP_TARGETS)[mask].cpu().numpy())
                if "tm_score_pred" in out:
                    tm_pred.append(out["tm_score_pred"].cpu().numpy())
                    tm_true.append(clip_unit(batch["tm"], CLIP_TARGETS).cpu().numpy())
                    tm2_pred.append(out["tm_score_pred2"].cpu().numpy())
                    tm2_true.append(clip_unit(batch["tm2"], CLIP_TARGETS).cpu().numpy())

                for side, key in (("1", "residue_z1"), ("2", "residue_z2")):
                    reference = teacher_embeddings(teacher, batch, side)
                    cosine = (out[key] * reference).sum(dim=-1)
                    agreement.append(cosine[mask].cpu().numpy())
            except Exception as exc:
                log(f"skipped validation batch: {exc}")

    if not fgw_pred:
        log("WARNING: validation split produced no examples")
        return None

    metrics = {"fgw": summarise(fgw_pred, fgw_true)}
    if tm_pred:
        metrics["tm"] = summarise(tm_pred, tm_true)
        metrics["tm2"] = summarise(tm2_pred, tm2_true)
    if agreement:
        metrics["agreement"] = float(np.concatenate(agreement).mean())
    return metrics


def load_teacher():
    if not os.path.exists(TEACHER_CHECKPOINT):
        raise FileNotFoundError(TEACHER_CHECKPOINT)

    checkpoint = torch.load(TEACHER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    teacher = SiameseEGNNTeacher(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        num_rbf=config.get("num_rbf", 16),
        rbf_max_distance=config.get("rbf_max_distance", 20.0),
        dropout=0.0,
        use_tm_head=config.get("use_tm_head", False),
    ).to(DEVICE)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    if not config.get("use_tm_head", False):
        log(
            "WARNING: teacher was trained without a TM head, so its embeddings "
            "carry no global fold signal for the student to distil"
        )

    teacher_fgw_target = config.get("fgw_target")
    if teacher_fgw_target is not None and teacher_fgw_target != FGW_TARGET:
        log(
            f"WARNING: teacher was trained on FGW target "
            f"{teacher_fgw_target!r} but this student is configured for "
            f"{FGW_TARGET!r}. The distilled embeddings encode a different "
            f"notion of similarity than the student's own FGW loss."
        )

    teacher_scale = config.get("gw_label_scale")
    if teacher_scale is not None and teacher_scale != GW_LABEL_SCALE:
        log(
            f"WARNING: teacher was trained with gw_label_scale={teacher_scale} "
            f"but fgw.py now sets {GW_LABEL_SCALE}. Its cosine similarities are "
            f"calibrated to a different label; retrain it or match the scale."
        )

    log(f"loaded teacher from epoch {checkpoint.get('epoch')}")
    return teacher, config


def save_checkpoint(
    student,
    balancer,
    optimizer,
    epoch,
    loss,
    name=None,
    buffer_idx=0,
    val_metrics=None,
    epoch_loss=0.0,
    epoch_examples=0,
    best_val=float("inf"),
    epoch_complete=False,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    name = name or f"student_epoch_{epoch:03d}.pt"
    path = os.path.join(CHECKPOINT_DIR, name)
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "buffer_idx": buffer_idx,
            "epoch_complete": epoch_complete,
            "model_state_dict": student.state_dict(),
            "balancer_state_dict": balancer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "epoch_loss": epoch_loss,
            "epoch_examples": epoch_examples,
            "best_val": best_val,
            "val_metrics": val_metrics,
            "config": {
                "input_dim": INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "output_dim": student.output_dim,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "ff_dim": FF_DIM,
                "max_length": MAX_SEQ_LENGTH + 24,
                "use_tm_head": W_TM > 0,
                "fgw_target": FGW_TARGET,
                "gw_label_scale": GW_LABEL_SCALE,
                "fgw_label_scale": FGW_LABEL_SCALE,
                "loss_balance": LOSS_BALANCE,
                "extra_distill_residues": EXTRA_DISTILL_RESIDUES,
                "teacher_checkpoint": TEACHER_CHECKPOINT,
            },
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def main():
    log("starting train_student.py")
    log(f"FGW CSV: {FGW_CSV}")
    log(f"Teacher: {TEACHER_CHECKPOINT}")
    log(f"Checkpoint dir: {CHECKPOINT_DIR}")
    log(f"Device: {DEVICE}")
    log(f"Protein pairs per batch: {BATCH_SIZE}")
    log(
        f"Loss weights: distill={W_DISTILL} global={W_GLOBAL} "
        f"fgw={W_FGW} tm={W_TM} (balance={LOSS_BALANCE})"
    )
    log(
        f"FGW target: {FGW_TARGET} (label scales: structure={GW_LABEL_SCALE}, "
        f"composite={FGW_LABEL_SCALE})"
    )
    log(f"Extra distillation residues: {EXTRA_DISTILL_RESIDUES} per protein")
    log(f"Split: {split_summary()}")

    for path in (FGW_CSV, PDB_DIR, EMBEDDING_DIR):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    teacher, teacher_config = load_teacher()

    student = SequenceStudent(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=teacher_config["output_dim"],
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_length=MAX_SEQ_LENGTH + 24,
        use_tm_head=W_TM > 0,
    ).to(DEVICE)
    balancer = build_balancer().to(DEVICE)
    log("student loaded onto device")

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(balancer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    (
        start_epoch,
        skip_groups,
        start_buffer,
        resume_loss,
        resume_examples,
        best_val,
    ) = load_resume_state(student, balancer, optimizer)

    for epoch in range(start_epoch, EPOCHS + 1):
        log(f"===== epoch {epoch}/{EPOCHS} =====")
        cache = ProteinDataCache(
            PDB_DIR, EMBEDDING_DIR, max_size=PROTEIN_CACHE_SIZE
        )
        epoch_loss = resume_loss
        epoch_examples = resume_examples
        buffer_idx = start_buffer
        # only the first epoch of a resumed run skips ahead
        resume_loss, resume_examples, start_buffer = 0.0, 0, 0

        for groups in iter_group_buffers(
            FGW_CSV,
            split="train",
            buffer_size=GROUPS_PER_BUFFER,
            chunk_size=CSV_CHUNK_SIZE,
            max_groups=MAX_GROUPS,
            skip_groups=skip_groups,
        ):
            buffer_idx += 1
            log(f"epoch {epoch}: buffer {buffer_idx} ({len(groups)} protein pairs)")
            buffer_loss, n = train_on_buffer(
                student, teacher, balancer, optimizer, groups, cache, epoch, buffer_idx
            )
            epoch_loss += buffer_loss * n
            epoch_examples += n

            if VALIDATE_EVERY_BUFFERS and buffer_idx % VALIDATE_EVERY_BUFFERS == 0:
                log_validation(
                    validate(student, teacher, cache),
                    f"epoch {epoch} buffer {buffer_idx}:",
                )

            if (
                CHECKPOINT_EVERY_BUFFERS and buffer_idx % CHECKPOINT_EVERY_BUFFERS == 0
            ) or STOP_REQUESTED:
                path = save_checkpoint(
                    student,
                    balancer,
                    optimizer,
                    epoch,
                    epoch_loss / max(epoch_examples, 1),
                    name=LATEST_CHECKPOINT_NAME,
                    buffer_idx=buffer_idx,
                    epoch_loss=epoch_loss,
                    epoch_examples=epoch_examples,
                    best_val=best_val,
                )
                log(f"epoch {epoch} buffer {buffer_idx}: saved {path}")

            if STOP_REQUESTED:
                log(f"stopped after epoch {epoch} buffer {buffer_idx}")
                log("resubmit the same command to continue from here")
                return

        skip_groups = 0

        if epoch_examples == 0:
            raise ValueError("No training examples processed")

        avg_loss = epoch_loss / epoch_examples
        val_metrics = validate(student, teacher, cache, max_groups=None)
        log(f"epoch {epoch} finished: train loss={avg_loss:.6f}")
        log_validation(val_metrics, f"epoch {epoch}:")
        log(f"protein cache hit rate: {cache.hit_rate():.1%}")
        log(f"effective loss weights: {balancer.describe()}")

        if val_metrics is not None:
            score = val_metrics["fgw"]["mse"]
            if "tm" in val_metrics:
                score += W_TM * val_metrics["tm"]["mse"]
            if score < best_val:
                best_val = score
                best_path = save_checkpoint(
                    student,
                    balancer,
                    optimizer,
                    epoch,
                    avg_loss,
                    name="student_best.pt",
                    val_metrics=val_metrics,
                    best_val=best_val,
                    epoch_complete=True,
                )
                log(f"new best validation score {score:.6f}: saved {best_path}")

        # the numbered checkpoint and the latest one: both mark the epoch as
        # complete, so a resume starts the next epoch instead of redoing this one
        for name in (None, LATEST_CHECKPOINT_NAME):
            path = save_checkpoint(
                student,
                balancer,
                optimizer,
                epoch,
                avg_loss,
                name=name,
                val_metrics=val_metrics,
                best_val=best_val,
                epoch_complete=True,
            )
            log(f"saved checkpoint: {path}")

    log("done")


if __name__ == "__main__":
    main()
