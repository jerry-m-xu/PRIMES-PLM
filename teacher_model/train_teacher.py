#!/usr/bin/env python3
"""Train the structural teacher on both similarity signals."""

import os
import signal
import sys

import numpy as np
import torch

from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SCRIPTS_DIR = os.path.join(REPO_ROOT, "data_scripts")
for _path in (os.path.dirname(os.path.abspath(__file__)), DATA_SCRIPTS_DIR):
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
from patches import K_NEIGHBORS  # noqa: E402
from splits import split_summary  # noqa: E402

# global config
FGW_CSV = "/jet/home/jxu23/OCEANDIR/fgw_scores.csv"
PDB_DIR = "/jet/home/jxu23/OCEANDIR/pdbs"
EMBEDDING_DIR = "/jet/home/jxu23/OCEANDIR/embeddings"
CHECKPOINT_DIR = "/jet/home/jxu23/OCEANDIR/teacher_checkpoints"
LATEST_CHECKPOINT_NAME = "teacher_latest.pt"

CSV_CHUNK_SIZE = 5000
GROUPS_PER_BUFFER = 256  # protein pairs held in memory before shuffling
MAX_GROUPS = None  # None for the whole train split

CLIP_TARGETS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7

BATCH_SIZE = 4  # protein pairs per step
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_EVERY_BATCHES = 20

INPUT_DIM = 480
HIDDEN_DIM = 256
OUTPUT_DIM = 128
NUM_EGNN_LAYERS = 4
DROPOUT = 0.1
NUM_RBF = 16  # gaussian distance features per edge, spanning 0-20 A
RBF_MAX_DISTANCE = 20.0

USE_TM_HEAD = True
# prior weights; under LOSS_BALANCE = "uncertainty" each term is also rescaled
# by a learned precision, so the ~0.01-scale MSEs are not drowned (losses.py)
W_FGW = 1.0
W_TM = 0.2
LOSS_BALANCE = "uncertainty"  # or "fixed"

FGW_TARGET = "structure"  # see pair_data.FGW_TARGETS

PROTEIN_CACHE_SIZE = 512
CHECKPOINT_EVERY_BUFFERS = 10
VALIDATE_EVERY_BUFFERS = 50  # 0 to validate only at epoch end
VAL_MAX_GROUPS = 200

# resume the interrupted run in CHECKPOINT_DIR by default; set
# RESUME = False for a clean start after changing hyperparameters
RESUME = True
RESUME_PATH = None

CONFIG_FINGERPRINT = {
    "use_tm_head": USE_TM_HEAD,
    "fgw_target": FGW_TARGET,
    "gw_label_scale": GW_LABEL_SCALE,
    "fgw_label_scale": FGW_LABEL_SCALE,
    "w_fgw": W_FGW,
    "w_tm": W_TM,
    "loss_balance": LOSS_BALANCE,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "num_layers": NUM_EGNN_LAYERS,
    "num_rbf": NUM_RBF,
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


def build_model():
    return SiameseEGNNTeacher(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_EGNN_LAYERS,
        dropout=DROPOUT,
        num_rbf=NUM_RBF,
        rbf_max_distance=RBF_MAX_DISTANCE,
        use_tm_head=USE_TM_HEAD,
    )


def build_balancer():
    return LossBalancer({"fgw": W_FGW, "tm": W_TM}, mode=LOSS_BALANCE)


def forward_batch(model, batch):
    return model.forward_protein_pair(
        batch["patch_features1"],
        batch["patch_coords1"],
        batch["patch_features2"],
        batch["patch_coords2"],
        mask1=batch["patch_mask1"],
        mask2=batch["patch_mask2"],
        pair_mask=batch["pair_mask"],
    )


def compute_losses(outputs, batch, balancer):
    pair_mask = batch["pair_mask"]
    losses = {
        "fgw": masked_mse(
            outputs["local_similarity"], fgw_target(batch, FGW_TARGET, CLIP_TARGETS), pair_mask
        )
    }
    if "tm_score_pred" in outputs:
        losses["tm"] = torch.mean(
            (outputs["tm_score_pred"] - clip_unit(batch["tm"], CLIP_TARGETS)) ** 2
        )

    total = balancer(losses)
    return total, {name: float(value.detach()) for name, value in losses.items()}


def train_on_buffer(model, balancer, optimizer, groups, cache, epoch, buffer_idx):
    dataset = ProteinPairDataset(groups, cache)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pairs,
    )

    model.train()
    totals = {"loss": 0.0, "fgw": 0.0, "tm": 0.0}
    examples = 0
    skipped = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if batch is None:  # every pair in this batch failed to load
            skipped += 1
            continue
        try:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            outputs = forward_batch(model, batch)
            loss, parts = compute_losses(outputs, batch, balancer)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(balancer.parameters()), max_norm=GRAD_CLIP
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
                    f"(fgw={totals['fgw'] / max(examples, 1):.4f} "
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
def validate(model, cache, max_groups=VAL_MAX_GROUPS):
    """Loss and correlations on held-out proteins."""
    model.eval()
    fgw_pred, fgw_true, tm_pred, tm_true = [], [], [], []

    for groups in iter_group_buffers(
        FGW_CSV,
        split="val",
        buffer_size=GROUPS_PER_BUFFER,
        chunk_size=CSV_CHUNK_SIZE,
        max_groups=max_groups,
    ):
        loader = DataLoader(
            ProteinPairDataset(groups, cache),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_pairs,
        )
        for batch in loader:
            if batch is None:
                continue
            try:
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                outputs = forward_batch(model, batch)
                mask = batch["pair_mask"]

                fgw_pred.append(outputs["local_similarity"][mask].cpu().numpy())
                fgw_true.append(fgw_target(batch, FGW_TARGET, CLIP_TARGETS)[mask].cpu().numpy())
                if "tm_score_pred" in outputs:
                    tm_pred.append(outputs["tm_score_pred"].cpu().numpy())
                    tm_true.append(clip_unit(batch["tm"], CLIP_TARGETS).cpu().numpy())
            except Exception as exc:
                log(f"skipped validation batch: {exc}")

    if not fgw_pred:
        log("WARNING: validation split produced no examples")
        return None

    metrics = {"fgw": summarise(fgw_pred, fgw_true)}
    if tm_pred:
        metrics["tm"] = summarise(tm_pred, tm_true)
    return metrics


def save_checkpoint(
    model,
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
    name = name or f"teacher_epoch_{epoch:03d}.pt"
    path = os.path.join(CHECKPOINT_DIR, name)
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "buffer_idx": buffer_idx,
            "epoch_complete": epoch_complete,
            "model_state_dict": model.state_dict(),
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
                "output_dim": OUTPUT_DIM,
                "num_layers": NUM_EGNN_LAYERS,
                "num_rbf": NUM_RBF,
                "rbf_max_distance": RBF_MAX_DISTANCE,
                "k_neighbors": K_NEIGHBORS,
                "batch_size": BATCH_SIZE,
                "use_tm_head": USE_TM_HEAD,
                "fgw_target": FGW_TARGET,
                "gw_label_scale": GW_LABEL_SCALE,
                "fgw_label_scale": FGW_LABEL_SCALE,
                "w_fgw": W_FGW,
                "w_tm": W_TM,
                "loss_balance": LOSS_BALANCE,
            },
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic, so a crash mid-write cannot corrupt the file
    return path


def main():
    log("starting train_teacher.py")
    log(f"FGW CSV: {FGW_CSV}")
    log(f"PDB dir: {PDB_DIR}")
    log(f"Embedding dir: {EMBEDDING_DIR}")
    log(f"Checkpoint dir: {CHECKPOINT_DIR}")
    log(f"Device: {DEVICE}")
    log(f"Protein pairs per batch: {BATCH_SIZE}")
    log(f"Epochs: {EPOCHS}")
    log(f"TM head: {USE_TM_HEAD} (w_fgw={W_FGW}, w_tm={W_TM}, balance={LOSS_BALANCE})")
    log(
        f"FGW target: {FGW_TARGET} (label scales: structure={GW_LABEL_SCALE}, "
        f"composite={FGW_LABEL_SCALE})"
    )
    log(f"Split: {split_summary()}")

    for path in (FGW_CSV, PDB_DIR, EMBEDDING_DIR):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = build_model().to(DEVICE)
    balancer = build_balancer().to(DEVICE)
    log("model loaded onto device")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(balancer.parameters()),
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
    ) = load_resume_state(model, balancer, optimizer)

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
                model, balancer, optimizer, groups, cache, epoch, buffer_idx
            )
            epoch_loss += buffer_loss * n
            epoch_examples += n

            if VALIDATE_EVERY_BUFFERS and buffer_idx % VALIDATE_EVERY_BUFFERS == 0:
                log_validation(
                    validate(model, cache), f"epoch {epoch} buffer {buffer_idx}:"
                )

            if (
                CHECKPOINT_EVERY_BUFFERS and buffer_idx % CHECKPOINT_EVERY_BUFFERS == 0
            ) or STOP_REQUESTED:
                path = save_checkpoint(
                    model,
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
        val_metrics = validate(model, cache, max_groups=None)
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
                    model,
                    balancer,
                    optimizer,
                    epoch,
                    avg_loss,
                    name="teacher_best.pt",
                    val_metrics=val_metrics,
                    best_val=best_val,
                    epoch_complete=True,
                )
                log(f"new best validation score {score:.6f}: saved {best_path}")

        # the numbered checkpoint and the latest one: both mark the epoch as
        # complete, so a resume starts the next epoch instead of redoing this one
        for name in (None, LATEST_CHECKPOINT_NAME):
            path = save_checkpoint(
                model,
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
