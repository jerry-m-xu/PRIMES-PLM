#!/usr/bin/env python3
"""Evaluate a student checkpoint on held-out proteins."""

import os
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

from common import log  # noqa: E402
from metrics import report, report_by_group, report_esm_baseline  # noqa: E402
from pair_data import (  # noqa: E402
    PAIR_TYPES,
    ProteinDataCache,
    clip_unit,
    collate_pairs,
    esm_baseline_similarity,
    iter_group_buffers,
)
from splits import split_summary  # noqa: E402
from student_model import SequenceStudent  # noqa: E402
from train_student import (  # noqa: E402
    CHECKPOINT_DIR,
    CLIP_TARGETS,
    CSV_CHUNK_SIZE,
    EMBEDDING_DIR,
    FGW_CSV,
    FGW_TARGET,
    PDB_DIR,
    load_teacher,
    make_dataset,
    student_forward,
    teacher_embeddings,
)

# global config
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "student_best.pt")
SPLIT = "test"
EVAL_BATCH_SIZE = 8
GROUPS_PER_BUFFER = 256
PROTEIN_CACHE_SIZE = 512
MAX_GROUPS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_student(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    student = SequenceStudent(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        dropout=0.0,
        max_length=config["max_length"],
        use_tm_head=config.get("use_tm_head", True),
    ).to(DEVICE)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.eval()
    return student, checkpoint


@torch.no_grad()
def evaluate(student, teacher, cache):
    fgw_pred, tm_pred, tm_true, tm2_pred, tm2_true, agreement = [], [], [], [], [], []
    pair_types = []
    targets = {"structure": [], "composite": [], "tm_term": []}
    esm_baseline = []
    skipped = 0
    buffer_idx = 0

    for groups in iter_group_buffers(
        FGW_CSV,
        split=SPLIT,
        buffer_size=GROUPS_PER_BUFFER,
        chunk_size=CSV_CHUNK_SIZE,
        max_groups=MAX_GROUPS,
    ):
        buffer_idx += 1
        loader = DataLoader(
            make_dataset(groups, cache),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_pairs,
        )
        log(f"eval buffer {buffer_idx}: {len(groups)} protein pairs")

        for batch in loader:
            if batch is None:
                skipped += 1
                continue
            try:
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                out = student_forward(student, batch)
                mask = batch["pair_mask"]

                fgw_pred.append(out["local_similarity"][mask].cpu().numpy())
                targets["structure"].append(
                    clip_unit(batch["fgw_structure"], CLIP_TARGETS)[mask].cpu().numpy()
                )
                targets["composite"].append(
                    clip_unit(batch["fgw"], CLIP_TARGETS)[mask].cpu().numpy()
                )
                targets["tm_term"].append(batch["tm_term"][mask].cpu().numpy())
                pair_types.append(batch["pair_type"][mask].cpu().numpy())
                esm_baseline.append(esm_baseline_similarity(batch)[mask].cpu().numpy())
                if "tm_score_pred" in out:
                    tm_pred.append(out["tm_score_pred"].cpu().numpy())
                    tm_true.append(clip_unit(batch["tm"], CLIP_TARGETS).cpu().numpy())
                    tm2_pred.append(out["tm_score_pred2"].cpu().numpy())
                    tm2_true.append(clip_unit(batch["tm2"], CLIP_TARGETS).cpu().numpy())

                if teacher is not None:
                    for side, key in (("1", "residue_z1"), ("2", "residue_z2")):
                        reference = teacher_embeddings(teacher, batch, side)
                        cosine = (out[key] * reference).sum(dim=-1)
                        agreement.append(cosine[mask].cpu().numpy())
            except Exception as exc:
                skipped += 1
                log(f"skipped eval batch in buffer {buffer_idx}: {exc}")

    if not fgw_pred:
        raise ValueError("No evaluation examples processed")

    return {
        "fgw_pred": np.concatenate(fgw_pred),
        "targets": {k: np.concatenate(v) for k, v in targets.items()},
        "pair_types": np.concatenate(pair_types),
        "esm_baseline": np.concatenate(esm_baseline),
        "tm": (np.concatenate(tm_pred), np.concatenate(tm_true)) if tm_pred else None,
        "tm2": (np.concatenate(tm2_pred), np.concatenate(tm2_true)) if tm2_pred else None,
        "agreement": np.concatenate(agreement) if agreement else None,
        "skipped": skipped,
    }


def main():
    log("starting test_student.py")
    log(f"Checkpoint: {CHECKPOINT_PATH}")
    log(f"FGW CSV: {FGW_CSV}")
    log(f"Device: {DEVICE}")
    log(f"Split: {SPLIT} ({split_summary()})")

    if not os.path.exists(FGW_CSV):
        raise FileNotFoundError(FGW_CSV)

    student, checkpoint = load_student(CHECKPOINT_PATH)
    log(
        f"loaded student: epoch {checkpoint.get('epoch')}, "
        f"train loss {checkpoint.get('loss', float('nan')):.6f}"
    )

    try:
        teacher, _ = load_teacher()
    except FileNotFoundError as exc:
        log(f"teacher unavailable ({exc}); skipping distillation agreement")
        teacher = None

    cache = ProteinDataCache(PDB_DIR, EMBEDDING_DIR, max_size=PROTEIN_CACHE_SIZE)
    results = evaluate(student, teacher, cache)

    predictions = results["fgw_pred"]
    targets = results["targets"]
    trained_on = checkpoint["config"].get("fgw_target", FGW_TARGET)

    log("")
    log("=" * 60)
    report(
        f"FGW structure-only  (trained on: {trained_on})",
        predictions,
        targets["structure"],
    )
    report("FGW composite (fused: 70% structure, 30% ESM cosine)", predictions, targets["composite"])
    report("TM term (per residue pair, from TM-align's superposition)", predictions, targets["tm_term"])

    report_esm_baseline(results["esm_baseline"], predictions, targets)

    report_by_group(
        predictions, targets.get(trained_on, targets["structure"]), results["pair_types"],
        PAIR_TYPES, f"{trained_on} target by pair type (aligned = on TM-align's path)",
    )

    if results["tm"] is not None:
        report("TM   normalised by protein 1 (per protein pair)", *results["tm"])
        report("TM   normalised by protein 2 (per protein pair)", *results["tm2"])
    else:
        log("")
        log("  TM   : student has no TM head")

    if results["agreement"] is not None:
        agreement = results["agreement"]
        log("")
        log("  distillation agreement with teacher (cosine, 1.0 = identical)")
        log(f"    mean           {agreement.mean():.4f}")
        log(f"    median         {np.median(agreement):.4f}")
        log(f"    10th pct       {np.percentile(agreement, 10):.4f}")
        log(f"    below 0.5      {(agreement < 0.5).mean():.1%}")

    log("")
    log(f"  skipped batches  {results['skipped']}")
    log(f"  cache hit rate   {cache.hit_rate():.1%}")
    log("=" * 60)


if __name__ == "__main__":
    main()
