"""Regression metrics and reporting, shared by every training and eval script."""

import numpy as np

from common import log


def average_ranks(values):
    """Ranks with ties averaged, for Spearman correlation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    summed = np.bincount(inverse, weights=ranks)
    return (summed / counts)[inverse]


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(predictions, targets):
    """R^2 here is against predicting the target mean, not against zero."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    error = predictions - targets
    mse = float(np.mean(error**2))
    baseline_mse = float(np.mean((targets - targets.mean()) ** 2))

    return {
        "n": len(targets),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "baseline_mse": baseline_mse,
        "r2": float(1.0 - mse / baseline_mse) if baseline_mse > 0 else float("nan"),
        "pearson": correlation(predictions, targets),
        "spearman": correlation(average_ranks(predictions), average_ranks(targets)),
        "pred_mean": float(predictions.mean()),
        "pred_std": float(predictions.std()),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
    }


def summarise(prediction_chunks, target_chunks):
    """Compact metrics for in-training validation logging."""
    predictions = np.concatenate(prediction_chunks)
    targets = np.concatenate(target_chunks)
    metrics = compute_metrics(predictions, targets)
    return {key: metrics[key] for key in ("n", "mse", "r2", "pearson")}


def log_validation(metrics, prefix):
    """One line per signal. metrics maps a signal name to summarise() output."""
    if metrics is None:
        return
    for name in ("fgw", "tm", "tm2"):
        if name not in metrics:
            continue
        values = metrics[name]
        log(
            f"{prefix} val {name.upper():<3}: n={values['n']} "
            f"mse={values['mse']:.6f} r2={values['r2']:.4f} "
            f"r={values['pearson']:.4f}"
        )
    if "agreement" in metrics:
        log(f"{prefix} teacher agreement (cosine): {metrics['agreement']:.4f}")


def report_by_group(predictions, targets, groups, names, title):
    """One metrics line per group code, e.g. per pair type."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    groups = np.asarray(groups)
    log("")
    log(f"  {title}")
    log(
        f"  {'group':<10}{'n':>9}{'mse':>10}{'r2':>8}{'pearson':>9}"
        f"{'mean pred':>11}{'mean target':>13}"
    )
    for code, name in enumerate(names):
        mask = groups == code
        if not mask.any():
            continue
        values = compute_metrics(predictions[mask], targets[mask])
        log(
            f"  {name:<10}{values['n']:>9}{values['mse']:>10.4f}{values['r2']:>8.3f}"
            f"{values['pearson']:>9.3f}{values['pred_mean']:>11.3f}{values['target_mean']:>13.3f}"
        )


def report(title, predictions, targets):
    """Full metrics block for the eval scripts."""
    metrics = compute_metrics(predictions, targets)
    log("")
    log(f"  {title}")
    log(f"    n              {metrics['n']}")
    log(f"    MSE            {metrics['mse']:.6f}")
    log(f"    RMSE           {metrics['rmse']:.6f}")
    log(f"    MAE            {metrics['mae']:.6f}")
    log(f"    bias           {metrics['bias']:+.6f}")
    log(f"    predict-mean   {metrics['baseline_mse']:.6f}")
    log(f"    R^2            {metrics['r2']:.4f}   <- <=0 means no better")
    log(f"    Pearson r      {metrics['pearson']:.4f}")
    log(f"    Spearman rho   {metrics['spearman']:.4f}")
    log(
        f"    pred mean/std  {metrics['pred_mean']:.4f} / {metrics['pred_std']:.4f}"
        f"   target {metrics['target_mean']:.4f} / {metrics['target_std']:.4f}"
    )
    if metrics["pred_std"] < 0.1 * metrics["target_std"]:
        log("    WARNING: predictions nearly constant (collapsed to the mean)")
    return metrics


def report_by_bucket(predictions, targets, title="FGW error by target range"):
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    log("")
    log(f"  {title}")
    log(f"  {'range':<14}{'n':>9}{'mse':>10}{'mae':>10}{'mean pred':>12}")
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (targets >= low) & (targets < high if high < 1.0 else targets <= high)
        if not mask.any():
            continue
        error = predictions[mask] - targets[mask]
        log(
            f"  [{low:.1f}, {high:.1f})   {mask.sum():>9}"
            f"{np.mean(error**2):>10.4f}{np.mean(np.abs(error)):>10.4f}"
            f"{predictions[mask].mean():>12.4f}"
        )


def report_esm_baseline(baseline, predictions, targets_by_name):
    """How much of each target is recoverable from the input alone?"""
    log("")
    log("  ESM-cosine baseline (no training, input only)")
    log(f"  {'target':<22}{'baseline R^2':>14}{'model R^2':>12}")
    for name, targets in targets_by_name.items():
        baseline_r2 = compute_metrics(baseline, targets)["r2"]
        model_r2 = compute_metrics(predictions, targets)["r2"]
        log(f"  {name:<22}{baseline_r2:>14.4f}{model_r2:>12.4f}")
