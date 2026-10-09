#!/usr/bin/env python3
"""Error distributions and breakdowns from saved test predictions.

    python data_scripts/error_report.py \\
        --pred teacher=$D/teacher_checkpoints/teacher_test_predictions.npz \\
        --pred student=$D/student_checkpoints_dense/student_test_predictions.npz \\
        --out $D/error_report

test_teacher.py and test_student.py save the .npz files. Each --pred adds one
model (name=path); every figure overlays the models given. Writes to --out:

    pis_error_hist.pdf       signed and absolute error of the PIS, per model
    tm_error_hist.pdf        the same for TM-score (normalised by protein 1)
    error_distributions.pdf  both in one 2 x 2 figure, MAE in the legends
    error_by_tm.pdf          box plots of the error by true TM-score interval:
                             TM-score, and the PIS of aligned residue pairs
    calibration.pdf          mean target against mean prediction, in bins
    tables.txt               the tables printed below, also as CSV files

--figures-dir also copies error_distributions.pdf and error_by_tm.pdf there,
e.g. manuscript/figures, where the paper and the slides pick them up.

Models without residue-level output (TM-Vec, from baselines/tmvec_predict.py)
appear in the TM-score panels and tables only. --tm-target max compares every
model with the larger of the two TM-scores instead of the one normalised by
protein 1; TM-Vec predicts a single symmetric number, so that is the fair
target for it, and models predicting both normalisations use the larger of
their two predictions. Models that predict only TM_P (the teacher's
auxiliary head) are left out of the TM-score comparison in that mode.

The tables break the error down by the pair's TM-score and by its sequence
identity over the TM-align alignment (identical residues / aligned residues,
read from tm_scores.csv), the latter being the test of whether the model works
beyond what sequence similarity already shows. TM-score bins follow the usual
reading: below 0.3 unrelated, 0.5 the same-fold threshold.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import DATA_DIR, log  # noqa: E402
from metrics import compute_metrics  # noqa: E402

TM_SCORES_CSV = os.path.join(DATA_DIR, "tm_scores.csv")
PAIR_TYPES = ("aligned", "shifted", "random")
TM_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
IDENTITY_BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]
BOX_INTERVALS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
COLOURS = ["#E69F00", "#0072B2", "#009E73", "#CC79A7"]


def load_predictions(spec):
    name, path = spec.split("=", 1)
    data = dict(np.load(path, allow_pickle=False))
    residues = f"{data['res_pred'].size:,} residue pairs, " if "res_pred" in data else "no residue output, "
    log(f"{name}: {path} ({residues}{data['pair_row'].size:,} protein pairs)")
    return name, data


def sequence_identity(rows, csv_path):
    """{row: identical / aligned residues} over TM-align's alignment."""
    wanted, identity = set(int(r) for r in rows), {}
    for chunk in pd.read_csv(csv_path, usecols=["row", "seqxA", "seqyA"], chunksize=100000,
                             dtype={"seqxA": str, "seqyA": str}, keep_default_na=False):
        chunk = chunk[chunk["row"].isin(wanted)]
        for row, x, y in zip(chunk["row"], chunk["seqxA"], chunk["seqyA"]):
            if int(row) in identity:
                continue
            a = np.frombuffer(x.encode(), dtype=np.uint8)
            b = np.frombuffer(y.encode(), dtype=np.uint8)
            both = (a != ord("-")) & (b != ord("-"))
            identity[int(row)] = float((a[both] == b[both]).mean()) if both.any() else 0.0
        if len(identity) == len(wanted):
            break
    missing = len(wanted) - len(identity)
    if missing:
        log(f"WARNING: {missing} protein pairs not found in {csv_path}; left out of identity tables")
    return identity


def bucket_label(low, high, percent=False):
    if percent:
        return f"{low * 100:.0f}-{min(high, 1.0) * 100:.0f}%" if high <= 1.0 else f">={low * 100:.0f}%"
    return f"[{low:.1f}, {high:.1f})" if high <= 1.0 else f">={low:.1f}"


def bucket_rows(values, buckets):
    for low, high in buckets:
        yield (low, high), (values >= low) & (values < high)


def describe(pred, target):
    if len(pred) < 3:
        return None
    m = compute_metrics(pred, target)
    return {"n": m["n"], "mae": m["mae"], "bias": m["bias"], "r2": m["r2"], "spearman": m["spearman"]}


class Tables:
    """Collects printed tables and writes them to tables.txt and CSV files."""

    def __init__(self, out):
        self.out, self.lines = out, []

    def table(self, title, frame, csv_name):
        text = f"\n{title}\n{frame.to_string(index=False, float_format=lambda v: f'{v:.4f}')}"
        log(text)
        self.lines.append(text)
        frame.to_csv(os.path.join(self.out, csv_name), index=False)

    def write(self):
        with open(os.path.join(self.out, "tables.txt"), "w") as handle:
            handle.write("\n".join(self.lines) + "\n")


def pair_frame(data, identity, name, tm_target="p"):
    """One row per protein pair: TM-score targets, predictions and identity.

    "tm" and "tm_pred" are the target and prediction every TM-score figure and
    table uses: normalised by protein 1, or with tm_target "max" the larger
    of the two normalisations.
    """
    frame = pd.DataFrame({"row": data["pair_row"]})
    for key in ("pair_tm", "pair_tm_pred", "pair_tm2", "pair_tm2_pred"):
        if key in data:
            frame[key[5:]] = data[key]
    if tm_target == "max":
        if "tm2" in frame:
            frame["tm"] = np.maximum(frame["tm"], frame["tm2"])
        if "tm2_pred" in frame:
            frame["tm_pred"] = np.maximum(frame["tm_pred"], frame["tm2_pred"])
        elif "tm_pred" in frame:
            log(f"{name}: predicts only TM_P, so it is left out of the TM-score comparison with --tm-target max")
            frame = frame.drop(columns=["tm_pred"])
    frame["identity"] = frame["row"].map(identity)
    return frame


def residue_frame(data, pairs):
    """One row per residue pair, with its protein pair's TM-score and identity.

    None for models without residue-level output (TM-Vec).
    """
    if "res_pred" not in data:
        return None
    frame = pd.DataFrame({
        "row": data["res_row"], "type": np.asarray(PAIR_TYPES)[data["res_type"]],
        "pred": data["res_pred"], "pis": data["res_pis"],
    })
    lookup = pairs.drop_duplicates("row").set_index("row")
    if "tm" in lookup:
        frame["pair_tm"] = frame["row"].map(lookup["tm"])
    frame["identity"] = frame["row"].map(lookup["identity"])
    return frame


def tm_tables(tables, models):
    for column, buckets, percent, title, csv in (
        ("tm", TM_BUCKETS, False, "TM-score error by true TM-score", "tm_by_tm.csv"),
        ("identity", IDENTITY_BUCKETS, True, "TM-score error by sequence identity", "tm_by_identity.csv"),
    ):
        rows = []
        for name, pairs, _ in models:
            if "tm_pred" not in pairs:
                continue
            for (low, high), mask in bucket_rows(pairs[column].to_numpy(), buckets):
                stats = describe(pairs["tm_pred"][mask].to_numpy(), pairs["tm"][mask].to_numpy())
                if stats:
                    rows.append({"model": name, "bucket": bucket_label(low, high, percent), **stats})
        if rows:
            tables.table(title, pd.DataFrame(rows), csv)

    rows = []
    for name, pairs, _ in models:
        if "tm_pred" not in pairs:
            continue
        true_fold, pred_fold = pairs["tm"] >= 0.5, pairs["tm_pred"] >= 0.5
        hits = (true_fold & pred_fold).sum()
        rows.append({
            "model": name, "n": len(pairs),
            "accuracy": float((true_fold == pred_fold).mean()),
            "precision": float(hits / max(pred_fold.sum(), 1)),
            "recall": float(hits / max(true_fold.sum(), 1)),
        })
    if rows:
        tables.table("Same-fold call at TM-score 0.5",
                     pd.DataFrame(rows), "tm_same_fold.csv")


def pis_tables(tables, models):
    for column, buckets, percent, title, csv in (
        ("pair_tm", TM_BUCKETS, False, "PIS error by the pair's TM-score and residue-pair type", "pis_by_tm.csv"),
        ("identity", IDENTITY_BUCKETS, True, "PIS error by sequence identity and residue-pair type", "pis_by_identity.csv"),
    ):
        rows = []
        for name, _, residues in models:
            if residues is None or column not in residues:
                continue
            values = residues[column].to_numpy()
            for (low, high), mask in bucket_rows(values, buckets):
                for pair_type in PAIR_TYPES:
                    sel = mask & (residues["type"].to_numpy() == pair_type)
                    stats = describe(residues["pred"][sel].to_numpy(), residues["pis"][sel].to_numpy())
                    if stats:
                        rows.append({"model": name, "bucket": bucket_label(low, high, percent),
                                     "type": pair_type, **stats})
        if rows:
            tables.table(title, pd.DataFrame(rows), csv)


def figures(out, models, figures_dir=None):
    import shutil

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # serif text with Computer Modern maths, close to the manuscript without needing LaTeX
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})

    def labelled(name, error):
        return f"{name} (MAE {np.mean(np.abs(error)):.3f})"

    colour_of = {name: COLOURS[k % len(COLOURS)] for k, (name, _, _) in enumerate(models)}

    def hist_pair(series, title, path, span):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        bins = np.linspace(-span, span, 61)
        abs_bins = np.linspace(0, span, 41)
        for name, error in series:
            colour = colour_of[name]
            axes[0].hist(error, bins=bins, alpha=0.55, color=colour, label=labelled(name, error), density=True)
            axes[1].hist(np.abs(error), bins=abs_bins, alpha=0.55, color=colour, label=labelled(name, error), density=True)
        axes[0].axvline(0, color="black", lw=0.8, ls="--")
        axes[0].set(title=f"{title}: error distribution", xlabel="prediction - target", ylabel="density")
        axes[1].set(title=f"{title}: absolute error", xlabel="|prediction - target|", ylabel="density")
        for ax in axes:
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"), dpi=200)
        plt.close(fig)

    hist_pair([(n, r["pred"] - r["pis"]) for n, _, r in models if r is not None], "PIS",
              os.path.join(out, "pis_error_hist.pdf"), 0.6)
    tm_series = [(n, p["tm_pred"] - p["tm"]) for n, p, _ in models if "tm_pred" in p]
    if tm_series:
        hist_pair(tm_series, "TM-score", os.path.join(out, "tm_error_hist.pdf"), 0.5)

    # the paper's figure: PIS and TM-score error, signed and absolute, MAE in the legends
    pis_series = [(n, (r["pred"] - r["pis"]).to_numpy()) for n, _, r in models if r is not None]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2))
    for row, (title, series, span) in enumerate((("PIS", pis_series, 0.6), ("TM-score", tm_series, 0.5))):
        bins, abs_bins = np.linspace(-span, span, 61), np.linspace(0, span, 41)
        for name, error in series:
            colour = colour_of[name]
            axes[row, 0].hist(error, bins=bins, alpha=0.55, color=colour, density=True, label=labelled(name, error))
            axes[row, 1].hist(np.abs(error), bins=abs_bins, alpha=0.55, color=colour, density=True,
                              label=labelled(name, error))
        axes[row, 0].axvline(0, color="black", lw=0.8, ls="--")
        axes[row, 0].set(title=f"({'ac'[row]}) {title}: error", xlabel="prediction $-$ target", ylabel="density")
        axes[row, 1].set(title=f"({'bd'[row]}) {title}: absolute error", xlabel="|prediction $-$ target|",
                         ylabel="density")
        for ax in axes[row]:
            if series:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_distributions.pdf"))
    fig.savefig(os.path.join(out, "error_distributions.png"), dpi=200)
    plt.close(fig)

    # error by true TM-score interval, models side by side (the paper's Figure 7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.8 / max(len(models), 1)
    labels = [bucket_label(low, high) for low, high in BOX_INTERVALS]
    for panel, (ax, title) in enumerate(zip(axes, ("TM-score", "PIS of aligned residue pairs"))):
        for k, (name, pairs, residues) in enumerate(models):
            if panel == 0:
                if "tm_pred" not in pairs:
                    continue
                values, error = pairs["tm"].to_numpy(), (pairs["tm_pred"] - pairs["tm"]).to_numpy()
            else:
                if residues is None or "pair_tm" not in residues:
                    continue
                aligned = residues[residues["type"] == "aligned"]
                values, error = aligned["pair_tm"].to_numpy(), (aligned["pred"] - aligned["pis"]).to_numpy()
            groups = [error[mask] for _, mask in bucket_rows(values, BOX_INTERVALS)]
            positions = np.arange(len(groups)) + (k - (len(models) - 1) / 2) * width
            boxes = ax.boxplot([g if len(g) else [np.nan] for g in groups], positions=positions,
                               widths=width * 0.9, patch_artist=True, showfliers=False)
            for box in boxes["boxes"]:
                box.set(facecolor=COLOURS[k % len(COLOURS)], alpha=0.6)
            ax.plot([], [], color=COLOURS[k % len(COLOURS)], lw=6, alpha=0.6, label=name)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30)
        ax.set(title=f"{title}: error by true TM-score", xlabel="true TM-score of the pair",
               ylabel="prediction - target")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "error_by_tm.pdf"))
    fig.savefig(os.path.join(out, "error_by_tm.png"), dpi=200)
    plt.close(fig)

    # calibration: binned by prediction, so a calibrated model sits on the diagonal
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, title in zip(axes, ("PIS", "TM-score")):
        for k, (name, pairs, residues) in enumerate(models):
            if title == "PIS":
                if residues is None:
                    continue
                pred, target = residues["pred"].to_numpy(), residues["pis"].to_numpy()
            elif "tm_pred" in pairs:
                pred, target = pairs["tm_pred"].to_numpy(), pairs["tm"].to_numpy()
            else:
                continue
            edges = np.quantile(pred, np.linspace(0, 1, 16))
            which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, 14)
            means = [(pred[which == b].mean(), target[which == b].mean()) for b in range(15) if (which == b).any()]
            ax.plot(*zip(*means), "o-", color=COLOURS[k % len(COLOURS)], ms=3, label=name)
        ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--")
        ax.set(title=f"{title} calibration", xlabel="mean prediction (15 bins)", ylabel="mean target",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.pdf"))
    fig.savefig(os.path.join(out, "calibration.png"), dpi=200)
    plt.close(fig)

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        for name in ("error_distributions.pdf", "error_by_tm.pdf"):
            shutil.copy(os.path.join(out, name), os.path.join(figures_dir, name))
        log(f"copied error_distributions.pdf and error_by_tm.pdf to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred", action="append", required=True, help="name=path/to/predictions.npz")
    parser.add_argument("--tm-scores", default=TM_SCORES_CSV)
    parser.add_argument("--tm-target", default="p", choices=("p", "max"),
                        help="TM-score normalised by protein 1 (p), or the larger of the two (max)")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "error_report"))
    parser.add_argument("--figures-dir", default=None,
                        help="also copy the paper figures here, e.g. manuscript/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    loaded = [load_predictions(spec) for spec in args.pred]
    rows = np.unique(np.concatenate([d["pair_row"] for _, d in loaded]))
    log(f"sequence identity for {len(rows):,} protein pairs from {args.tm_scores}")
    identity = sequence_identity(rows, args.tm_scores)

    models = []
    for name, data in loaded:
        pairs = pair_frame(data, identity, name, args.tm_target)
        models.append((name, pairs, residue_frame(data, pairs)))

    tables = Tables(args.out)
    tm_tables(tables, models)
    pis_tables(tables, models)
    tables.write()
    figures(args.out, models, args.figures_dir)
    log(f"\nwrote figures and tables to {args.out}")


if __name__ == "__main__":
    main()
