#!/usr/bin/env python3
"""TM-Vec's TM-score predictions for our test pairs, in error_report.py's format.

    python baselines/tmvec_predict.py --checkpoint $D/tm_vec_swiss_model.ckpt
    python baselines/tmvec_predict.py --checkpoint ... --config tm_vec_swiss_model_params.json
    python baselines/tmvec_predict.py --checkpoint ... --split val --max-pairs 2000

TM-Vec (Hamamsy et al., 2024) embeds each sequence with ProtT5-XL, passes the
per-residue embeddings through its transformer, and predicts the TM-score of
two proteins as the cosine of their pooled vectors. This follows TM-Vec's own
recipe (rare residues as X, space-separated, the end-of-sequence token dropped,
one sequence at a time without padding), as the preprint's code did.

Writes tmvec_<split>_predictions.npz next to the checkpoint (--out to change)
with pair_row, pair_tm (TM-score normalised by protein 1), pair_tm2 (by
protein 2) and the same prediction for both, since TM-Vec predicts one
symmetric number; error_report.py --tm-target max compares every model with
the larger of the two. It also writes tmvec_timing.json, the per-protein
embedding and per-pair comparison times, which speed_figure.py merges in.

Fairness: TM-Vec's SWISS-MODEL checkpoint was trained on pairs from the same
source table as ours with its own split, so some of our test pairs may be in
its training set. That favours TM-Vec; say so wherever it is compared.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(REPO_ROOT, "data_scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import DATA_DIR, log  # noqa: E402

FGW_CSV = os.path.join(DATA_DIR, "fgw_scores.csv")
TM_SCORES_CSV = os.path.join(DATA_DIR, "tm_scores.csv")
PROTT5 = "Rostlab/prot_t5_xl_uniref50"


def load_tmvec(checkpoint, config_path, device):
    from tm_vec_model import trans_basic_block, trans_basic_block_Config

    config = trans_basic_block_Config.from_json(config_path) if config_path else trans_basic_block_Config()
    model = trans_basic_block.load_from_checkpoint(checkpoint, config=config, map_location=device)
    return model.to(device).eval()


def load_prott5(device):
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(PROTT5, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTT5).to(device).eval()
    if device.type == "cuda":
        model = model.half()  # ProtT5-XL is 3B parameters; half precision as TM-Vec's own scripts allow
    return tokenizer, model


@torch.no_grad()
def embed(sequence, tokenizer, prott5, tmvec, device):
    spaced = " ".join(re.sub(r"[UZOB]", "X", sequence))
    ids = tokenizer(spaced, add_special_tokens=True, return_tensors="pt").to(device)
    residues = prott5(**ids).last_hidden_state[:, :-1].float()  # drop </s>
    padding = torch.zeros(residues.shape[:2], dtype=torch.bool, device=device)
    return tmvec(residues, src_mask=None, src_key_padding_mask=padding)[0]


def test_pairs(split, max_pairs):
    """(row, id1, id2, TM_P, TM_Q) of a split's protein pairs, in file order."""
    from pair_data import iter_pair_groups

    pairs = []
    for group in iter_pair_groups(FGW_CSV, split=split, max_groups=max_pairs):
        first = group.iloc[0]
        pairs.append((int(first["tm_data_row"]), first["id1"], first["id2"],
                      float(first["tm_score_norm1"]), float(first["tm_score_norm2"])))
    return pairs


def sequences_for(pairs, csv_path):
    """{protein id: sequence}, read from tm_scores.csv (parse_pdb's sequences)."""
    wanted = {p for _, a, b, _, _ in pairs for p in (a, b)}
    sequences = {}
    for chunk in pd.read_csv(csv_path, usecols=["id1", "id2", "seq1", "seq2"], chunksize=200000,
                             dtype=str, keep_default_na=False):
        for id_col, seq_col in (("id1", "seq1"), ("id2", "seq2")):
            for pid, seq in zip(chunk[id_col].str.strip(), chunk[seq_col]):
                if pid in wanted:
                    sequences.setdefault(pid, seq)
        if len(sequences) == len(wanted):
            break
    return sequences


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="TM-Vec checkpoint, e.g. tm_vec_swiss_model.ckpt")
    parser.add_argument("--config", default=None, help="TM-Vec params JSON; default: the class defaults")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                   f"tmvec_{args.split}_predictions.npz")

    pairs = test_pairs(args.split, args.max_pairs)
    sequences = sequences_for(pairs, TM_SCORES_CSV)
    missing = {p for _, a, b, _, _ in pairs for p in (a, b)} - set(sequences)
    if missing:
        log(f"WARNING: no sequence for {len(missing)} proteins; their pairs are skipped")
        pairs = [p for p in pairs if p[1] in sequences and p[2] in sequences]
    log(f"{len(pairs):,} {args.split} pairs, {len(sequences):,} proteins")

    tokenizer, prott5 = load_prott5(device)
    tmvec = load_tmvec(args.checkpoint, args.config, device)
    log(f"loaded ProtT5-XL and TM-Vec ({args.checkpoint}) on {device}")

    embeddings, embed_times = {}, []
    ids = sorted({p for _, a, b, _, _ in pairs for p in (a, b)})
    for k, pid in enumerate(ids):
        synchronise(device)
        start = time.perf_counter()
        embeddings[pid] = embed(sequences[pid], tokenizer, prott5, tmvec, device)
        synchronise(device)
        if k >= 3:  # the first calls include CUDA warm-up
            embed_times.append(time.perf_counter() - start)
        if (k + 1) % 1000 == 0:
            log(f"  embedded {k + 1:,}/{len(ids):,}")

    cosine = torch.nn.functional.cosine_similarity
    predictions, compare_times = [], []
    for _, a, b, _, _ in pairs:
        synchronise(device)
        start = time.perf_counter()
        value = cosine(embeddings[a][None], embeddings[b][None]).item()
        compare_times.append(time.perf_counter() - start)
        predictions.append(value)
    predictions = np.clip(np.asarray(predictions, dtype=np.float32), 0.0, 1.0)

    np.savez_compressed(
        out,
        model=np.array("TM-Vec"), split=np.array(args.split), checkpoint=np.array(args.checkpoint),
        pair_row=np.array([p[0] for p in pairs]),
        pair_tm=np.array([p[3] for p in pairs], dtype=np.float32), pair_tm_pred=predictions,
        pair_tm2=np.array([p[4] for p in pairs], dtype=np.float32), pair_tm2_pred=predictions,
    )
    timing = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "steps": {
            "embed_tmvec": {"n": len(embed_times), "median_s": statistics.median(embed_times),
                            "mean_s": statistics.fmean(embed_times)},
            "compare_tmvec": {"n": len(compare_times), "median_s": statistics.median(compare_times),
                              "mean_s": statistics.fmean(compare_times)},
        },
    }
    timing_path = os.path.join(os.path.dirname(out), "tmvec_timing.json")
    with open(timing_path, "w") as handle:
        json.dump(timing, handle, indent=2)

    tm_max = np.maximum([p[3] for p in pairs], [p[4] for p in pairs])
    log(f"MAE against TM-score by protein 1: {np.mean(np.abs(predictions - [p[3] for p in pairs])):.4f}; "
        f"against the larger TM-score: {np.mean(np.abs(predictions - tm_max)):.4f}")
    log(f"wrote {out} and {timing_path}")


if __name__ == "__main__":
    main()
