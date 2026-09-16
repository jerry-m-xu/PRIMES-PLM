"""Sequence-only student."""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teacher_model")
if TEACHER_DIR not in sys.path:
    sys.path.insert(0, TEACHER_DIR)

from egnn_model import SimilarityCalibration  # noqa: E402

NUM_SIMILARITY_STATS = 5
SOFT_ALIGNMENT_TEMPERATURE = 0.1
LENGTH_NORMALISER = math.log(1001.0)  # log1p(length) / this is in (0, 1] up to 1000 residues


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=1024):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1).float()
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds positional encoding "
                f"table ({self.encoding.shape[1]})"
            )
        return x + self.encoding[:, : x.shape[1]]


def similarity_statistics(similarity, mask1, mask2, temperature=SOFT_ALIGNMENT_TEMPERATURE):
    """Coverage-like summaries of the residue-by-residue similarity matrix.

    similarity: [B, L1, L2] cosines; mask1: [B, L1]; mask2: [B, L2].
    Returns [B, 5]: the mean over protein 1 of each residue's best partner
    in protein 2, the same from protein 2's side, the two soft-alignment
    pooled similarities (softmax over the partner axis), and the mean over
    all valid pairs. These are what TM-score measures beyond patch
    similarity: how much of each protein finds a good partner.
    """
    valid = mask1[:, :, None] & mask2[:, None, :]
    # cosines are in [-1, 1]; a finite fill keeps softmax free of NaN on
    # fully padded rows, which the masked means then drop anyway
    masked = similarity.masked_fill(~valid, -1e4)

    def masked_mean(values, mask):
        weights = mask.to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    row_best = masked.max(dim=2).values
    col_best = masked.max(dim=1).values
    row_soft = (torch.softmax(masked / temperature, dim=2) * similarity).sum(dim=2)
    col_soft = (torch.softmax(masked / temperature, dim=1) * similarity).sum(dim=1)
    valid_f = valid.to(similarity.dtype)
    mean_all = (similarity * valid_f).sum(dim=(1, 2)) / valid_f.sum(dim=(1, 2)).clamp(min=1.0)

    return torch.stack(
        [
            masked_mean(row_best, mask1),
            masked_mean(col_best, mask2),
            masked_mean(row_soft, mask1),
            masked_mean(col_soft, mask2),
            mean_all,
        ],
        dim=-1,
    )


class GlobalHead(nn.Module):
    """TM-score from what a whole-protein view can see.

    Inputs: the two pooled residue embeddings, coverage statistics of the
    similarity matrix, and both log lengths. Predicts both normalisations,
    by protein 1's length and by protein 2's, since TM-score is asymmetric.
    """

    def __init__(self, embedding_dim, hidden_dim=None, num_stats=NUM_SIMILARITY_STATS):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        input_dim = embedding_dim * 4 + num_stats + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, g1, g2, stats, length1, length2):
        log_lengths = torch.stack(
            [torch.log1p(length1.float()), torch.log1p(length2.float())], dim=-1
        ) / LENGTH_NORMALISER
        features = torch.cat(
            [g1, g2, torch.abs(g1 - g2), g1 * g2, stats, log_lengths.to(g1.dtype)], dim=-1
        )
        return self.mlp(features)


class SequenceStudent(nn.Module):
    def __init__(
        self,
        input_dim=480,
        hidden_dim=256,
        output_dim=128,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_length=1024,
        use_tm_head=True,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.positional = SinusoidalPositionalEncoding(hidden_dim, max_length)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.output_dim = output_dim
        self.calibration = SimilarityCalibration()
        self.use_tm_head = use_tm_head
        self.global_head = GlobalHead(output_dim) if use_tm_head else None

    def forward(self, features, seq_mask=None):
        """features: [B, L, input_dim] -> per-residue unit vectors [B, L, output_dim].

        Padded positions come back as zero vectors.
        """
        h = self.input_proj(features)
        h = self.positional(h)

        padding_mask = None
        if seq_mask is not None:
            padding_mask = ~seq_mask.bool()
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        z = F.normalize(self.output_proj(h), dim=-1)
        if seq_mask is not None:
            z = z * seq_mask.unsqueeze(-1)
        return z

    @staticmethod
    def gather_residues(z, residue_idx):
        """z: [B, L, D], residue_idx: [B, K] -> [B, K, D]."""
        index = residue_idx.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        return torch.gather(z, 1, index)

    @staticmethod
    def masked_mean(z, mask):
        """Mask-aware pooling to a unit vector."""
        weights = mask.unsqueeze(-1).float()
        pooled = (z * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
        return F.normalize(pooled, dim=-1)

    def forward_pair(
        self,
        features1,
        seq_mask1,
        residue_idx1,
        features2,
        seq_mask2,
        residue_idx2,
        pair_mask=None,
        extra_residue_idx1=None,
        extra_residue_idx2=None,
    ):
        """One protein pair per batch row, with K sampled residue pairs each.

        "cosine_similarity" is the raw cosine between the sampled residues'
        embeddings; "local_similarity" is its calibrated version, trained
        against the label. "similarity_matrix" is the full [B, L1, L2]
        cosine matrix, which the global head summarises and which an
        alignment decoder can consume.
        """
        z1 = self.forward(features1, seq_mask1)
        z2 = self.forward(features2, seq_mask2)

        residue_z1 = self.gather_residues(z1, residue_idx1)
        residue_z2 = self.gather_residues(z2, residue_idx2)
        cosine_similarity = (residue_z1 * residue_z2).sum(dim=-1)

        global_seq_z1 = self.masked_mean(z1, seq_mask1)
        global_seq_z2 = self.masked_mean(z2, seq_mask2)

        similarity_matrix = torch.bmm(z1, z2.transpose(1, 2))

        outputs = {
            "residue_z1": residue_z1,
            "residue_z2": residue_z2,
            "cosine_similarity": cosine_similarity,
            "local_similarity": self.calibration(cosine_similarity),
            "global_seq_z1": global_seq_z1,
            "global_seq_z2": global_seq_z2,
            "similarity_matrix": similarity_matrix,
        }

        if pair_mask is not None:
            outputs["global_sampled_z1"] = self.masked_mean(residue_z1, pair_mask)
            outputs["global_sampled_z2"] = self.masked_mean(residue_z2, pair_mask)

        if extra_residue_idx1 is not None:
            outputs["extra_z1"] = self.gather_residues(z1, extra_residue_idx1)
        if extra_residue_idx2 is not None:
            outputs["extra_z2"] = self.gather_residues(z2, extra_residue_idx2)

        if self.use_tm_head and self.global_head is not None:
            stats = similarity_statistics(similarity_matrix, seq_mask1.bool(), seq_mask2.bool())
            tm = self.global_head(
                global_seq_z1,
                global_seq_z2,
                stats,
                seq_mask1.sum(dim=1),
                seq_mask2.sum(dim=1),
            )
            outputs["tm_score_pred"] = tm[:, 0]
            outputs["tm_score_pred2"] = tm[:, 1]
            outputs["similarity_stats"] = stats
        return outputs


def distillation_loss(student_z, teacher_z, mask):
    """1 - cosine, averaged over valid residue slots. Both inputs are unit norm."""
    cosine = (student_z * teacher_z).sum(dim=-1)
    valid = mask.float()
    return ((1.0 - cosine) * valid).sum() / valid.sum().clamp(min=1.0)
