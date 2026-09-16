"""Multi-task loss balancing, shared by the teacher and student trainers."""

import torch
import torch.nn as nn


class LossBalancer(nn.Module):
    """Combine several losses so that none dominates by scale alone.

    mode "uncertainty" (Kendall, Gal and Cipolla 2018):

        total = sum_k w_k * (exp(-s_k) * L_k + s_k)

    with one learnable s_k = log(sigma_k^2) per term, initialised at 0 so the
    first steps use the fixed weights. Each term's effective weight,
    w_k * exp(-s_k), settles near w_k / L_k, so the ~0.01-scale label MSEs
    are no longer drowned by a distillation cosine that is ten times larger.
    The "+ s_k" term keeps the precisions from growing without bound.

    mode "fixed": total = sum_k w_k * L_k.

    Terms missing from a batch, or given zero weight, are skipped.
    """

    def __init__(self, weights, mode="uncertainty"):
        super().__init__()
        if mode not in ("uncertainty", "fixed"):
            raise ValueError(f"unknown mode {mode!r}, expected 'uncertainty' or 'fixed'")

        self.names = tuple(weights)
        self.mode = mode
        self.register_buffer(
            "weights", torch.tensor([float(weights[name]) for name in self.names])
        )
        self.log_variances = nn.Parameter(torch.zeros(len(self.names)))

    def forward(self, losses):
        total = None
        for index, name in enumerate(self.names):
            loss = losses.get(name)
            if loss is None or float(self.weights[index]) == 0.0:
                continue

            if self.mode == "uncertainty":
                log_variance = self.log_variances[index]
                term = self.weights[index] * (torch.exp(-log_variance) * loss + log_variance)
            else:
                term = self.weights[index] * loss

            total = term if total is None else total + term

        if total is None:
            raise ValueError("no loss to balance: every term was missing or zero-weighted")
        return total

    @torch.no_grad()
    def effective_weights(self):
        precisions = torch.exp(-self.log_variances) if self.mode == "uncertainty" else 1.0
        values = self.weights * precisions
        return {name: float(values[index]) for index, name in enumerate(self.names)}

    def describe(self):
        return " ".join(f"w_{name}={value:.3g}" for name, value in self.effective_weights().items())
