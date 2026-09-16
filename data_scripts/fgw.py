"""Patch-level Gromov-Wasserstein: the local structural label.

Distances are divided by one shared constant (DIST_SCALE, in angstrom), not
by each patch's own mean, so every pair lives in the same units, one epsilon
serves all of them, and a scaled-up patch is no longer identical to the
original. Sinkhorn runs in the log domain so epsilon can be small without the
plan losing mass.

The structure label is the GW term at a coupling solved WITHOUT the feature
term (alpha = 1). Letting the feature term steer that coupling deflates the
label exactly where ESM embeddings disagree, i.e. for remote homologues.
The fused objective is still available as a separate, secondary score.

Everything here returns RAW distortions. Turning one into a bounded
similarity, exp(-distortion / scale), is done at training time by
pair_data.py so the scale can be chosen from the data without regenerating
the CSV. similarity_from_distortion() and the default scales live here so
that the audit, the loader and the trainers agree.
"""

import numpy as np

# distance units and solver defaults, shared by fgw_data.py
DIST_SCALE = 10.0  # angstrom; distances are divided by this before GW
EPS = 0.005  # final entropic regularisation, in DIST_SCALE units
EPS_START = 0.05  # annealed geometrically down to EPS over the outer steps
ALPHA = 0.7  # structure weight in the FUSED objective only
OUTER_ITER = 30  # projected-gradient steps
INNER_ITER = 20  # Sinkhorn iterations per step

# training-time label scales: label = exp(-raw / scale). See pair_data.py.
GW_LABEL_SCALE = 0.05
FGW_LABEL_SCALE = 0.1


def pairwise_dist(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff**2).sum(-1))


def cosine_feature_dist(F1, F2):
    """Cosine distance in [0, 2]."""
    F1_norm = F1 / (np.linalg.norm(F1, axis=1, keepdims=True) + 1e-9)
    F2_norm = F2 / (np.linalg.norm(F2, axis=1, keepdims=True) + 1e-9)
    similarity = F1_norm @ F2_norm.T
    return 1 - np.clip(similarity, -1, 1)


def structure_matrices(X, Y):
    """Intra-patch distance matrices in DIST_SCALE units."""
    return pairwise_dist(X) / DIST_SCALE, pairwise_dist(Y) / DIST_SCALE


def feature_matrix(F1, F2):
    """Feature cost in [0, 1], the same units as the structure cost."""
    return cosine_feature_dist(F1, F2) / 2.0


def similarity_from_distortion(distortion, scale):
    """Bounded similarity in (0, 1]; 1 means zero distortion."""
    return np.exp(-np.asarray(distortion, dtype=np.float64) / scale)


def _logsumexp(M, axis):
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn(C, a, b, eps=EPS, n_iter=INNER_ITER):
    """Entropic OT plan with marginals a and b, solved in the log domain.

    exp(-C / eps) underflows once a row's costs exceed a few multiples of
    eps, and the plain multiplicative updates then leak mass. The log-domain
    form is exact for any cost scale.
    """
    log_K = -C / eps
    log_a, log_b = np.log(a), np.log(b)
    f = np.zeros_like(a)
    g = np.zeros_like(b)

    for _ in range(n_iter):
        f = log_a - _logsumexp(log_K + g[None, :], 1)
        g = log_b - _logsumexp(log_K + f[:, None], 0)

    return np.exp(f[:, None] + log_K + g[None, :])


def gw_term(C1, C2, T):
    """Squared-loss Gromov-Wasserstein distortion of the coupling T."""
    a = T.sum(axis=1)
    b = T.sum(axis=0)
    return (
        np.sum((C1**2) * np.outer(a, a))
        + np.sum((C2**2) * np.outer(b, b))
        - 2 * np.trace(C1 @ T @ C2 @ T.T)
    )


def fgw_terms(C1, C2, F1, F2, T, D_feat=None):
    if D_feat is None:
        D_feat = feature_matrix(F1, F2)
    return gw_term(C1, C2, T), np.sum(D_feat * T)


def fgw_loss(C1, C2, F1, F2, T, alpha=ALPHA, D_feat=None):
    term_struct, term_feat = fgw_terms(C1, C2, F1, F2, T, D_feat=D_feat)
    return alpha * term_struct + (1 - alpha) * term_feat


def fgw_cost_matrix(C1, C2, T):
    """Linearised GW cost L(C1, C2) (x) T, the gradient direction for T."""
    a = T.sum(axis=1)
    b = T.sum(axis=0)

    term1 = (C1**2) @ a
    term2 = (C2**2) @ b
    cross = C1 @ T @ C2.T

    return term1[:, None] + term2[None, :] - 2 * cross


def gw_coupling(
    C1,
    C2,
    M_feat=None,
    alpha=1.0,
    eps=EPS,
    eps_start=EPS_START,
    outer_iter=OUTER_ITER,
    inner_iter=INNER_ITER,
):
    """Projected-gradient (entropic) GW; fused when M_feat is given and alpha < 1.

    GW is non-convex, and from the uniform coupling a small epsilon can settle
    on a wrong basin even for two copies of the same patch. Annealing epsilon
    geometrically from eps_start to eps smooths the early steps and sharpens
    the late ones; pass eps_start=eps to disable it.
    """
    n, m = len(C1), len(C2)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)

    T = np.outer(a, b)
    for step in range(outer_iter):
        fraction = step / max(outer_iter - 1, 1)
        eps_step = eps_start * (eps / eps_start) ** fraction
        cost = fgw_cost_matrix(C1, C2, T)
        if M_feat is not None and alpha < 1.0:
            cost = alpha * cost + (1 - alpha) * M_feat
        T = sinkhorn(cost, a, b, eps=eps_step, n_iter=inner_iter)

    return T


def compute_structure_gw(
    X,
    Y,
    eps=EPS,
    outer_iter=OUTER_ITER,
    inner_iter=INNER_ITER,
    return_coupling=False,
):
    """Raw GW distortion between two patches. Features are not consulted."""
    C1, C2 = structure_matrices(X, Y)
    T = gw_coupling(C1, C2, eps=eps, outer_iter=outer_iter, inner_iter=inner_iter)
    distortion = gw_term(C1, C2, T)

    if return_coupling:
        return distortion, T
    return distortion


def compute_fgw_from_features(
    X,
    Y,
    F1,
    F2,
    alpha=ALPHA,
    eps=EPS,
    outer_iter=OUTER_ITER,
    inner_iter=INNER_ITER,
    return_components=False,
):
    """Raw fused distortion alpha * GW + (1 - alpha) * feature, at the fused coupling."""
    C1, C2 = structure_matrices(X, Y)
    M_feat = feature_matrix(F1, F2)

    T = gw_coupling(
        C1, C2, M_feat, alpha=alpha, eps=eps, outer_iter=outer_iter, inner_iter=inner_iter
    )

    term_struct, term_feat = fgw_terms(C1, C2, F1, F2, T, D_feat=M_feat)
    score = alpha * term_struct + (1 - alpha) * term_feat

    if return_components:
        return score, term_struct, term_feat

    return score


def compute_fgw(
    pdb1,
    pdb2,
    alpha=ALPHA,
    eps=EPS,
    outer_iter=OUTER_ITER,
    device="cpu",
    model=None,
    batch_converter=None,
):
    """Whole-protein fused GW between two PDB files, embedding them with ESM-2."""
    from embed_esm2 import get_esm_embeddings, load_esm
    from parse_pdb import parse_pdb

    if model is None or batch_converter is None:
        model, _, batch_converter = load_esm(device=device)

    X, seq1 = parse_pdb(pdb1)
    Y, seq2 = parse_pdb(pdb2)

    F1 = get_esm_embeddings(seq1, model, batch_converter, device=device)
    F2 = get_esm_embeddings(seq2, model, batch_converter, device=device)

    return compute_fgw_from_features(
        X, Y, F1, F2, alpha=alpha, eps=eps, outer_iter=outer_iter
    )
