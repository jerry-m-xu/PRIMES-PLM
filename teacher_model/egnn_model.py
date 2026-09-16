import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, hidden_dim, output_dim, num_layers=2, activation=nn.SiLU):
    layers = []
    current_dim = input_dim

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class RadialBasis(nn.Module):
    """Gaussian expansion of a distance.

    The edge MLP then sees a smooth unit-scale code instead of raw squared
    angstroms (up to several hundred) next to LayerNormed features.
    """

    def __init__(self, num_rbf=16, max_distance=20.0):
        super().__init__()
        if num_rbf < 1:
            raise ValueError("num_rbf must be at least 1")
        centers = torch.linspace(0.0, max_distance, num_rbf)
        spacing = max_distance / max(num_rbf - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (2.0 * spacing**2)
        self.num_rbf = num_rbf

    def forward(self, distance):
        """[..., 1] distances -> [..., num_rbf]."""
        return torch.exp(-self.gamma * (distance - self.centers) ** 2)


class EGNNLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        edge_hidden_dim=None,
        coord_update_scale=0.1,
        num_rbf=16,
        rbf_max_distance=20.0,
    ):
        super().__init__()
        edge_hidden_dim = edge_hidden_dim or hidden_dim

        self.radial = RadialBasis(num_rbf, rbf_max_distance)
        self.edge_mlp = mlp(
            input_dim=(2 * hidden_dim) + num_rbf,
            hidden_dim=edge_hidden_dim,
            output_dim=edge_hidden_dim,
            num_layers=3,
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, 1),
            nn.Tanh(),
        )
        self.node_mlp = mlp(
            input_dim=hidden_dim + edge_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.coord_update_scale = coord_update_scale

    def forward(self, h, x, node_mask=None):
        batch_size, num_nodes, _ = h.shape

        h_i = h[:, :, None, :].expand(batch_size, num_nodes, num_nodes, -1)
        h_j = h[:, None, :, :].expand(batch_size, num_nodes, num_nodes, -1)

        rel = x[:, :, None, :] - x[:, None, :, :]
        distance = torch.sqrt((rel**2).sum(dim=-1, keepdim=True) + 1e-8)

        edge_input = torch.cat([h_i, h_j, self.radial(distance)], dim=-1)
        messages = self.edge_mlp(edge_input)

        eye = torch.eye(num_nodes, device=h.device, dtype=torch.bool)
        edge_mask = ~eye[None, :, :]

        if node_mask is not None:
            node_mask = node_mask.bool()
            edge_mask = edge_mask & node_mask[:, :, None] & node_mask[:, None, :]

        messages = messages * edge_mask[..., None]

        coord_weights = self.coord_mlp(messages) * edge_mask[..., None]
        coord_update = (rel * coord_weights).sum(dim=2)
        denom = edge_mask.sum(dim=2, keepdim=True).clamp(min=1)
        coord_update = coord_update / denom
        x = x + self.coord_update_scale * coord_update

        aggregated = messages.sum(dim=2) / denom
        h_update = self.node_mlp(torch.cat([h, aggregated], dim=-1))
        h = self.node_norm(h + h_update)

        if node_mask is not None:
            h = h * node_mask[..., None]
            x = x * node_mask[..., None]

        return h, x


class EGNNPatchEncoder(nn.Module):

    def __init__(
        self,
        input_dim=480,
        hidden_dim=256,
        output_dim=128,
        num_layers=4,
        edge_hidden_dim=None,
        coord_update_scale=0.1,
        dropout=0.0,
        num_rbf=16,
        rbf_max_distance=20.0,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    coord_update_scale=coord_update_scale,
                    num_rbf=num_rbf,
                    rbf_max_distance=rbf_max_distance,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_features, coords, node_mask=None, center_idx=None):
        coords = coords - coords[:, :1, :]
        h = self.input_proj(node_features)

        if node_mask is not None:
            h = h * node_mask[..., None]

        for layer in self.layers:
            h, coords = layer(h, coords, node_mask=node_mask)

        if center_idx is None:
            center_h = h[:, 0, :]
        elif isinstance(center_idx, int):
            center_h = h[:, center_idx, :]
        else:
            batch_idx = torch.arange(h.shape[0], device=h.device)
            center_h = h[batch_idx, center_idx]

        z = self.output_proj(center_h)
        return F.normalize(z, dim=-1)


class SimilarityCalibration(nn.Module):
    """Learnable affine map from cosine similarity to the label's range.

    Unit-vector cosines span [-1, 1] while the labels live in (0, 1]. Without
    this the embedding geometry itself has to reproduce the label's exact
    scale, which wastes half the cosine's span. Initialised to (cos + 1) / 2.
    """

    def __init__(self, scale=0.5, offset=0.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, cosine):
        return self.scale * cosine + self.offset


class TMScoreHead(nn.Module):
    """The teacher's auxiliary global head, from pooled patch embeddings.

    It sees only the sampled residues' patches, never coverage or length, so
    it is a training signal for the embeddings rather than a TM predictor.
    The student's GlobalHead is the real one.
    """

    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim
        input_dim = embedding_dim * 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, g1, g2):
        pair_features = torch.cat([g1, g2, torch.abs(g1 - g2), g1 * g2], dim=-1)
        return self.mlp(pair_features).squeeze(-1)


class SiameseEGNNTeacher(nn.Module):

    def __init__(self, encoder=None, use_tm_head=True, **encoder_kwargs):
        super().__init__()
        self.encoder = encoder or EGNNPatchEncoder(**encoder_kwargs)
        output_dim = encoder_kwargs.get("output_dim", 128)
        self.calibration = SimilarityCalibration()
        self.use_tm_head = use_tm_head
        self.tm_head = TMScoreHead(output_dim) if use_tm_head else None

    def encode_patches(self, features, coords, mask=None, center_idx=None):
        return self.encoder(features, coords, node_mask=mask, center_idx=center_idx)

    def forward(
        self,
        features1,
        coords1,
        features2,
        coords2,
        mask1=None,
        mask2=None,
        center_idx1=None,
        center_idx2=None,
    ):
        z1 = self.encode_patches(features1, coords1, mask1, center_idx1)
        z2 = self.encode_patches(features2, coords2, mask2, center_idx2)
        cosine_similarity = (z1 * z2).sum(dim=-1)
        cosine_distance = 1 - cosine_similarity

        return {
            "z1": z1,
            "z2": z2,
            "cosine_similarity": cosine_similarity,
            "cosine_distance": cosine_distance,
            "local_similarity": self.calibration(cosine_similarity),
        }

    @staticmethod
    def _masked_mean(z, pair_mask):
        """Pool over residue pairs, ignoring padded slots."""
        if pair_mask is None:
            return z.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).to(z.dtype)
        return (z * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)

    def forward_protein_pair(
        self,
        features1,
        coords1,
        features2,
        coords2,
        mask1=None,
        mask2=None,
        pair_mask=None,
    ):
        """Per-residue local similarities and an auxiliary TM prediction.

        features/coords: [batch, num_pairs, nodes, ...]; pair_mask marks
        which residue-pair slots are real. Pass it for any padded batch.
        "cosine_similarity" is the raw embedding cosine (what the student is
        distilled against); "local_similarity" is its calibrated version,
        the quantity trained against the label.
        """
        batch_size, num_pairs = features1.shape[:2]

        flat_features1 = features1.reshape(batch_size * num_pairs, *features1.shape[2:])
        flat_features2 = features2.reshape(batch_size * num_pairs, *features2.shape[2:])
        flat_coords1 = coords1.reshape(batch_size * num_pairs, *coords1.shape[2:])
        flat_coords2 = coords2.reshape(batch_size * num_pairs, *coords2.shape[2:])
        flat_mask1 = mask1.reshape(batch_size * num_pairs, *mask1.shape[2:])
        flat_mask2 = mask2.reshape(batch_size * num_pairs, *mask2.shape[2:])

        z1 = self.encode_patches(flat_features1, flat_coords1, flat_mask1)
        z2 = self.encode_patches(flat_features2, flat_coords2, flat_mask2)

        z1 = z1.view(batch_size, num_pairs, -1)
        z2 = z2.view(batch_size, num_pairs, -1)
        cosine_similarity = (z1 * z2).sum(dim=-1)

        g1 = F.normalize(self._masked_mean(z1, pair_mask), dim=-1)
        g2 = F.normalize(self._masked_mean(z2, pair_mask), dim=-1)

        outputs = {
            "z1": z1,
            "z2": z2,
            "cosine_similarity": cosine_similarity,
            "local_similarity": self.calibration(cosine_similarity),
            "global_z1": g1,
            "global_z2": g2,
        }

        if self.use_tm_head and self.tm_head is not None:
            outputs["tm_score_pred"] = self.tm_head(g1, g2)

        return outputs
