#!/usr/bin/env python3
"""Tests for the teacher/student pipeline."""

import contextlib
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _name in ("data_scripts", "teacher_model", "student_model"):
    _path = os.path.join(REPO_ROOT, _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common  # noqa: E402
import patches  # noqa: E402
import splits  # noqa: E402

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

try:
    import fgw
    import fgw_data

    HAVE_FGW = True
except ImportError:  # pragma: no cover
    HAVE_FGW = False

if HAVE_TORCH:
    import egnn_model
    import pair_data
    import student_model as student_module
else:  # pragma: no cover
    egnn_model = pair_data = student_module = None

needs_torch = unittest.skipUnless(HAVE_TORCH, "torch not installed")
needs_fgw = unittest.skipUnless(HAVE_FGW, "fgw/fgw_data not importable")


@contextlib.contextmanager
def module_globals(module, **overrides):
    saved = {key: getattr(module, key) for key in overrides}
    for key, value in overrides.items():
        setattr(module, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(module, key, value)


def random_backbone(num_residues=40, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(num_residues, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps * 3.8, axis=0).astype(np.float32)


def proper_rotation(seed=0):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


def ideal_helix(num_residues=16):
    """Alpha-helix CA trace: 2.3 A radius, 100 degrees and 1.5 A rise per residue."""
    angle = np.arange(num_residues) * np.deg2rad(100.0)
    return np.stack(
        [2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * np.arange(num_residues)],
        axis=1,
    ).astype(np.float32)


def compact_patch(num_residues=16, seed=0):
    """Random points in a ~12 A ball, compact like a real k-NN patch."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(num_residues * 4, 3))
    points = points[np.linalg.norm(points, axis=1) < 1.5][:num_residues]
    return (points * 8.0).astype(np.float32)


class TestPatchDefinition(unittest.TestCase):
    def test_center_residue_comes_first(self):
        """The encoder reads the patch centre from index 0."""
        coords = random_backbone(50, seed=2)
        for residue_idx in (0, 7, 25, 49):
            with self.subTest(residue_idx=residue_idx):
                self.assertEqual(patches.knn_indices(coords, residue_idx)[0], residue_idx)

    def test_returns_k_neighbors(self):
        coords = random_backbone(100, seed=3)
        indices = patches.knn_indices(coords, 10)
        self.assertEqual(len(indices), patches.K_NEIGHBORS)
        self.assertEqual(len(set(indices.tolist())), len(indices))

    def test_clamps_to_protein_length(self):
        self.assertEqual(len(patches.knn_indices(random_backbone(5, seed=4), 2)), 5)

    def test_neighbors_ordered_by_distance(self):
        coords = random_backbone(60, seed=5)
        indices = patches.knn_indices(coords, 30)
        distances = np.linalg.norm(coords[indices] - coords[30], axis=1)
        self.assertTrue(np.all(np.diff(distances) >= 0))

    @needs_fgw
    def test_label_generation_uses_the_shared_definition(self):
        self.assertIs(fgw_data.knn_indices, patches.knn_indices)
        self.assertIs(fgw_data.K_NEIGHBORS, patches.K_NEIGHBORS)

    def test_full_patch_is_unpadded(self):
        k = patches.K_NEIGHBORS
        coords = random_backbone(k, seed=6)
        features = np.arange(k * 4, dtype=np.float32).reshape(k, 4)
        padded_coords, padded_features, mask = patches.pad_patch(coords, features)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(padded_coords, coords)
        np.testing.assert_allclose(padded_features, features)

    def test_short_patch_is_zero_padded_and_masked(self):
        num_real = 6
        coords = random_backbone(num_real, seed=7)
        features = np.ones((num_real, 4), dtype=np.float32)
        padded_coords, padded_features, mask = patches.pad_patch(coords, features)
        self.assertEqual(mask.sum(), num_real)
        self.assertFalse(mask[num_real:].any())
        np.testing.assert_allclose(padded_coords[:num_real], coords)
        self.assertEqual(padded_coords[num_real:].sum(), 0.0)
        self.assertEqual(padded_features[num_real:].sum(), 0.0)


class TestDataGenerationHelpers(unittest.TestCase):
    """The three stages append to shared outputs over hand-chosen row windows."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp()

    def test_header_is_written_for_a_zero_byte_file(self):
        """A zero-byte file still needs a header written."""
        path = os.path.join(self.dir, "out.csv")
        open(path, "w").close()

        handle, writer = common.open_appending_writer(path, ["a", "b"])
        writer.writerow({"a": 1, "b": 2})
        handle.close()

        self.assertEqual(open(path).readline().strip(), "a,b")

    def test_header_is_not_repeated_on_append(self):
        path = os.path.join(self.dir, "out.csv")
        for value in (1, 2):
            handle, writer = common.open_appending_writer(path, ["a", "b"])
            writer.writerow({"a": value, "b": value})
            handle.close()
        self.assertEqual(open(path).read().count("a,b"), 1)

    def test_read_done_rows_supports_resume(self):
        path = os.path.join(self.dir, "fgw.csv")
        with open(path, "w") as handle:
            handle.write("source_row,v\n10,x\n10,y\n11,z\n")
        self.assertEqual(common.read_done_rows(path, "source_row"), {10, 11})

    def test_read_done_rows_tolerates_missing_file_and_column(self):
        self.assertEqual(
            common.read_done_rows(os.path.join(self.dir, "nope.csv"), "source_row"),
            set(),
        )
        path = os.path.join(self.dir, "other.csv")
        with open(path, "w") as handle:
            handle.write("a,b\n1,2\n")
        self.assertEqual(common.read_done_rows(path, "source_row"), set())

    def test_iter_row_groups_skips_leading_groups(self):
        class Meta:
            def __init__(self, sizes):
                self.sizes = sizes

            def row_group(self, i):
                return type("G", (), {"num_rows": self.sizes[i]})()

        class Parquet:
            def __init__(self, sizes):
                self.num_row_groups = len(sizes)
                self.metadata = Meta(sizes)

        parquet = Parquet([100, 100, 100, 100])
        self.assertEqual(
            [g for g, _ in common.iter_row_groups(parquet, 0, None)], [0, 1, 2, 3]
        )
        self.assertEqual(
            [g for g, _ in common.iter_row_groups(parquet, 250, None)], [2, 3]
        )
        self.assertEqual(
            [(g, o) for g, o in common.iter_row_groups(parquet, 50, 150)],
            [(0, 0), (1, 100)],
        )

    def test_run_manifest_appends_one_json_line_per_run(self):
        import json

        path = os.path.join(self.dir, "out.csv")
        common.write_run_manifest(path, {"stage": "fgw_data", "start_row": 0})
        manifest = common.write_run_manifest(path, {"stage": "fgw_data", "start_row": 50})

        records = [json.loads(line) for line in open(manifest)]
        self.assertEqual([r["start_row"] for r in records], [0, 50])
        self.assertTrue(all("finished_at" in r for r in records))

    def test_failure_histogram_groups_by_exception_type(self):
        counts = common.summarise_failures(
            [ValueError("a"), ValueError("b"), FileNotFoundError("c")]
        )
        self.assertEqual(counts, {"ValueError": 2, "FileNotFoundError": 1})


class TestDataScriptDependencies(unittest.TestCase):
    def test_fgw_imports_without_the_esm_package(self):
        """fgw.py's numerics must not drag in the ESM install."""
        self.assertNotIn("esm", sys.modules)
        import fgw  # noqa: F401

        self.assertNotIn("esm", sys.modules)

    def test_patch_helpers_do_not_require_torch(self):
        """fgw_data runs on CPU nodes; patches.py must stay numpy-only."""
        source = open(os.path.join(REPO_ROOT, "data_scripts", "patches.py")).read()
        self.assertNotIn("import torch", source)
        common = open(os.path.join(REPO_ROOT, "data_scripts", "common.py")).read()
        self.assertNotIn("import torch", common)


class TestSplits(unittest.TestCase):
    IDS = [f"P{i:05d}" for i in range(3000)]

    def pairs(self):
        return list(zip(self.IDS[::2], self.IDS[1::2]))

    def test_deterministic(self):
        pairs = self.pairs()
        first = [splits.row_split(a, b) for a, b in pairs]
        second = [splits.row_split(a, b) for a, b in pairs]
        self.assertEqual(first, second)

    def test_stable_bucket_does_not_use_salted_hash(self):
        """A regression guard: hash() would differ per process."""
        self.assertEqual(splits.stable_bucket("P00001"), splits.stable_bucket("P00001"))
        self.assertTrue(0 <= splits.stable_bucket("anything") < 100)

    def test_proteins_never_cross_splits(self):
        by_split = {}
        for id1, id2 in self.pairs():
            split = splits.row_split(id1, id2)
            if split != "discard":
                by_split.setdefault(split, set()).update([id1, id2])

        names = list(by_split)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = by_split[names[i]] & by_split[names[j]]
                self.assertEqual(overlap, set(), f"{names[i]} vs {names[j]} share proteins")

    def test_every_split_is_populated(self):
        seen = {splits.row_split(a, b) for a, b in self.pairs()}
        for split in splits.SPLITS:
            self.assertIn(split, seen)

    def test_pair_mode_assigns_every_row(self):
        with module_globals(splits, HOLDOUT_MODE="pair"):
            assignments = {splits.row_split(a, b) for a, b in self.pairs()}
        self.assertNotIn("discard", assignments)


@needs_fgw
class TestAlignedResiduePairs(unittest.TestCase):
    def test_gaps_on_both_sides_are_accounted_for(self):
        pairs = list(fgw_data.iter_aligned_residue_pairs("AB-CD", ":::::", "A-BCD"))
        self.assertEqual(
            [(pos, i1, i2) for pos, i1, i2, _, _, _ in pairs],
            [(0, 0, 0), (3, 2, 2), (4, 3, 3)],
        )

    def test_ungapped_alignment_is_identity(self):
        seq = "ACDEFGHIK"
        pairs = list(fgw_data.iter_aligned_residue_pairs(seq, ":" * len(seq), seq))
        self.assertEqual(len(pairs), len(seq))
        for pos, idx1, idx2, _, _, _ in pairs:
            self.assertEqual((idx1, idx2), (pos, pos))

    def test_marker_outside_seqm_values_is_skipped(self):
        pairs = list(fgw_data.iter_aligned_residue_pairs("ABCD", ":X::", "ABCD"))
        self.assertEqual([p[0] for p in pairs], [0, 2, 3])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            list(fgw_data.iter_aligned_residue_pairs("ABC", "::", "ABC"))


@needs_fgw
class TestFgwProperties(unittest.TestCase):
    """The label is a raw GW distortion in fgw.DIST_SCALE units.

    Patches here are compact, like real k-NN patches. GW is non-convex, so
    on unstructured random walks the solver can land in different basins
    depending on argument order; that is a property of the problem, not a
    bug, and the label is computed once per pair in the id1 -> id2 direction.
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.X = compact_patch(16, seed=1)
        cls.Y = compact_patch(16, seed=2)
        cls.H = ideal_helix(16)
        cls.F1 = rng.normal(size=(16, 32))
        cls.F2 = rng.normal(size=(16, 32))

    def assertRelativelyClose(self, first, second, tolerance):
        self.assertLess(abs(first - second), tolerance * max(abs(first), abs(second)))

    def test_identical_patches_have_zero_distortion(self):
        for patch in (self.X, self.H):
            self.assertLess(fgw.compute_structure_gw(patch, patch), 1e-3)

    def test_structure_label_ignores_features(self):
        """Same structure, rotated, unrelated features: still ~0.

        Training pairs live here (remote homologues never share ESM
        features). The old fused label fell to ~0.2 in this case, because
        the feature term steered the coupling the structure term was read
        from. The fused solver still does that, which is why it is only a
        secondary score.
        """
        rotated = (self.X - self.X.mean(0)) @ proper_rotation(seed=3).T + 40.0
        structure = fgw.compute_structure_gw(self.X, rotated)
        self.assertLess(structure, 1e-3)

        _, fused_structure_term, _ = fgw.compute_fgw_from_features(
            self.X, rotated, self.F1, self.F2, return_components=True
        )
        self.assertGreater(fused_structure_term, structure)

    def test_unrelated_patches_are_further(self):
        same = fgw.compute_structure_gw(self.X, self.X)
        different = fgw.compute_structure_gw(self.X, self.Y)
        self.assertGreater(different, same + 0.01)

    def test_scaled_patch_is_not_identical(self):
        """A shared distance unit keeps absolute size. Per-patch mean
        normalisation used to make a 1.25x copy score as identical."""
        self.assertGreater(fgw.compute_structure_gw(self.X, 1.25 * self.X), 1e-3)

    def test_distance_is_symmetric(self):
        forward = fgw.compute_structure_gw(self.H, self.X)
        backward = fgw.compute_structure_gw(self.X, self.H)
        self.assertRelativelyClose(forward, backward, 0.1)

    def test_fused_score_is_symmetric(self):
        forward = fgw.compute_fgw_from_features(self.H, self.X, self.F1, self.F2)
        backward = fgw.compute_fgw_from_features(self.X, self.H, self.F2, self.F1)
        self.assertRelativelyClose(forward, backward, 0.1)

    def test_sinkhorn_plan_has_requested_marginals(self):
        n = 12
        rng = np.random.default_rng(12)
        cost = np.abs(rng.normal(size=(n, n)))
        a = b = np.ones(n) / n
        plan = fgw.sinkhorn(cost, a, b, eps=0.05, n_iter=200)
        self.assertAlmostEqual(plan.sum(), 1.0, places=4)
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-3)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-3)

    def test_sinkhorn_keeps_mass_at_large_cost_to_eps_ratio(self):
        """Costs 50-100x eps: exp(-C / eps) underflows to ~1e-22 and the old
        multiplicative solver returned a plan with ~1e-28 total mass. The
        log-domain solver must return the requested marginals."""
        n = 12
        rng = np.random.default_rng(13)
        cost = rng.uniform(5.0, 10.0, size=(n, n))
        a = b = np.ones(n) / n
        plan = fgw.sinkhorn(cost, a, b, eps=0.1, n_iter=300)
        self.assertAlmostEqual(plan.sum(), 1.0, places=6)
        np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-4)
        np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-4)

    def test_similarity_from_distortion_is_bounded_and_monotone(self):
        self.assertEqual(fgw.similarity_from_distortion(0.0, 0.05), 1.0)
        values = fgw.similarity_from_distortion(np.array([0.01, 0.1, 5.0]), 0.05)
        self.assertTrue(np.all(values > 0) and np.all(values <= 1))
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_pairwise_dist_known_case(self):
        points = np.array([[0.0, 0, 0], [3.0, 4, 0]])
        np.testing.assert_allclose(fgw.pairwise_dist(points), [[0, 5], [5, 0]], atol=1e-6)


@needs_torch
class TestEncoderInvariants(unittest.TestCase):
    INPUT_DIM = 16
    NUM_NODES = 12

    def setUp(self):
        torch.manual_seed(0)
        self.encoder = egnn_model.EGNNPatchEncoder(
            input_dim=self.INPUT_DIM, hidden_dim=32, output_dim=8, num_layers=2, dropout=0.0
        ).eval()

    def make_patch(self, seed=0):
        rng = np.random.default_rng(seed)
        coords = torch.from_numpy(random_backbone(self.NUM_NODES, seed=seed)[None])
        features = torch.from_numpy(
            rng.normal(size=(1, self.NUM_NODES, self.INPUT_DIM)).astype(np.float32)
        )
        return features, coords

    def test_output_is_unit_norm(self):
        features, coords = self.make_patch(seed=30)
        with torch.no_grad():
            z = self.encoder(features, coords)
        np.testing.assert_allclose(z.norm(dim=-1).numpy(), 1.0, atol=1e-5)

    def test_invariant_to_rotation_and_translation(self):
        features, coords = self.make_patch(seed=31)
        rotation = torch.from_numpy(proper_rotation(seed=31))
        moved = coords @ rotation.T + torch.tensor([5.0, -3.0, 12.0])
        with torch.no_grad():
            z = self.encoder(features, coords)
            z_moved = self.encoder(features, moved)
        np.testing.assert_allclose(z.numpy(), z_moved.numpy(), atol=1e-4)

    def test_masked_slots_do_not_affect_the_embedding(self):
        num_real = 5
        features, coords = self.make_patch(seed=32)
        mask = torch.zeros(1, self.NUM_NODES, dtype=torch.bool)
        mask[:, :num_real] = True

        clean_f, clean_c = features.clone(), coords.clone()
        clean_f[:, num_real:] = 0.0
        clean_c[:, num_real:] = 0.0
        noisy_f, noisy_c = features.clone(), coords.clone()
        noisy_f[:, num_real:] = 99.0
        noisy_c[:, num_real:] = -42.0

        with torch.no_grad():
            z_clean = self.encoder(clean_f, clean_c, node_mask=mask)
            z_noisy = self.encoder(noisy_f, noisy_c, node_mask=mask)
        np.testing.assert_allclose(z_clean.numpy(), z_noisy.numpy(), atol=1e-5)


@needs_torch
class TestPairMasking(unittest.TestCase):
    INPUT_DIM = 16
    NUM_NODES = 10
    REAL_PAIRS = 3
    PADDED_SLOTS = 2

    def setUp(self):
        torch.manual_seed(1)
        self.model = egnn_model.SiameseEGNNTeacher(
            input_dim=self.INPUT_DIM,
            hidden_dim=32,
            output_dim=8,
            num_layers=2,
            dropout=0.0,
            use_tm_head=True,
        ).eval()

        rng = np.random.default_rng(70)
        shape = (1, self.REAL_PAIRS, self.NUM_NODES, self.INPUT_DIM)
        self.features = torch.from_numpy(rng.normal(size=shape).astype(np.float32))
        self.coords = torch.from_numpy(
            rng.normal(size=(1, self.REAL_PAIRS, self.NUM_NODES, 3)).astype(np.float32) * 3.8
        )
        self.node_mask = torch.ones(1, self.REAL_PAIRS, self.NUM_NODES, dtype=torch.bool)

    def pad(self, tensor, fill):
        extra = list(tensor.shape)
        extra[1] = self.PADDED_SLOTS
        return torch.cat([tensor, torch.full(extra, fill, dtype=tensor.dtype)], dim=1)

    def run_model(self, features, coords, node_mask, pair_mask):
        with torch.no_grad():
            return self.model.forward_protein_pair(
                features, coords, features, coords,
                mask1=node_mask, mask2=node_mask, pair_mask=pair_mask,
            )

    def test_padding_does_not_change_the_tm_prediction(self):
        unpadded = self.run_model(
            self.features, self.coords, self.node_mask,
            torch.ones(1, self.REAL_PAIRS, dtype=torch.bool),
        )

        pair_mask = torch.zeros(1, self.REAL_PAIRS + self.PADDED_SLOTS, dtype=torch.bool)
        pair_mask[:, : self.REAL_PAIRS] = True
        padded = self.run_model(
            self.pad(self.features, 7.0),
            self.pad(self.coords, -5.0),
            self.pad(self.node_mask, False),
            pair_mask,
        )

        np.testing.assert_allclose(
            padded["tm_score_pred"].numpy(), unpadded["tm_score_pred"].numpy(), atol=1e-5
        )
        np.testing.assert_allclose(
            padded["global_z1"].numpy(), unpadded["global_z1"].numpy(), atol=1e-5
        )

    def test_omitting_the_pair_mask_corrupts_pooling(self):
        """Documents why pair_mask is required, not optional."""
        unpadded = self.run_model(
            self.features, self.coords, self.node_mask,
            torch.ones(1, self.REAL_PAIRS, dtype=torch.bool),
        )
        with torch.no_grad():
            unmasked = self.model.forward_protein_pair(
                self.pad(self.features, 7.0), self.pad(self.coords, -5.0),
                self.pad(self.features, 7.0), self.pad(self.coords, -5.0),
                mask1=self.pad(self.node_mask, False),
                mask2=self.pad(self.node_mask, False),
                pair_mask=None,
            )
        difference = float(
            np.abs(unmasked["global_z1"].numpy() - unpadded["global_z1"].numpy()).max()
        )
        self.assertGreater(difference, 1e-3)

    def test_student_pooling_is_also_masked(self):
        student = student_module.SequenceStudent(
            input_dim=self.INPUT_DIM, hidden_dim=32, output_dim=8,
            num_layers=1, num_heads=2, ff_dim=32, dropout=0.0, max_length=64,
        ).eval()

        rng = np.random.default_rng(71)
        real_len = 9
        z = torch.from_numpy(rng.normal(size=(1, 14, 8)).astype(np.float32))
        z = torch.nn.functional.normalize(z, dim=-1)
        mask = torch.zeros(1, 14, dtype=torch.bool)
        mask[:, :real_len] = True

        pooled = student.masked_mean(z, mask)
        expected = torch.nn.functional.normalize(z[:, :real_len].mean(dim=1), dim=-1)
        np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=1e-6)


@needs_torch
class TestPairBatching(unittest.TestCase):
    NUM_RESIDUES = 40
    FEATURE_DIM = 8

    def setUp(self):
        import tempfile

        self.num_pairs = 9
        self.rows_per_pair = 7
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        handle.write(
            "tm_data_row,id1,id2,residue_idx1,residue_idx2,"
            "gw_raw,fgw_raw,tm_term,pair_type,tm_score_norm1,tm_score_norm2\n"
        )
        rng = np.random.default_rng(80)
        for pair_idx in range(self.num_pairs):
            tm = rng.random()
            for row_idx in range(self.rows_per_pair):
                pair_type = pair_data.PAIR_TYPES[row_idx % len(pair_data.PAIR_TYPES)]
                handle.write(
                    f"{pair_idx},A{pair_idx:03d},B{pair_idx:03d},"
                    f"{rng.integers(0, self.NUM_RESIDUES)},"
                    f"{rng.integers(0, self.NUM_RESIDUES)},"
                    f"{rng.random() * 0.2:.4f},{rng.random() * 0.2:.4f},"
                    f"{rng.random():.4f},{pair_type},{tm:.4f},{tm * 0.9:.4f}\n"
                )
        handle.close()
        self.csv = handle.name
        self.addCleanup(os.unlink, self.csv)

    def groups(self, chunk_size):
        return list(
            pair_data.iter_pair_groups(self.csv, split=None, chunk_size=chunk_size)
        )

    def test_groups_survive_chunk_boundaries(self):
        """A pair's rows are contiguous but can straddle a read chunk."""
        for chunk_size in (3, 5, 7, 8, 1000):
            with self.subTest(chunk_size=chunk_size):
                groups = self.groups(chunk_size)
                self.assertEqual(len(groups), self.num_pairs)
                self.assertTrue(all(len(g) == self.rows_per_pair for g in groups))
                self.assertTrue(all(g["id1"].nunique() == 1 for g in groups))

    def test_every_row_is_emitted_exactly_once(self):
        groups = self.groups(4)
        keys = sorted(int(g["tm_data_row"].iloc[0]) for g in groups)
        self.assertEqual(keys, list(range(self.num_pairs)))

    def make_cache(self):
        outer = self

        class FakeCache:
            def get(self, protein_id):
                rng = np.random.default_rng(abs(hash(protein_id)) % 2**31)
                return (
                    random_backbone(outer.NUM_RESIDUES, seed=3),
                    rng.normal(size=(outer.NUM_RESIDUES, outer.FEATURE_DIM)).astype(
                        np.float32
                    ),
                )

        return FakeCache()

    def test_collate_pads_and_masks_variable_residue_counts(self):
        groups = self.groups(1000)
        trimmed = [groups[0].iloc[:3], groups[1]]  # 3 residues vs 7
        dataset = pair_data.ProteinPairDataset(trimmed, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        k = patches.K_NEIGHBORS
        self.assertEqual(batch["pair_mask"].shape, (2, 7))
        self.assertEqual(batch["pair_mask"][0].sum().item(), 3)
        self.assertEqual(batch["pair_mask"][1].sum().item(), 7)
        self.assertEqual(
            batch["patch_features1"].shape, (2, 7, k, self.FEATURE_DIM)
        )
        self.assertEqual(batch["fgw"].shape, (2, 7))
        self.assertEqual(batch["tm"].shape, (2,))
        # padded slots are zeroed
        self.assertEqual(batch["patch_features1"][0, 3:].abs().sum().item(), 0.0)
        self.assertFalse(batch["patch_mask1"][0, 3:].any())

    def test_sequence_mode_adds_sequence_tensors(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), include_sequence=True
        )
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])
        for key in ("features1", "seq_mask1", "residue_idx1"):
            self.assertIn(key, batch)
        self.assertEqual(batch["features1"].shape[-1], self.FEATURE_DIM)
        self.assertEqual(batch["seq_mask1"].sum().item(), 2 * self.NUM_RESIDUES)

    def test_targets_are_exp_of_the_raw_distortions(self):
        """The CSV holds raw distortions; the loader maps them into (0, 1]."""
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        item = dataset[0]
        expected_structure = np.exp(
            -groups[0]["gw_raw"].to_numpy() / pair_data.GW_LABEL_SCALE
        )
        expected_composite = np.exp(
            -groups[0]["fgw_raw"].to_numpy() / pair_data.FGW_LABEL_SCALE
        )
        np.testing.assert_allclose(item["fgw_structure"], expected_structure, rtol=1e-5)
        np.testing.assert_allclose(item["fgw"], expected_composite, rtol=1e-5)
        self.assertTrue(np.all(item["fgw_structure"] > 0))
        self.assertTrue(np.all(item["fgw_structure"] <= 1))

    def test_select_fgw_target_picks_the_right_column(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        np.testing.assert_allclose(
            pair_data.select_fgw_target(batch, "structure").numpy(),
            batch["fgw_structure"].numpy(),
        )
        np.testing.assert_allclose(
            pair_data.select_fgw_target(batch, "composite").numpy(),
            batch["fgw"].numpy(),
        )
        # the two targets must actually differ, or the fix is a no-op
        self.assertGreater(
            float(np.abs(batch["fgw_structure"] - batch["fgw"]).max()), 1e-3
        )
        with self.assertRaises(ValueError):
            pair_data.select_fgw_target(batch, "nonsense")

    def test_esm_baseline_uses_the_centre_residue(self):
        """knn puts the centre at patch index 0; the baseline must read it."""
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        baseline = pair_data.esm_baseline_similarity(batch)
        self.assertEqual(baseline.shape, batch["fgw"].shape)
        self.assertTrue(torch.all(baseline >= -1.001))
        self.assertTrue(torch.all(baseline <= 1.001))

        centre1 = torch.nn.functional.normalize(
            batch["patch_features1"][0, 0, 0], dim=-1
        )
        centre2 = torch.nn.functional.normalize(
            batch["patch_features2"][0, 0, 0], dim=-1
        )
        self.assertAlmostEqual(
            baseline[0, 0].item(), float((centre1 * centre2).sum()), places=5
        )

    def test_extra_distill_residues_are_sampled_and_masked(self):
        """Unlabelled residues exist only as distillation targets."""
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), include_sequence=True,
            extra_distill_residues=11,
        )
        batch = pair_data.collate_pairs([dataset[0], dataset[1]])

        k = patches.K_NEIGHBORS
        for side in ("1", "2"):
            self.assertEqual(batch[f"extra_mask{side}"].shape, (2, 11))
            self.assertTrue(batch[f"extra_mask{side}"].all())
            self.assertEqual(
                batch[f"extra_patch_features{side}"].shape,
                (2, 11, k, self.FEATURE_DIM),
            )
            indices = batch[f"extra_residue_idx{side}"]
            self.assertTrue(int(indices.max()) < self.NUM_RESIDUES)
            # sampled without replacement
            for row in indices:
                self.assertEqual(len(set(row.tolist())), 11)

    def test_extra_distill_clamps_to_protein_length(self):
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(
            groups, self.make_cache(), extra_distill_residues=self.NUM_RESIDUES + 50
        )
        item = dataset[0]
        self.assertEqual(len(item["extra_residue_idx1"]), self.NUM_RESIDUES)

    def test_extra_residues_absent_when_disabled(self):
        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        self.assertNotIn("extra_residue_idx1", dataset[0])

    def test_max_groups_is_respected_including_the_trailing_group(self):
        for limit in (1, 4, self.num_pairs, self.num_pairs + 5):
            with self.subTest(limit=limit):
                groups = list(
                    pair_data.iter_pair_groups(
                        self.csv, split=None, chunk_size=5, max_groups=limit
                    )
                )
                self.assertEqual(len(groups), min(limit, self.num_pairs))

    def test_unreadable_pair_becomes_None_not_an_exception(self):
        """A dataset failure must not escape the loop and kill the run."""
        class BrokenCache:
            def get(self, protein_id):
                raise ValueError(f"{protein_id}: 70 coords but 140 embeddings")

        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, BrokenCache())
        self.assertIsNone(dataset[0])
        self.assertEqual(len(dataset.errors), 1)

    def test_skip_errors_False_still_raises(self):
        class BrokenCache:
            def get(self, protein_id):
                raise ValueError("boom")

        groups = self.groups(1000)[:1]
        dataset = pair_data.ProteinPairDataset(
            groups, BrokenCache(), skip_errors=False
        )
        with self.assertRaises(ValueError):
            dataset[0]

    def test_collate_drops_failed_items(self):
        groups = self.groups(1000)[:2]
        dataset = pair_data.ProteinPairDataset(groups, self.make_cache())
        good = dataset[0]

        batch = pair_data.collate_pairs([good, None])
        self.assertEqual(batch["pair_mask"].shape[0], 1)

    def test_collate_returns_None_when_every_item_failed(self):
        self.assertIsNone(pair_data.collate_pairs([None, None]))

    def test_masked_mse_ignores_padded_slots(self):
        predictions = torch.tensor([[1.0, 2.0, 99.0]])
        targets = torch.tensor([[1.0, 4.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        # only the second slot contributes: (2-4)^2 / 2 valid slots = 2.0
        self.assertAlmostEqual(
            pair_data.masked_mse(predictions, targets, mask).item(), 2.0, places=5
        )


def pdb_atom_line(serial, resname, chain, resseq, xyz):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  CA  {resname} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def write_fake_pdb(path, coords, chain="A", resname="ALA"):
    """One CA per residue, one chain."""
    write_multichain_pdb(path, [(chain, coords, resname)])


def write_multichain_pdb(path, chains, models=1):
    """chains: [(chain_id, coords, resname)], written in order, `models` times.

    Each model is wrapped in MODEL/ENDMDL when there is more than one, with
    coordinates offset per model so the copies are distinguishable.
    """
    with open(path, "w") as handle:
        serial = 1
        for model_idx in range(models):
            if models > 1:
                handle.write(f"MODEL     {model_idx + 1:4d}\n")
            for chain_id, coords, resname in chains:
                for resseq, xyz in enumerate(np.asarray(coords) + 100.0 * model_idx, start=1):
                    handle.write(pdb_atom_line(serial, resname, chain_id, resseq, xyz))
                    serial += 1
                handle.write("TER\n")
            if models > 1:
                handle.write("ENDMDL\n")
        handle.write("END\n")


class TestParquetStreaming(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("pyarrow not installed")

    def test_window_and_batches_agree_with_row_indices(self):
        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"a": [f"A{i}" for i in range(12)], "b": list(range(12))})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "t.parquet")
            pq.write_table(table, path, row_group_size=5)  # groups of 5, 5, 2
            parquet_file = pq.ParquetFile(path)
            self.assertEqual(parquet_file.num_row_groups, 3)

            rows = list(common.iter_parquet_rows(parquet_file, ["a", "b"], 3, 9, batch_size=2))
            self.assertEqual([index for index, _ in rows], list(range(3, 9)))
            self.assertEqual([row["b"] for _, row in rows], list(range(3, 9)))
            self.assertEqual(rows[0][1]["a"], "A3")

            everything = list(common.iter_parquet_rows(parquet_file, ["b"], 0, None))
            self.assertEqual(len(everything), 12)


class TestSuperposition(unittest.TestCase):
    def test_round_trip(self):
        translation = [1.5, -2.0, 3.25]
        rotation = np.eye(3)
        text = common.format_superposition(translation, rotation)
        self.assertEqual(len(text.split()), 12)
        t, u = common.parse_superposition(text)
        np.testing.assert_allclose(t, translation)
        np.testing.assert_allclose(u, rotation)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            common.parse_superposition("1 2 3")


@needs_fgw
class TestLocalLabelSampling(unittest.TestCase):
    def test_tm_d0_matches_tm_align(self):
        self.assertEqual(fgw_data.tm_d0(10), 0.5)
        self.assertAlmostEqual(fgw_data.tm_d0(141), 1.24 * 126 ** (1 / 3) - 1.8, places=6)

    def test_tm_term_is_one_at_zero_distance(self):
        self.assertEqual(fgw_data.tm_term(0.0, 4.0), 1.0)
        self.assertAlmostEqual(fgw_data.tm_term(4.0, 4.0), 0.5)

    def test_negatives_are_off_path_and_alternate_types(self):
        aligned = [(k, k, k + 3, "A", ":", "A") for k in range(100)]  # i -> i + 3
        rng = np.random.default_rng(0)
        with module_globals(fgw_data, ALIGNED_RESIDUE_STRIDE=10, NEGATIVES_PER_POSITIVE=1):
            sampled = fgw_data.sample_residue_pairs(aligned, 100, 110, rng)

        positives = [p for p in sampled if p[0] == "aligned"]
        negatives = [p for p in sampled if p[0] != "aligned"]
        self.assertEqual(len(positives), 10)
        self.assertEqual(len(negatives), 10)
        self.assertEqual([p[0] for p in negatives][:4], ["shifted", "random", "shifted", "random"])

        on_path = {(i, j) for _, i, j, _, _, _ in aligned}
        for pair_type, align_pos, i, j, marker in negatives:
            self.assertNotIn((i, j), on_path)
            self.assertEqual(align_pos, -1)
            self.assertTrue(0 <= i < 100 and 0 <= j < 110)
        for (_, _, i, j, _), (kind, _, ni, nj, _) in zip(positives, negatives):
            if kind == "shifted":
                self.assertEqual(ni, i)
                self.assertTrue(fgw_data.SHIFT_RANGE[0] <= abs(nj - j) <= fgw_data.SHIFT_RANGE[1])

    def test_sampling_is_deterministic_per_source_row(self):
        aligned = [(k, k, k, "A", ":", "A") for k in range(64)]
        first = fgw_data.sample_residue_pairs(aligned, 64, 64, np.random.default_rng(5))
        second = fgw_data.sample_residue_pairs(aligned, 64, 64, np.random.default_rng(5))
        self.assertEqual(first, second)

    def test_process_tm_row_end_to_end(self):
        """Fake PDBs, a rotated copy, identity alignment: aligned pairs get a
        superposed distance of ~0 and a TM term of ~1; negatives are written."""
        try:
            import Bio  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("biopython not installed")
        import csv
        import tempfile

        num_residues = 40
        coords1 = compact_patch(num_residues, seed=9).astype(np.float64)
        rotation = proper_rotation(seed=9).astype(np.float64)
        translation = np.array([4.0, -7.0, 2.5])
        coords2 = coords1 @ rotation.T + translation

        with tempfile.TemporaryDirectory() as directory:
            pdb_dir = os.path.join(directory, "pdbs")
            emb_dir = os.path.join(directory, "emb")
            os.makedirs(pdb_dir)
            os.makedirs(emb_dir)
            write_fake_pdb(os.path.join(pdb_dir, "X1.pdb"), coords1)
            write_fake_pdb(os.path.join(pdb_dir, "X2.pdb"), coords2)
            rng = np.random.default_rng(1)
            for name in ("X1", "X2"):
                np.save(
                    os.path.join(emb_dir, f"{name}.npy"),
                    rng.normal(size=(num_residues, 8)).astype(np.float16),
                )

            row = {
                "row": 3, "id1": "X1", "id2": "X2",
                "tm_score_norm1": 1.0, "tm_score_norm2": 1.0,
                "seqxA": "A" * num_residues, "seqM": ":" * num_residues,
                "seqyA": "A" * num_residues,
                "superposition": common.format_superposition(translation, rotation),
            }
            out_path = os.path.join(directory, "out.csv")
            with open(out_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fgw_data.output_fields())
                writer.writeheader()
                with module_globals(fgw_data, PDB_DIR=pdb_dir, EMBEDDING_DIR=emb_dir):
                    written = fgw_data.process_tm_row(row, 3, writer)

            with open(out_path, newline="") as handle:
                records = list(csv.DictReader(handle))

        positives = [r for r in records if r["pair_type"] == "aligned"]
        negatives = [r for r in records if r["pair_type"] != "aligned"]
        self.assertEqual(written, len(records))
        self.assertEqual(len(positives), -(-num_residues // fgw_data.ALIGNED_RESIDUE_STRIDE))
        self.assertEqual(len(negatives), len(positives) * fgw_data.NEGATIVES_PER_POSITIVE)
        for record in positives:
            self.assertLess(float(record["superposed_dist"]), 1e-2)
            self.assertGreater(float(record["tm_term"]), 0.999)
            self.assertLess(float(record["gw_raw"]), 1e-3)
            self.assertEqual(record["residue_idx1"], record["residue_idx2"])
        for record in negatives:
            self.assertIn(record["pair_type"], ("shifted", "random"))
            self.assertNotEqual(record["residue_idx1"], record["residue_idx2"])
            self.assertEqual(record["align_pos"], "-1")
        self.assertEqual(set(records[0]) , set(fgw_data.output_fields()))


class TestClusterSplit(unittest.TestCase):
    def test_members_of_a_cluster_share_a_split(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "clusters.tsv")
            with open(path, "w") as handle:
                handle.write("REP1\tREP1\nREP1\tP00001\nREP1\tP00002\nREP2\tP00003\n")
            with module_globals(splits, CLUSTER_TSV=path, _CLUSTER_MAP=None):
                self.assertEqual(splits.split_key("P00001"), "REP1")
                self.assertEqual(splits.split_key("P99999"), "P99999")  # not in the map
                expected = splits.bucket_split(splits.stable_bucket("REP1"))
                self.assertEqual(splits.protein_split("P00001"), expected)
                self.assertEqual(splits.protein_split("P00002"), expected)
                self.assertIn("clusters", splits.split_summary())

    def test_missing_map_falls_back_to_ids(self):
        with module_globals(splits, CLUSTER_TSV="/nonexistent/clusters.tsv", _CLUSTER_MAP=None):
            self.assertEqual(splits.split_key("P00001"), "P00001")
            self.assertIn("NO cluster map", splits.split_summary())


@needs_torch
class TestModelComponents(unittest.TestCase):
    def test_radial_basis_is_unit_scale(self):
        basis = egnn_model.RadialBasis(num_rbf=8, max_distance=10.0)
        out = basis(torch.tensor([[0.0], [10.0], [400.0]]))
        self.assertEqual(tuple(out.shape), (3, 8))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))
        self.assertAlmostEqual(out[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(out[1, -1].item(), 1.0, places=5)

    def test_calibration_starts_at_half_cosine_plus_half(self):
        calibration = egnn_model.SimilarityCalibration()
        out = calibration(torch.tensor([1.0, 0.0, -1.0]))
        np.testing.assert_allclose(out.detach().numpy(), [1.0, 0.5, 0.0])
        self.assertEqual(len(list(calibration.parameters())), 2)

    def test_teacher_reports_calibrated_similarity(self):
        torch.manual_seed(0)
        model = egnn_model.SiameseEGNNTeacher(
            input_dim=6, hidden_dim=16, output_dim=8, num_layers=1, num_rbf=4
        ).eval()
        rng = np.random.default_rng(3)
        features = torch.from_numpy(rng.normal(size=(1, 2, 5, 6)).astype(np.float32))
        coords = torch.from_numpy(rng.normal(size=(1, 2, 5, 3)).astype(np.float32) * 4)
        mask = torch.ones(1, 2, 5, dtype=torch.bool)
        with torch.no_grad():
            out = model.forward_protein_pair(
                features, coords, features, coords, mask, mask, torch.ones(1, 2, dtype=torch.bool)
            )
        np.testing.assert_allclose(
            out["local_similarity"].numpy(), (0.5 * out["cosine_similarity"] + 0.5).numpy(), atol=1e-6
        )

    def test_loss_balancer_modes(self):
        import losses

        terms = {"fgw": torch.tensor(0.02), "tm": torch.tensor(0.5)}
        fixed = losses.LossBalancer({"fgw": 1.0, "tm": 0.2}, mode="fixed")
        self.assertAlmostEqual(fixed(terms).item(), 0.02 + 0.1, places=6)

        balanced = losses.LossBalancer({"fgw": 1.0, "tm": 0.2}, mode="uncertainty")
        self.assertAlmostEqual(balanced(terms).item(), 0.02 + 0.1, places=6)  # s = 0 at init
        with torch.no_grad():
            balanced.log_variances[0] = float(np.log(4.0))
        self.assertAlmostEqual(balanced.effective_weights()["fgw"], 0.25, places=6)
        expected = 1.0 * (0.02 / 4 + np.log(4.0)) + 0.2 * 0.5
        self.assertAlmostEqual(balanced(terms).item(), expected, places=5)

        # a missing or zero-weighted term is skipped, never an error
        self.assertAlmostEqual(fixed({"fgw": torch.tensor(0.02)}).item(), 0.02, places=6)
        zero = losses.LossBalancer({"fgw": 1.0, "tm": 0.0}, mode="fixed")
        self.assertAlmostEqual(zero(terms).item(), 0.02, places=6)
        with self.assertRaises(ValueError):
            fixed({})

    def test_similarity_statistics_ignore_padding(self):
        rng = np.random.default_rng(4)
        z = torch.nn.functional.normalize(
            torch.from_numpy(rng.normal(size=(1, 6, 8)).astype(np.float32)), dim=-1
        )
        full = torch.ones(1, 6, dtype=torch.bool)
        stats = student_module.similarity_statistics(torch.bmm(z, z.transpose(1, 2)), full, full)
        self.assertEqual(tuple(stats.shape), (1, 5))
        np.testing.assert_allclose(stats[0, :2].numpy(), [1.0, 1.0], atol=1e-5)  # self-match

        padded = torch.cat([z, torch.zeros(1, 3, 8)], dim=1)
        mask = torch.cat([full, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
        stats_padded = student_module.similarity_statistics(
            torch.bmm(padded, padded.transpose(1, 2)), mask, mask
        )
        np.testing.assert_allclose(stats_padded.numpy(), stats.numpy(), atol=1e-5)

    def make_student(self):
        torch.manual_seed(2)
        return student_module.SequenceStudent(
            input_dim=6, hidden_dim=16, output_dim=8, num_layers=1, num_heads=2,
            ff_dim=16, dropout=0.0, max_length=64, use_tm_head=True,
        ).eval()

    def student_batch(self, length1, length2, slots1, slots2, seed=5):
        rng = np.random.default_rng(seed)
        features1 = torch.zeros(1, slots1, 6)
        features2 = torch.zeros(1, slots2, 6)
        features1[:, :length1] = torch.from_numpy(rng.normal(size=(1, length1, 6)).astype(np.float32))
        features2[:, :length2] = torch.from_numpy(rng.normal(size=(1, length2, 6)).astype(np.float32))
        mask1 = torch.zeros(1, slots1, dtype=torch.bool)
        mask2 = torch.zeros(1, slots2, dtype=torch.bool)
        mask1[:, :length1] = True
        mask2[:, :length2] = True
        residue_idx = torch.tensor([[0, 3, 5]])
        return features1, mask1, residue_idx, features2, mask2, residue_idx

    def test_student_predicts_both_tm_normalisations_and_ignores_padding(self):
        student = self.make_student()
        with torch.no_grad():
            tight = student.forward_pair(*self.student_batch(9, 7, 9, 7), pair_mask=torch.ones(1, 3, dtype=torch.bool))
            loose = student.forward_pair(*self.student_batch(9, 7, 20, 15), pair_mask=torch.ones(1, 3, dtype=torch.bool))

        for key in ("tm_score_pred", "tm_score_pred2", "local_similarity", "similarity_matrix", "similarity_stats"):
            self.assertIn(key, tight)
        self.assertEqual(tuple(tight["similarity_matrix"].shape), (1, 9, 7))
        self.assertTrue(0.0 <= tight["tm_score_pred"].item() <= 1.0)
        self.assertTrue(0.0 <= tight["tm_score_pred2"].item() <= 1.0)
        self.assertNotAlmostEqual(tight["tm_score_pred"].item(), tight["tm_score_pred2"].item(), places=6)

        np.testing.assert_allclose(loose["tm_score_pred"].numpy(), tight["tm_score_pred"].numpy(), atol=1e-4)
        np.testing.assert_allclose(loose["tm_score_pred2"].numpy(), tight["tm_score_pred2"].numpy(), atol=1e-4)
        np.testing.assert_allclose(loose["local_similarity"].numpy(), tight["local_similarity"].numpy(), atol=1e-4)

    def test_student_length_changes_the_tm_prediction(self):
        """The global head must see length: the same residues padded is not the
        same protein made longer."""
        student = self.make_student()
        short = self.student_batch(9, 7, 9, 7)
        longer = self.student_batch(9, 7, 9, 7)
        with torch.no_grad():
            base = student.forward_pair(*short)["tm_score_pred"].item()
            # append real residues to protein 1
            features1 = torch.cat([longer[0], torch.randn(1, 12, 6)], dim=1)
            mask1 = torch.ones(1, 21, dtype=torch.bool)
            changed = student.forward_pair(features1, mask1, longer[2], *longer[3:])["tm_score_pred"].item()
        self.assertNotAlmostEqual(base, changed, places=4)


@needs_torch
class TestResumeBookkeeping(unittest.TestCase):
    def test_epoch_complete_checkpoint_starts_the_next_epoch(self):
        import tempfile

        import train_teacher as trainer

        model = egnn_model.SiameseEGNNTeacher(
            input_dim=4, hidden_dim=8, output_dim=4, num_layers=1, num_rbf=4, use_tm_head=True
        )
        balancer = trainer.build_balancer()
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(balancer.parameters()))

        with tempfile.TemporaryDirectory() as directory:
            with module_globals(trainer, CHECKPOINT_DIR=directory, RESUME=True, RESUME_PATH=None):
                trainer.save_checkpoint(
                    model, balancer, optimizer, epoch=3, loss=0.1,
                    name=trainer.LATEST_CHECKPOINT_NAME, buffer_idx=0,
                    best_val=0.5, epoch_complete=True,
                )
                state = trainer.load_resume_state(model, balancer, optimizer)
                self.assertEqual(state[:3], (4, 0, 0))
                self.assertEqual(state[4], 0)
                self.assertEqual(state[5], 0.5)

                trainer.save_checkpoint(
                    model, balancer, optimizer, epoch=3, loss=0.1,
                    name=trainer.LATEST_CHECKPOINT_NAME, buffer_idx=7,
                    epoch_loss=12.0, epoch_examples=100, best_val=0.5,
                )
                state = trainer.load_resume_state(model, balancer, optimizer)
                self.assertEqual(state[0], 3)
                self.assertEqual(state[1], 7 * trainer.GROUPS_PER_BUFFER)
                self.assertEqual(state[2], 7)
                self.assertEqual(state[3:5], (12.0, 100))


class TestParsePdb(unittest.TestCase):
    """The parser reads exactly one chain of the first model."""

    def setUp(self):
        try:
            import Bio  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("biopython not installed")
        import tempfile

        import parse_pdb

        self.parse_pdb = parse_pdb
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.a = compact_patch(6, seed=30)
        self.b = compact_patch(6, seed=31) + 50.0

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def test_single_chain_round_trips(self):
        write_fake_pdb(self.path("mono.pdb"), self.a)
        coords, sequence = self.parse_pdb.parse_pdb(self.path("mono.pdb"))
        self.assertEqual(sequence, "A" * len(self.a))
        self.assertEqual(coords.dtype, np.float32)
        np.testing.assert_allclose(coords, self.a, atol=1e-3)

    def test_reads_only_the_first_chain(self):
        """A homo-dimer model must come back as one copy, not two glued together."""
        write_multichain_pdb(self.path("dimer.pdb"), [("A", self.a, "ALA"), ("B", self.b, "GLY")])
        coords, sequence = self.parse_pdb.parse_pdb(self.path("dimer.pdb"))
        self.assertEqual(sequence, "A" * len(self.a))
        np.testing.assert_allclose(coords, self.a, atol=1e-3)
        self.assertEqual(
            list(self.parse_pdb.chain_lengths(self.path("dimer.pdb")).items()),
            [("A", len(self.a)), ("B", len(self.b))],
        )

    def test_chain_id_selects_a_chain(self):
        write_multichain_pdb(self.path("dimer.pdb"), [("A", self.a, "ALA"), ("B", self.b, "GLY")])
        coords, sequence = self.parse_pdb.parse_pdb(self.path("dimer.pdb"), chain_id="B")
        self.assertEqual(sequence, "G" * len(self.b))
        np.testing.assert_allclose(coords, self.b, atol=1e-3)
        with self.assertRaises(ValueError):
            self.parse_pdb.parse_pdb(self.path("dimer.pdb"), chain_id="Z")

    def test_skips_a_leading_chain_without_standard_residues(self):
        """Unknown residue names count as no chain, so the next chain is used."""
        write_multichain_pdb(self.path("odd.pdb"), [("A", self.a, "UNK"), ("B", self.b, "GLY")])
        coords, sequence = self.parse_pdb.parse_pdb(self.path("odd.pdb"))
        self.assertEqual(sequence, "G" * len(self.b))
        np.testing.assert_allclose(coords, self.b, atol=1e-3)
        self.assertEqual(self.parse_pdb.chain_lengths(self.path("odd.pdb"))["A"], 0)

    def test_first_model_only(self):
        write_multichain_pdb(self.path("multi.pdb"), [("A", self.a, "ALA")], models=3)
        coords, sequence = self.parse_pdb.parse_pdb(self.path("multi.pdb"))
        self.assertEqual(len(sequence), len(self.a))
        np.testing.assert_allclose(coords, self.a, atol=1e-3)

    def test_empty_file_gives_empty_result(self):
        with open(self.path("empty.pdb"), "w") as handle:
            handle.write("END\n")
        coords, sequence = self.parse_pdb.parse_pdb(self.path("empty.pdb"))
        self.assertEqual(sequence, "")
        self.assertEqual(tuple(coords.shape), (0, 3))
        self.assertEqual(len(self.parse_pdb.chain_lengths(self.path("empty.pdb"))), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
