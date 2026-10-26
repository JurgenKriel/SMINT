import numpy as np
import pytest

from smint.alignment import centroid_registration as cr


def synthetic_section(seed=0, spacing=17.0, size=1500.0):
    """
    Xenium-like target cells and post-stain 'nuclei' with known correspondence.

    The nuclei are a subset of the cells, moved by a small rotation, a shift
    and a smooth warp (~10 um in total, comparable to the cell spacing), with
    segmentation noise and some nuclei that have no cell.
    """
    rng = np.random.default_rng(seed)
    grid = np.mgrid[0:size:spacing, 0:size:spacing].reshape(2, -1).T
    target = grid + rng.normal(0, spacing * 0.25, grid.shape)

    keep = rng.random(len(target)) < 0.85
    truth = target[keep]
    theta = np.deg2rad(1.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = truth.mean(axis=0)
    source = (truth - centre) @ rot.T + centre + np.array([7.0, -5.0])
    source += np.column_stack([5 * np.sin(truth[:, 1] / 300), 5 * np.cos(truth[:, 0] / 350)])
    source += rng.normal(0, 1.5, source.shape)

    extra = rng.uniform(0, size, (int(0.15 * len(source)), 2))
    source_all = np.vstack([source, extra])
    truth_all = np.vstack([truth, np.full_like(extra, np.nan)])
    return source_all, target, truth_all


def true_error(result, source, truth):
    has_truth = np.isfinite(truth[:, 0])
    moved = result["transform"](source[has_truth])
    return np.median(np.linalg.norm(moved - truth[has_truth], axis=1))


def test_multiround_recovers_true_positions():
    source, target, truth = synthetic_section()
    start = np.median(np.linalg.norm(source - truth, axis=1)[np.isfinite(truth[:, 0])])
    result = cr.register_centroids(source, target, method="multiround")
    err = true_error(result, source, truth)
    assert start > 12         # misaligned by more than the ~12 um cell spacing
    assert err < 2.2          # ends at the noise floor (1.5 um per axis, ~1.8 radial)
    assert result["tre_validation"][0] < result["tre_initial"][0]
    assert [r["stage"] for r in result["rounds"]][:3] == ["initial", "affine", "tps"]
    after, before = result["validation"]["after"], result["validation"]["before"]
    assert after["mutual_within_5"] > 2 * before["mutual_within_5"]


def test_multiround_beats_single_round():
    source, target, truth = synthetic_section(seed=1)
    multi = true_error(cr.register_centroids(source, target, method="multiround"), source, truth)
    single = true_error(cr.register_centroids(source, target, method="ransac+tps"), source, truth)
    assert multi < 0.6 * single


def test_multiround_validation_uses_held_out_points():
    source, target, _ = synthetic_section(seed=2)
    result = cr.register_centroids(source, target, method="multiround", validation_fraction=0.3)
    fitted = set(result["source_idx"].tolist())
    rng = np.random.default_rng(0)
    held = np.flatnonzero(rng.random(len(source)) < 0.3)
    assert fitted.isdisjoint(held.tolist())


def test_multiround_rejects_rounds_that_do_not_help():
    source, target, truth = synthetic_section(seed=3)
    # A last round with blocks too small to vote reliably must not be kept
    schedule = ((100, 500, 2.0), (25, 250, 1.0), (10, 40, 1.0))
    result = cr.register_centroids(source, target, method="multiround", schedule=schedule)
    assert not result["rounds"][-1]["accepted"] or result["rounds"][-1]["blocks"] >= 3
    assert true_error(result, source, truth) < 2.2


def test_voting_finds_offset_larger_than_cell_spacing():
    rng = np.random.default_rng(6)
    # Irregular, tissue-like layout: a perfect lattice is periodic, so any
    # whole-spacing shift would match it equally well.
    target = np.mgrid[0:800:12, 0:800:12].reshape(2, -1).T + rng.normal(0, 3.5, (4489, 2))
    source = target + np.array([18.0, -11.0]) + rng.normal(0, 1.0, target.shape)
    centres, peaks = cr.vote_block_displacements(source, target, max_distance=50, block_size=400)
    assert np.allclose(np.median(peaks, axis=0), [-18.0, 11.0], atol=2.0)


def test_ransac_ignores_outliers_and_stays_small():
    rng = np.random.default_rng(4)
    src = rng.uniform(0, 1000, (40000, 2))
    true = np.array([[1.02, 0.05, 30.0], [-0.04, 0.98, -12.0], [0, 0, 1]])
    tgt = cr.apply_affine(src, true) + rng.normal(0, 0.5, src.shape)
    bad = rng.random(len(src)) < 0.4
    tgt[bad] = rng.uniform(0, 1000, (bad.sum(), 2))
    matrix, inliers = cr.estimate_affine_ransac(src, tgt, residual_threshold=50.0)
    assert np.allclose(matrix[:2, :2], true[:2, :2], atol=2e-3)
    assert np.allclose(matrix[:2, 2], true[:2, 2], atol=0.5)
    assert inliers[~bad].mean() > 0.99


def test_single_round_methods_still_work():
    source, target, truth = synthetic_section(seed=5)
    for method in ("affine", "ransac", "tps", "ransac+tps"):
        result = cr.register_centroids(source, target, method=method)
        assert np.isfinite(result["tre_validation"][0])
