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


def true_partner(target, truth):
    """Index of each source point's true target (-1 for points with none)."""
    has = np.isfinite(truth[:, 0])
    idx = np.full(len(truth), -1)
    d, j = cr.cKDTree(target).query(truth[has])
    idx[has] = np.where(d < 1e-9, j, -1)
    return idx


def test_refinement_snaps_onto_the_correct_cells():
    source, target, truth = synthetic_section(seed=7)
    reg = cr.register_centroids(source, target)
    ref = cr.refine_by_assignment(source, target, transform=reg["transform"])
    snapped = ref["status"] == "snapped"
    correct = ref["target_idx"][snapped] == true_partner(target, truth)[snapped]
    pairable = true_partner(target, truth) >= 0
    assert snapped[pairable].mean() > 0.95           # nearly every nucleus with a cell gets it
    assert correct.mean() > 0.97                     # and nearly all partners are right
    assert len(ref["snapped_per_pass"]) >= 2         # later passes add pairs
    assert np.allclose(ref["refined_xy"][snapped], target[ref["target_idx"][snapped]])
    assert ref["validation"]["median_error"] < 3.0   # neighbours predict held-out pairs well


def test_refinement_fills_the_rest_from_neighbours():
    source, target, truth = synthetic_section(seed=8)
    transform = cr.register_centroids(source, target)["transform"]
    ref = cr.refine_by_assignment(source, target, transform=transform)
    filled = (ref["status"] == "filled") & np.isfinite(truth[:, 0])
    err = np.linalg.norm(ref["refined_xy"][filled] - truth[filled], axis=1)
    smooth = np.linalg.norm(transform(source[filled]) - truth[filled], axis=1)
    # The unsnapped points are the hard ones; filling must not make them worse
    assert filled.any() and np.median(err) <= 1.05 * np.median(smooth) and np.median(err) < 4.0


def test_refinement_rejects_pairs_that_disagree_with_neighbours():
    rng = np.random.default_rng(9)
    target = np.mgrid[0:600:15, 0:600:15].reshape(2, -1).T + rng.normal(0, 4, (1600, 2))
    source = target + rng.normal(0, 1.0, target.shape)
    source[::50] += np.array([8.0, 0.0])             # every 50th point pushed off by 8 um
    ref = cr.refine_by_assignment(source, target, consistency_tol=4.0)
    assert (ref["status"][::50] != "snapped").mean() > 0.8


def test_register_centroid_files_writes_refined_and_smooth(tmp_path):
    source, target, _ = synthetic_section(seed=10)
    import pandas as pd
    pd.DataFrame(source, columns=["centroid_x", "centroid_y"]).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame(target, columns=["x_centroid", "y_centroid"]).to_csv(tmp_path / "t.csv", index=False)
    df, result = cr.register_centroid_files(str(tmp_path / "s.csv"), str(tmp_path / "t.csv"))
    assert {"x_transformed", "x_transformed_smooth", "registration_status", "target_index",
            "neighbour_deviation"} <= set(df.columns)
    assert result["refinement"]["n_snapped"] == int((df.registration_status == "snapped").sum())


def test_assigned_pairs_are_placed_and_flagged_by_their_deviation():
    source, target, truth = synthetic_section(seed=12)
    ref = cr.refine_by_assignment(source, target, transform=cr.register_centroids(source, target)["transform"])
    assigned, snapped = ref["status"] == "assigned", ref["status"] == "snapped"
    assert assigned.any()
    assert np.allclose(ref["refined_xy"][assigned], target[ref["target_idx"][assigned]])
    assert len(set(ref["target_idx"][ref["target_idx"] >= 0])) == np.sum(ref["target_idx"] >= 0)  # one-to-one
    # Synthetic leftovers mostly have no true cell; their deviation must show it
    assert ref["median_deviation"]["assigned"] > 2 * ref["median_deviation"]["snapped"]


def test_assign_remaining_can_be_disabled():
    source, target, _ = synthetic_section(seed=12)
    ref = cr.refine_by_assignment(source, target, transform=cr.register_centroids(source, target)["transform"],
                                  assign_remaining=False)
    assert not np.any(ref["status"] == "assigned")
