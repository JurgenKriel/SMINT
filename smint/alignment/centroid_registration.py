"""
Centroid-to-centroid registration for SMINT.

Two regimes, with different correct answers:

**Sequential sections** -- the two modalities come from different physical
sections, so the tissue genuinely differs and there is no true one-to-one
correspondence between cells. Use the STalign LDDMM path in
:mod:`smint.alignment.st_sm_registration`.

**Same section, post-staining** -- the second modality is acquired on the
*exact same* section, so centroids do correspond one-to-one and registration
becomes a correspondence problem rather than a warping problem. That is what
this module handles, in increasing order of flexibility:

===============  =========================================================
``affine``       Least-squares affine on all matched pairs.
``ransac``       RANSAC-robust affine; rejects bad correspondences.
``tps``          Thin-plate spline after an affine pre-alignment.
``ransac+tps``   RANSAC affine, then TPS on the inliers.
``multiround``   Coarse-to-fine displacement voting: an affine through
                 per-block displacement peaks, then a smooth TPS through
                 finer blocks. Recommended, and the default.
===============  =========================================================

The single-round methods match every centroid once, by nearest neighbour,
before anything is fitted. When the starting misalignment is comparable to
the cell spacing, many of those first matches pair a nucleus with the wrong
cell, and no fit to them can recover -- nearest-neighbour refinement then pulls
each nucleus toward whichever cell happens to be closest. ``multiround``
avoids committing to matches: within each block, every nucleus votes for the
displacements to several nearby cells, and the true offset -- shared by the
whole block -- is the peak of those votes (:func:`vote_block_displacements`).
Round 1 fits an affine through the block peaks; later rounds vote again in
smaller blocks on the result and add a smooth TPS through those peaks, to
follow local warps. A round is kept only if it improves the alignment of the
fitting points.

Measuring quality honestly
--------------------------
Target Registration Error (TRE) is the mean Euclidean distance between
transformed source points and their matched targets. It is easy to compute it
in a way that is **circular**: correspondences here are established by nearest
neighbour, so a sufficiently flexible transform can drive TRE to ~0 on the
pairs it was fitted to, regardless of whether those correspondences are right.
An unregularised TPS with as many control points as pairs will always report
TRE ~ 0 on those pairs -- that is interpolation, not accuracy.

Every method here therefore holds data out of fitting and reports TRE on it.
The single-round methods hold out matched pairs. ``multiround`` re-matches
every round, so it holds out *source points* instead, before any matching:
TRE is then the distance from each held-out point, after transformation, to
its mutual nearest target within ``max_distance``.
**Only the validation TRE is meaningful.** A large gap between fit TRE and
validation TRE means the transform is memorising correspondences.

Coordinate conventions
----------------------
Everything in this module is plain ``(x, y)``. The row-col ordering and the
transposed output frame described in :mod:`smint.alignment.st_sm_registration`
are artefacts of STalign specifically and do **not** apply here.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

VALID_METHODS = ("affine", "ransac", "tps", "ransac+tps", "multiround")


# --------------------------------------------------------------------------
# Correspondence
# --------------------------------------------------------------------------

def match_centroids(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    max_distance: float = 100.0,
    mutual: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Establish nearest-neighbour correspondences between two centroid sets.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` / ``(M, 2)`` arrays of ``(x, y)`` centroids.
    max_distance : float, optional
        Maximum distance for a pair to be considered corresponding, in the
        coordinate units (microns for Xenium).
    mutual : bool, optional
        If True (default), keep only pairs that are each other's nearest
        neighbour. Plain one-directional matching lets many source points
        collapse onto a single popular target, which biases the fit and
        inflates the apparent match count.

    Returns
    -------
    source_idx, target_idx : numpy.ndarray
        Indices of matched pairs.
    distances : numpy.ndarray
        Distance for each matched pair.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    if source_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=float)

    tree_t = cKDTree(target_xy)
    dist_st, idx_st = tree_t.query(source_xy, k=1, distance_upper_bound=max_distance)

    keep = np.isfinite(dist_st) & (dist_st <= max_distance)
    source_idx = np.flatnonzero(keep)
    target_idx = idx_st[keep]
    distances = dist_st[keep]

    if mutual and source_idx.size:
        tree_s = cKDTree(source_xy)
        _, back_idx = tree_s.query(target_xy[target_idx], k=1)
        reciprocal = back_idx == source_idx
        source_idx, target_idx, distances = (
            source_idx[reciprocal],
            target_idx[reciprocal],
            distances[reciprocal],
        )

    logger.info(
        "Matched %d pairs (max_distance=%s, mutual=%s) from %d source / %d target",
        len(source_idx), max_distance, mutual, len(source_xy), len(target_xy),
    )
    return source_idx, target_idx, distances


def split_pairs(
    n_pairs: int,
    validation_fraction: float = 0.25,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split matched pairs into fit and held-out validation index sets.

    Returns
    -------
    fit_idx, val_idx : numpy.ndarray
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}"
        )
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_pairs)
    n_val = int(round(n_pairs * validation_fraction))
    return perm[n_val:], perm[:n_val]


# --------------------------------------------------------------------------
# Error metric
# --------------------------------------------------------------------------

def target_registration_error(
    source_xy: np.ndarray, target_xy: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Target Registration Error between paired point sets.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` arrays of corresponding points, same length and order.

    Returns
    -------
    mean, std : float
        Mean and standard deviation of the per-pair distance.
    per_pair : numpy.ndarray
        The per-pair distances.

    Notes
    -----
    Interpret with care on points used for fitting -- see the module
    docstring. Prefer the held-out value reported by
    :func:`register_centroids`.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if source_xy.shape != target_xy.shape:
        raise ValueError(
            f"Point sets must have the same shape, got {source_xy.shape} "
            f"and {target_xy.shape}"
        )
    if source_xy.shape[0] == 0:
        return float("nan"), float("nan"), np.empty(0)

    per_pair = np.linalg.norm(source_xy - target_xy, axis=1)
    return float(per_pair.mean()), float(per_pair.std()), per_pair


# --------------------------------------------------------------------------
# Affine
# --------------------------------------------------------------------------

def apply_affine(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous affine matrix to ``(N, 2)`` points."""
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.shape[0] == 0:
        return points_xy.reshape(0, 2)
    homogeneous = np.hstack([points_xy, np.ones((points_xy.shape[0], 1))])
    return (np.asarray(matrix) @ homogeneous.T).T[:, :2]


def estimate_affine(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """
    Least-squares affine transform mapping source onto target.

    Returns
    -------
    numpy.ndarray
        3x3 homogeneous matrix; identity if there are too few pairs.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    if source_xy.shape[0] < 3:
        logger.warning(
            "Affine needs at least 3 pairs, got %d; returning identity.",
            source_xy.shape[0],
        )
        return np.eye(3)

    augmented = np.hstack([source_xy, np.ones((source_xy.shape[0], 1))])
    params, _, rank, _ = np.linalg.lstsq(augmented, target_xy, rcond=None)
    if rank < augmented.shape[1]:
        logger.warning(
            "Rank-deficient affine fit (rank %d < %d); the pairs may be "
            "collinear or degenerate.", rank, augmented.shape[1],
        )

    matrix = np.eye(3)
    matrix[0, :] = [params[0, 0], params[1, 0], params[2, 0]]
    matrix[1, :] = [params[0, 1], params[1, 1], params[2, 1]]
    return matrix


def estimate_affine_ransac(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    residual_threshold: float = 10.0,
    max_trials: int = 1000,
    min_samples: int = 3,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC-robust affine transform, rejecting outlier correspondences.

    Nearest-neighbour matching inevitably produces some wrong pairs -- an
    unmatched cell picks up whichever neighbour is closest. Least squares
    lets those wrong pairs drag the fit; RANSAC finds the transform supported
    by the largest consistent subset and reports the rest as outliers.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` corresponding points.
    residual_threshold : float, optional
        Max residual (in coordinate units) for a pair to count as an inlier.
    max_trials : int, optional
        RANSAC iterations.
    min_samples : int, optional
        Points per RANSAC sample; 3 is the minimum for an affine.
    random_state : int, optional
        Seed, for reproducibility.

    Returns
    -------
    matrix : numpy.ndarray
        3x3 homogeneous affine.
    inliers : numpy.ndarray
        Boolean mask of inlier pairs.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    n = source_xy.shape[0]
    if n < min_samples:
        logger.warning(
            "RANSAC needs at least %d pairs, got %d; falling back to "
            "least-squares affine.", min_samples, n,
        )
        return estimate_affine(source_xy, target_xy), np.ones(n, dtype=bool)

    # Plain NumPy: scikit-image's ransac peaked at ~75 GB on 35k pairs with a
    # wide residual threshold. Candidate affines come from exact 3-point
    # solves, scored in batches so memory stays at batch x n.
    rng = np.random.default_rng(random_state)
    homogeneous = np.hstack([source_xy, np.ones((n, 1))])
    threshold_sq = float(residual_threshold) ** 2
    best_count, best_params = -1, None
    batch = 64
    for start in range(0, max_trials, batch):
        size = min(batch, max_trials - start)
        picks = np.stack([rng.choice(n, min_samples, replace=False) for _ in range(size)])
        A = homogeneous[picks[:, :3]]                      # (size, 3, 3)
        B = target_xy[picks[:, :3]]                        # (size, 3, 2)
        ok = np.abs(np.linalg.det(A)) > 1e-9               # skip collinear triples
        if not ok.any():
            continue
        params = np.linalg.solve(A[ok], B[ok])             # (k, 3, 2)
        residual_sq = ((homogeneous @ params - target_xy) ** 2).sum(axis=2)  # (k, n)
        counts = (residual_sq < threshold_sq).sum(axis=1)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best_count, best_params = int(counts[k]), params[k]

    if best_params is None or best_count < min_samples:
        logger.warning("RANSAC failed to find a model; using least squares.")
        return estimate_affine(source_xy, target_xy), np.ones(n, dtype=bool)

    inliers = ((homogeneous @ best_params - target_xy) ** 2).sum(axis=1) < threshold_sq
    refit = np.linalg.lstsq(homogeneous[inliers], target_xy[inliers], rcond=None)[0]
    inliers = ((homogeneous @ refit - target_xy) ** 2).sum(axis=1) < threshold_sq
    matrix = np.eye(3)
    matrix[:2, :] = refit.T
    logger.info(
        "RANSAC affine: %d/%d inliers (%.1f%%) at residual_threshold=%s",
        int(inliers.sum()), n, 100 * inliers.mean(), residual_threshold,
    )
    return matrix, inliers


# --------------------------------------------------------------------------
# Thin-plate spline
# --------------------------------------------------------------------------

def _tps_radial_basis(r_squared: np.ndarray) -> np.ndarray:
    """TPS kernel U(r) = r^2 log(r^2), with the r=0 singularity zeroed."""
    result = np.zeros_like(r_squared)
    mask = r_squared > 1e-12
    result[mask] = r_squared[mask] * np.log(r_squared[mask])
    return result


def fit_tps(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    regularization: float = 1.0,
    n_control: Optional[int] = 1000,
    random_state: Optional[int] = 0,
) -> Optional[dict]:
    """
    Fit a thin-plate spline warping source control points onto targets.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` corresponding points.
    regularization : float, optional
        Ridge term added to the TPS system. ``0`` gives exact interpolation
        of every control point -- which drives fitted-point TRE to zero and
        tells you nothing (see module docstring). Larger values trade exact
        control-point fit for smoothness. Default 1.0.
    n_control : int, optional
        Number of control points to subsample. TPS solves an
        ``(n_control + 3)`` square system and evaluates an
        ``(N x n_control)`` kernel, so cost grows as the cube / product of
        this: 74,000 control points needs a ~44 GB solve. ``None`` uses every
        pair, which is rarely what you want.
    random_state : int, optional
        Seed for control-point subsampling.

    Returns
    -------
    dict or None
        Model with keys ``W``, ``A``, ``control_points``; None if the system
        could not be solved.
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    n_pairs, dims = source_xy.shape
    if n_pairs == 0:
        return None
    if n_pairs < dims + 1:
        logger.warning("TPS needs at least %d control points, got %d.", dims + 1, n_pairs)

    if n_control is not None and n_pairs > n_control:
        rng = np.random.default_rng(random_state)
        pick = rng.choice(n_pairs, n_control, replace=False)
        source_ctrl, target_ctrl = source_xy[pick], target_xy[pick]
        logger.info("TPS: subsampled %d control points from %d pairs", n_control, n_pairs)
    else:
        source_ctrl, target_ctrl = source_xy, target_xy

    n_ctrl = source_ctrl.shape[0]
    kernel = _tps_radial_basis(cdist(source_ctrl, source_ctrl, metric="sqeuclidean"))
    poly = np.hstack([np.ones((n_ctrl, 1)), source_ctrl])

    system = np.vstack([
        np.hstack([kernel, poly]),
        np.hstack([poly.T, np.zeros((dims + 1, dims + 1))]),
    ])
    if regularization:
        system[:n_ctrl, :n_ctrl] += np.eye(n_ctrl) * regularization

    rhs = np.vstack([target_ctrl, np.zeros((dims + 1, dims))])
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        try:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            logger.error("TPS system could not be solved.")
            return None

    return {
        "W": solution[:n_ctrl, :],
        "A": solution[n_ctrl:, :],
        "control_points": source_ctrl,
    }


def apply_tps(points_xy: np.ndarray, model: Optional[dict], chunk_size: int = 50000) -> np.ndarray:
    """
    Warp points with a fitted TPS model.

    Parameters
    ----------
    points_xy : numpy.ndarray
        ``(N, 2)`` points to warp.
    model : dict or None
        Output of :func:`fit_tps`. ``None`` returns the input unchanged.
    chunk_size : int, optional
        Points processed per block. The kernel is ``(chunk x n_control)``
        dense, so this bounds peak memory rather than materialising
        ``N x n_control`` at once.

    Returns
    -------
    numpy.ndarray
        Warped ``(N, 2)`` points.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.shape[0] == 0 or model is None:
        return points_xy

    W, A, control = model["W"], model["A"], model["control_points"]

    out = np.empty_like(points_xy)
    for start in range(0, points_xy.shape[0], chunk_size):
        block = points_xy[start:start + chunk_size]
        poly = np.hstack([np.ones((block.shape[0], 1)), block])
        kernel = _tps_radial_basis(cdist(block, control, metric="sqeuclidean"))
        out[start:start + chunk_size] = poly @ A + kernel @ W
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _mutual_tre(moved_xy: np.ndarray, target_xy: np.ndarray, max_distance: float):
    """``(mean, std)`` distance from points to their mutual nearest target within ``max_distance``."""
    _, _, distances = match_centroids(moved_xy, target_xy, max_distance=max_distance, mutual=True)
    if distances.size == 0:
        return (float("nan"), float("nan"))
    return (float(distances.mean()), float(distances.std()))


def alignment_quality(moved_xy: np.ndarray, target_xy: np.ndarray) -> dict:
    """
    Correspondence-free alignment summary of ``moved_xy`` against ``target_xy``.

    Returns ``median_nn`` (median distance to the nearest target) and
    ``mutual_within_5`` / ``mutual_within_10``: the fraction of points whose
    nearest target has them as its nearest point too, within 5 / 10 units.
    """
    moved_xy = np.asarray(moved_xy, dtype=float)
    if len(moved_xy) == 0:
        return {"median_nn": float("nan"), "mutual_within_5": float("nan"), "mutual_within_10": float("nan")}
    d, j = cKDTree(target_xy).query(moved_xy)
    back = cKDTree(moved_xy).query(np.asarray(target_xy)[j])[1]
    mutual = back == np.arange(len(moved_xy))
    return {
        "median_nn": float(np.median(d)),
        "mutual_within_5": float(np.mean(mutual & (d < 5))),
        "mutual_within_10": float(np.mean(mutual & (d < 10))),
    }


def vote_block_displacements(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    max_distance: float,
    block_size: float,
    bin_size: float = 2.0,
    k: Optional[int] = None,
    min_points: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the local source -> target displacement in square blocks.

    Every source point votes for the displacements to its ``k`` nearest
    targets within ``max_distance``. Wrong candidates point in random
    directions, but the true offset is shared by every point in the block, so
    it is the peak of the (lightly smoothed) 2D vote histogram. This works even
    when the misalignment exceeds the cell spacing, where nearest-neighbour
    matching cannot -- provided the tissue is not a regular lattice, whose
    periodicity makes every whole-spacing shift look equally good.

    ``k`` defaults to the expected number of targets within ``max_distance``
    (from the target density, capped at 128), so the true partner is among
    the candidates anywhere in that radius.

    Returns
    -------
    centres, displacements : numpy.ndarray
        ``(B, 2)`` block centroids and their modal displacements, for blocks
        holding at least ``min_points`` source points.
    """
    from scipy.ndimage import gaussian_filter

    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    if k is None:
        span = np.ptp(target_xy, axis=0)
        density = len(target_xy) / max(span[0] * span[1], 1e-9)
        k = int(np.clip(np.ceil(density * np.pi * max_distance ** 2), 8, 128))
    d, j = cKDTree(target_xy).query(source_xy, k=k, distance_upper_bound=max_distance)
    valid = np.isfinite(d)
    displacement = target_xy[np.where(valid, j, 0)] - source_xy[:, None, :]
    edges = np.arange(-max_distance, max_distance + bin_size, bin_size)

    keys = np.floor((source_xy - source_xy.min(axis=0)) / block_size).astype(np.int64)
    flat = keys[:, 0] * 1_000_003 + keys[:, 1]
    order = np.argsort(flat, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(flat[order])) + 1)

    centres, peaks = [], []
    for idx in groups:
        if len(idx) < min_points:
            continue
        votes = displacement[idx][valid[idx]]
        if len(votes) == 0:
            continue
        hist = gaussian_filter(np.histogram2d(votes[:, 0], votes[:, 1], bins=[edges, edges])[0], 1.0)
        ix, iy = np.unravel_index(np.argmax(hist), hist.shape)
        peaks.append([(edges[ix] + edges[ix + 1]) / 2, (edges[iy] + edges[iy + 1]) / 2])
        centres.append(source_xy[idx].mean(axis=0))
    return np.asarray(centres).reshape(-1, 2), np.asarray(peaks).reshape(-1, 2)


def _register_multiround(
    source_xy, target_xy, max_distance, validation_fraction, max_trials,
    tps_regularization, random_state, schedule, min_improvement,
) -> dict:
    """Coarse-to-fine displacement voting; see :func:`register_centroids`."""
    rng = np.random.default_rng(random_state)
    held_out = rng.random(len(source_xy)) < validation_fraction
    fit_points = np.flatnonzero(~held_out)
    fit_xy, val_xy = source_xy[~held_out], source_xy[held_out]
    if schedule is None:
        schedule = ((max_distance, 500.0, 2.0), (0.25 * max_distance, 250.0, 1.0))

    affine_matrix, tps_models = np.eye(3), []

    def transform_with(matrix, models):
        def transform(points):
            moved = apply_affine(points, matrix)
            for model in models:
                moved = apply_tps(moved, model)
            return moved
        return transform

    tre_initial = _mutual_tre(val_xy if held_out.any() else fit_xy, target_xy, max_distance)
    logger.info("Initial TRE (held-out points, mutual NN <= %s): %.4f", max_distance, tre_initial[0])
    score = alignment_quality(fit_xy, target_xy)["median_nn"]
    rounds = [{"stage": "initial", "median_nn": score}]

    for k, (round_distance, block_size, bin_size) in enumerate(schedule):
        current = transform_with(affine_matrix, tps_models)(fit_xy)
        centres, peaks = vote_block_displacements(current, target_xy, round_distance, block_size, bin_size)
        if len(centres) < 3:
            logger.info("Round %d: only %d blocks with enough points; stopping.", k + 1, len(centres))
            break
        step, inliers = estimate_affine_ransac(
            centres, centres + peaks, residual_threshold=3 * bin_size,
            max_trials=max_trials, random_state=random_state,
        )
        if k == 0:
            candidate = (step @ affine_matrix, tps_models)
            stage = "affine"
        else:
            model = fit_tps(centres[inliers], centres[inliers] + peaks[inliers],
                            regularization=tps_regularization, n_control=None)
            candidate = (affine_matrix, tps_models + [model])
            stage = "tps"
        new_score = alignment_quality(transform_with(*candidate)(fit_xy), target_xy)["median_nn"]
        accepted = new_score < score * (1 - min_improvement)
        rounds.append({"stage": stage, "max_distance": float(round_distance),
                       "block_size": float(block_size), "blocks": int(len(centres)),
                       "consistent_blocks": float(inliers.mean()), "median_nn": float(new_score),
                       "accepted": bool(accepted)})
        logger.info("Round %d (%s, %g-unit blocks): %d blocks, %.0f%% consistent, median NN %.2f -> %.2f%s",
                    k + 1, stage, block_size, len(centres), 100 * inliers.mean(), score, new_score,
                    "" if accepted else " -- not an improvement, stopping")
        if not accepted:
            break
        affine_matrix, tps_models = candidate
        score = new_score

    transform = transform_with(affine_matrix, tps_models)
    si, ti, match_distances = match_centroids(transform(fit_xy), target_xy,
                                              max_distance=max_distance, mutual=True)
    tre_fit = _mutual_tre(transform(fit_xy), target_xy, max_distance)
    tre_validation = (_mutual_tre(transform(val_xy), target_xy, max_distance)
                      if held_out.any() else (float("nan"), float("nan")))
    return {
        "affine": affine_matrix, "tps": tps_models[-1] if tps_models else None,
        "tps_models": tps_models, "transform": transform,
        "source_idx": fit_points[si], "target_idx": ti, "match_distances": match_distances,
        "inliers": None, "n_matched": int(len(si)), "rounds": rounds,
        "tre_initial": tre_initial, "tre_fit": tre_fit, "tre_validation": tre_validation,
        "validation": {
            "before": alignment_quality(val_xy, target_xy),
            "after": alignment_quality(transform(val_xy), target_xy),
        },
    }


def register_centroids(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    method: str = "multiround",
    max_distance: float = 100.0,
    mutual: bool = True,
    validation_fraction: float = 0.25,
    residual_threshold: float = 10.0,
    max_trials: int = 1000,
    tps_regularization: Optional[float] = None,
    n_control: Optional[int] = 1000,
    random_state: Optional[int] = 0,
    schedule: Optional[Sequence[Tuple[float, float, float]]] = None,
    min_improvement: float = 0.01,
) -> dict:
    """
    Register two centroid sets acquired from the *same* physical section.

    ``multiround`` (the default) re-matches and refits over several rounds;
    the single-round methods match once by nearest neighbour, hold out a
    fraction of the pairs, and fit the rest. All report TRE on held-out data.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` / ``(M, 2)`` ``(x, y)`` centroids. The source is moved onto
        the target.
    method : {'multiround', 'affine', 'ransac', 'tps', 'ransac+tps'}, optional
        Registration model. See the module docstring.
    max_distance : float, optional
        Correspondence cutoff, in coordinate units.
    mutual : bool, optional
        Require reciprocal nearest neighbours.
    validation_fraction : float, optional
        Fraction held out from fitting: matched pairs for the single-round
        methods, source points for ``multiround``. Set to ``0.0`` only if you
        accept that the reported TRE is then uninformative.
    residual_threshold, max_trials : optional
        RANSAC settings (``multiround`` derives each round's threshold from
        its radius).
    tps_regularization, n_control : optional
        TPS settings. Regularisation defaults to 1000 for ``multiround``
        (whose control points are block centres) and 1 for the single-round
        methods.
    random_state : int, optional
        Seed for the split, RANSAC and control-point sampling.
    schedule : sequence of (max_distance, block_size, bin_size), optional
        ``multiround`` voting rounds, coarse to fine. Round 1 fits an affine
        through the block displacements; later rounds add a smooth TPS
        through them. Default ``((max_distance, 500, 2),
        (0.25 * max_distance, 250, 1))``, in coordinate units (microns).
    min_improvement : float, optional
        ``multiround`` keeps a round only if it lowers the fit points' median
        nearest-target distance by at least this fraction, and stops at the
        first round that does not.

    Returns
    -------
    dict
        ``transform`` (callable mapping ``(N,2)`` -> ``(N,2)``),
        ``affine`` (3x3 or None), ``tps`` (model or None),
        ``source_idx`` / ``target_idx`` / ``match_distances``,
        ``inliers`` (or None), ``n_matched``,
        ``tre_initial``, ``tre_fit``, ``tre_validation`` -- each a
        ``(mean, std)`` tuple, with ``tre_validation`` the one to trust.
        ``multiround`` adds ``rounds`` (a per-round log) and ``validation``,
        :func:`alignment_quality` of the held-out points ``before`` and
        ``after`` -- a correspondence-free check alongside TRE.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    if method == "multiround":
        result = _register_multiround(
            source_xy, target_xy, max_distance, validation_fraction, max_trials,
            1000.0 if tps_regularization is None else tps_regularization,
            random_state, schedule, min_improvement,
        )
        tre_initial, tre_fit, tre_validation = (
            result["tre_initial"], result["tre_fit"], result["tre_validation"])
        logger.info("TRE after multiround -- fit: %.4f | held-out validation: %.4f",
                    tre_fit[0], tre_validation[0])
        if np.isfinite(tre_validation[0]) and tre_validation[0] >= tre_initial[0]:
            logger.warning(
                "Held-out TRE (%.4f) is no better than the initial TRE (%.4f): the "
                "transform does not improve the alignment.", tre_validation[0], tre_initial[0])
        return {"method": method, **result}

    if tps_regularization is None:
        tps_regularization = 1.0

    source_idx, target_idx, match_distances = match_centroids(
        source_xy, target_xy, max_distance=max_distance, mutual=mutual
    )
    n_matched = len(source_idx)
    if n_matched == 0:
        raise ValueError(
            "No centroid pairs matched. Increase max_distance, or check that "
            "the two sets share a coordinate frame."
        )

    paired_source = source_xy[source_idx]
    paired_target = target_xy[target_idx]

    fit_idx, val_idx = split_pairs(n_matched, validation_fraction, random_state)
    fit_source, fit_target = paired_source[fit_idx], paired_target[fit_idx]
    val_source, val_target = paired_source[val_idx], paired_target[val_idx]

    tre_initial = target_registration_error(paired_source, paired_target)[:2]
    logger.info("Initial TRE over %d matched pairs: %.4f", n_matched, tre_initial[0])

    affine_matrix = None
    inliers = None
    tps_model = None

    if method in ("ransac", "ransac+tps"):
        affine_matrix, inliers = estimate_affine_ransac(
            fit_source, fit_target,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state,
        )
    else:
        affine_matrix = estimate_affine(fit_source, fit_target)

    if method in ("tps", "ransac+tps"):
        # TPS refines what the affine leaves behind, so fit it on residuals
        # in the affine-corrected frame, using inliers only when we have them.
        tps_fit_source = fit_source if inliers is None else fit_source[inliers]
        tps_fit_target = fit_target if inliers is None else fit_target[inliers]
        tps_model = fit_tps(
            apply_affine(tps_fit_source, affine_matrix),
            tps_fit_target,
            regularization=tps_regularization,
            n_control=n_control,
            random_state=random_state,
        )

    def transform(points: np.ndarray) -> np.ndarray:
        moved = apply_affine(points, affine_matrix)
        if tps_model is not None:
            moved = apply_tps(moved, tps_model)
        return moved

    tre_fit = target_registration_error(transform(fit_source), fit_target)[:2]
    tre_validation = (
        target_registration_error(transform(val_source), val_target)[:2]
        if len(val_idx) else (float("nan"), float("nan"))
    )

    logger.info(
        "TRE after %s -- fit: %.4f | held-out validation: %.4f",
        method, tre_fit[0], tre_validation[0],
    )
    if len(val_idx) and np.isfinite(tre_validation[0]) and tre_validation[0] >= tre_initial[0]:
        logger.warning(
            "Held-out TRE (%.4f) is no better than the initial TRE (%.4f): the "
            "transform does not improve the alignment. The data may already be "
            "as aligned as a global transform allows, with the residual being "
            "correspondence ambiguity; compare it with the target's cell spacing.",
            tre_validation[0], tre_initial[0],
        )
    if len(val_idx) and np.isfinite(tre_validation[0]) and tre_fit[0] > 0:
        if tre_validation[0] > 5 * max(tre_fit[0], 1e-9):
            logger.warning(
                "Validation TRE (%.4f) far exceeds fit TRE (%.4f): the "
                "transform is memorising correspondences rather than "
                "generalising. Raise tps_regularization or lower n_control.",
                tre_validation[0], tre_fit[0],
            )

    return {
        "method": method,
        "transform": transform,
        "affine": affine_matrix,
        "tps": tps_model,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "match_distances": match_distances,
        "inliers": inliers,
        "n_matched": n_matched,
        "tre_initial": tre_initial,
        "tre_fit": tre_fit,
        "tre_validation": tre_validation,
    }


def refine_by_assignment(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    transform=None,
    snap_radius: float = 10.0,
    neighbour_radius: float = 100.0,
    consistency_tol: Optional[float] = None,
    min_neighbours: int = 5,
    k_fill: int = 8,
    validation_fraction: float = 0.1,
    random_state: Optional[int] = 0,
    passes: int = 4,
) -> dict:
    """
    Snap registered source points exactly onto their partner targets.

    A port of the lab's ``alignment_refinement.r``. A smooth transform leaves
    each nucleus a few microns from its cell -- partly because a nucleus
    really is off-centre in its cell. This step instead *assigns* nuclei to
    cells and moves each confidently assigned nucleus onto its partner:

    1. Pair transformed source points one-to-one with targets (mutual nearest
       neighbours within ``snap_radius``).
    2. Keep a pair only if its offset agrees, within ``consistency_tol``, with
       the median offset of the other pairs within ``neighbour_radius`` -- a
       pairing with the wrong cell disagrees with its neighbourhood.
    3. Snap kept pairs: the source point moves exactly onto its target.
    4. Every other point moves by the median offset of its ``k_fill`` nearest
       snapped neighbours (within ``neighbour_radius``), or not at all.
    5. Repeat: the moved, still-unsnapped points are paired again with the
       still-unmatched targets, and kept if they agree with the snapped field
       around them -- up to ``passes`` times, until nothing new snaps.

    Snapped points coincide with their targets by construction, so their
    residual says nothing about correctness. The check is ``validation``: a
    fraction of the kept pairs is left out of the offset field, their
    positions are predicted from their neighbours alone (step 4), and the
    distance from each prediction to its partner is reported.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` / ``(M, 2)`` points. ``source_xy`` in its original frame.
    transform : callable, optional
        Registration to apply first (e.g. ``register_centroids(...)["transform"]``).
        Default: none, ``source_xy`` is already registered.
    snap_radius : float
        Maximum distance for a pair.
    neighbour_radius : float
        Neighbourhood for the consistency test and for filling in.
    consistency_tol : float, optional
        Allowed deviation from the neighbourhood's median offset. Default
        0.4 x the target's median nearest-neighbour spacing.
    min_neighbours : int
        Pairs with fewer neighbouring pairs than this cannot be checked and
        are not snapped.
    passes : int
        Pairing passes (step 5); 1 disables re-pairing.
    k_fill : int
        Snapped neighbours whose median offset moves an unsnapped point.
    validation_fraction : float
        Fraction of kept pairs used for the held-out check.

    Returns
    -------
    dict
        ``refined_xy`` (``(N, 2)``), ``status`` (``'snapped'``, ``'filled'`` or
        ``'unmoved'`` per point), ``target_idx`` (partner index, -1 if none),
        ``n_snapped``, ``fraction_snapped`` (of source points),
        ``fraction_targets_matched``, ``consistency_tol`` and ``validation``
        (``median_error``, ``within_2``, ``within_5`` for the held-out pairs).
    """
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    moved = transform(source_xy) if transform is not None else source_xy.copy()
    n = len(moved)

    if consistency_tol is None:
        spacing = cKDTree(target_xy).query(target_xy, k=2)[0][:, 1]
        consistency_tol = 0.4 * float(np.median(spacing))

    si, ti, _ = match_centroids(moved, target_xy, max_distance=snap_radius, mutual=True)
    offsets = target_xy[ti] - moved[si]

    # Two passes: the second judges each pair against neighbours that passed the first
    keep = np.ones(len(si), dtype=bool)
    for _ in range(2):
        med = np.full((len(si), 2), np.nan)
        kept_idx = np.flatnonzero(keep)
        tree = cKDTree(moved[si][kept_idx])
        for i, idx in enumerate(tree.query_ball_point(moved[si], neighbour_radius)):
            idx = [kept_idx[j] for j in idx if kept_idx[j] != i]
            if len(idx) >= min_neighbours:
                med[i] = np.median(offsets[idx], axis=0)
        deviation = np.linalg.norm(offsets - med, axis=1)
        keep = np.isfinite(deviation) & (deviation <= consistency_tol)

    kept_src, kept_tgt, kept_off = si[keep], ti[keep], offsets[keep]

    # Held-out check: predict some kept pairs from the others' offsets alone
    rng = np.random.default_rng(random_state)
    held = rng.random(len(kept_src)) < validation_fraction
    validation = {"n": int(held.sum()), "median_error": float("nan"),
                  "within_2": float("nan"), "within_5": float("nan")}
    if held.any() and (~held).sum() >= k_fill:
        tree = cKDTree(moved[kept_src[~held]])
        d, j = tree.query(moved[kept_src[held]], k=k_fill, distance_upper_bound=neighbour_radius)
        ok = np.isfinite(d)
        pred_off = np.array([np.median(kept_off[~held][jj[o]], axis=0) if o.any() else [0.0, 0.0]
                             for jj, o in zip(j, ok)])
        err = np.linalg.norm(moved[kept_src[held]] + pred_off - target_xy[kept_tgt[held]], axis=1)
        validation.update(median_error=float(np.median(err)), within_2=float(np.mean(err < 2)),
                          within_5=float(np.mean(err < 5)))

    refined = moved.copy()
    status = np.full(n, "unmoved", dtype=object)
    partner = np.full(n, -1)
    refined[kept_src] = target_xy[kept_tgt]
    status[kept_src] = "snapped"
    partner[kept_src] = kept_tgt
    passes_snapped = [int(len(kept_src))]

    def field_median(points, min_count):
        """Median snapped offset of each point's k_fill nearest snapped neighbours."""
        snapped = np.flatnonzero(status == "snapped")
        out = np.full((len(points), 2), np.nan)
        if len(snapped) == 0:
            return out
        field = target_xy[partner[snapped]] - moved[snapped]
        k = min(k_fill, len(snapped))
        d, j = cKDTree(moved[snapped]).query(points, k=k, distance_upper_bound=neighbour_radius)
        if k == 1:
            d, j = d[:, None], j[:, None]
        for i, (dd, jj) in enumerate(zip(d, j)):
            ok = np.isfinite(dd)
            if ok.sum() >= min_count:
                out[i] = np.median(field[jj[ok]], axis=0)
        return out

    def fill():
        rest = np.flatnonzero(status != "snapped")
        if len(rest):
            med = field_median(moved[rest], 1)
            ok = np.isfinite(med[:, 0])
            refined[rest[ok]] = moved[rest[ok]] + med[ok]
            status[rest[ok]] = "filled"
            refined[rest[~ok]] = moved[rest[~ok]]
            status[rest[~ok]] = "unmoved"

    fill()
    for _ in range(passes - 1):
        free_src = np.flatnonzero(status != "snapped")
        taken = np.zeros(len(target_xy), dtype=bool)
        taken[partner[partner >= 0]] = True
        free_tgt = np.flatnonzero(~taken)
        if len(free_src) == 0 or len(free_tgt) == 0:
            break
        si2, ti2, _ = match_centroids(refined[free_src], target_xy[free_tgt],
                                      max_distance=snap_radius, mutual=True)
        src, tgt = free_src[si2], free_tgt[ti2]
        med = field_median(moved[src], min_neighbours)
        deviation = np.linalg.norm((target_xy[tgt] - moved[src]) - med, axis=1)
        accept = np.isfinite(deviation) & (deviation <= consistency_tol)
        if not accept.any():
            break
        refined[src[accept]] = target_xy[tgt[accept]]
        status[src[accept]] = "snapped"
        partner[src[accept]] = tgt[accept]
        passes_snapped.append(int(np.sum(status == "snapped")))
        fill()

    result = {
        "refined_xy": refined, "status": status, "target_idx": partner,
        "n_snapped": int(np.sum(status == "snapped")),
        "fraction_snapped": float(np.sum(status == "snapped") / max(n, 1)),
        "fraction_targets_matched": float(np.sum(status == "snapped") / max(len(target_xy), 1)),
        "snapped_per_pass": passes_snapped,
        "n_filled": int(np.sum(status == "filled")),
        "consistency_tol": float(consistency_tol),
        "validation": validation,
    }
    logger.info(
        "Refinement: %d points snapped (%.1f%% of source, %.1f%% of targets), %d filled from "
        "neighbours; held-out field check: median %.2f, %.0f%% within 2, %.0f%% within 5",
        result["n_snapped"], 100 * result["fraction_snapped"], 100 * result["fraction_targets_matched"],
        result["n_filled"], validation["median_error"], 100 * validation["within_2"],
        100 * validation["within_5"],
    )
    return result


def register_centroid_files(
    source_file: str,
    target_file: str,
    source_cols: Sequence[str] = ("centroid_x", "centroid_y"),
    target_cols: Sequence[str] = ("x_centroid", "y_centroid"),
    output_path: Optional[str] = None,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    refine: bool = True,
    refine_kwargs: Optional[dict] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, dict]:
    """
    Register centroids from two CSV files and write the transformed source.

    Every source row is transformed, not just the matched ones -- matching
    selects the pairs used to *fit* the transform, which then applies
    everywhere.

    Parameters
    ----------
    source_file, target_file : str
        CSVs holding the two centroid sets.
    source_cols, target_cols : sequence of str, optional
        ``(x, y)`` column names in each file. Defaults follow the project's
        convention: segmented centroids use ``centroid_x``/``centroid_y``,
        Xenium uses ``x_centroid``/``y_centroid``.
    output_path : str, optional
        Where to write the transformed source table.
    x_col, y_col : str, optional
        Output column names.
    refine : bool, optional
        After registering, snap confidently matched points onto their targets
        with :func:`refine_by_assignment` (default). ``x_col`` / ``y_col`` then
        hold the refined positions, ``x_col + "_smooth"`` the transform-only
        ones, and ``registration_status`` says how each point was placed.
    refine_kwargs : dict, optional
        Forwarded to :func:`refine_by_assignment`.
    **kwargs
        Forwarded to :func:`register_centroids`.

    Returns
    -------
    result_df : pandas.DataFrame
        Source table with transformed coordinates appended.
    result : dict
        The :func:`register_centroids` result.
    """
    source_df = pd.read_csv(source_file)
    target_df = pd.read_csv(target_file, usecols=list(target_cols))

    for cols, df, name in (
        (source_cols, source_df, source_file),
        (target_cols, target_df, target_file),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} not found in {name}. "
                f"Available: {', '.join(map(str, df.columns[:20]))}"
            )

    source_xy = source_df[list(source_cols)].to_numpy(dtype=float)
    target_xy = target_df[list(target_cols)].to_numpy(dtype=float)

    result = register_centroids(source_xy, target_xy, **kwargs)

    moved = result["transform"](source_xy)
    result_df = source_df.copy()
    if refine:
        refinement = refine_by_assignment(moved, target_xy, **(refine_kwargs or {}))
        result["refinement"] = refinement
        result_df[f"{x_col}_smooth"] = moved[:, 0]
        result_df[f"{y_col}_smooth"] = moved[:, 1]
        moved = refinement["refined_xy"]
        result_df["registration_status"] = refinement["status"]
        result_df["target_index"] = refinement["target_idx"]
    result_df[x_col] = moved[:, 0]
    result_df[y_col] = moved[:, 1]

    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info("Saved transformed centroids to %s", output_path)

    return result_df, result
