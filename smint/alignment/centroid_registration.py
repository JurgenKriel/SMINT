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
``ransac+tps``   RANSAC affine, then TPS on the inliers. Recommended.
===============  =========================================================

Measuring quality honestly
--------------------------
Target Registration Error (TRE) is the mean Euclidean distance between
transformed source points and their matched targets. It is easy to compute it
in a way that is **circular**: correspondences here are established by nearest
neighbour, so a sufficiently flexible transform can drive TRE to ~0 on the
pairs it was fitted to, regardless of whether those correspondences are right.
An unregularised TPS with as many control points as pairs will always report
TRE ~ 0 on those pairs -- that is interpolation, not accuracy.

Every function here therefore splits matched pairs into a fit set and a
held-out validation set, and :func:`register_centroids` reports TRE on both.
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

VALID_METHODS = ("affine", "ransac", "tps", "ransac+tps")


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
    from skimage.measure import ransac as _ransac
    from skimage.transform import AffineTransform

    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    if source_xy.shape[0] < min_samples:
        logger.warning(
            "RANSAC needs at least %d pairs, got %d; falling back to "
            "least-squares affine.", min_samples, source_xy.shape[0],
        )
        return estimate_affine(source_xy, target_xy), np.ones(
            source_xy.shape[0], dtype=bool
        )

    model, inliers = _ransac(
        (source_xy, target_xy),
        AffineTransform,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
        rng=random_state,
    )

    if model is None:
        logger.warning("RANSAC failed to find a model; using least squares.")
        return estimate_affine(source_xy, target_xy), np.ones(
            source_xy.shape[0], dtype=bool
        )

    if inliers is None:
        inliers = np.ones(source_xy.shape[0], dtype=bool)

    logger.info(
        "RANSAC affine: %d/%d inliers (%.1f%%) at residual_threshold=%s",
        int(inliers.sum()), len(inliers), 100 * inliers.mean(), residual_threshold,
    )
    return np.asarray(model.params, dtype=float), np.asarray(inliers, dtype=bool)


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

def register_centroids(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    method: str = "ransac+tps",
    max_distance: float = 100.0,
    mutual: bool = True,
    validation_fraction: float = 0.25,
    residual_threshold: float = 10.0,
    max_trials: int = 1000,
    tps_regularization: float = 1.0,
    n_control: Optional[int] = 1000,
    random_state: Optional[int] = 0,
) -> dict:
    """
    Register two centroid sets acquired from the *same* physical section.

    Matches centroids by nearest neighbour, holds out a fraction of the pairs,
    fits the requested transform on the rest, and reports TRE on both sets.

    Parameters
    ----------
    source_xy, target_xy : numpy.ndarray
        ``(N, 2)`` / ``(M, 2)`` ``(x, y)`` centroids. The source is moved onto
        the target.
    method : {'affine', 'ransac', 'tps', 'ransac+tps'}, optional
        Registration model. See the module docstring.
    max_distance : float, optional
        Correspondence cutoff, in coordinate units.
    mutual : bool, optional
        Require reciprocal nearest neighbours.
    validation_fraction : float, optional
        Fraction of matched pairs held out from fitting. Set to ``0.0`` only
        if you accept that the reported TRE is then uninformative.
    residual_threshold, max_trials : optional
        RANSAC settings.
    tps_regularization, n_control : optional
        TPS settings.
    random_state : int, optional
        Seed for the split, RANSAC and control-point sampling.

    Returns
    -------
    dict
        ``transform`` (callable mapping ``(N,2)`` -> ``(N,2)``),
        ``affine`` (3x3 or None), ``tps`` (model or None),
        ``source_idx`` / ``target_idx`` / ``match_distances``,
        ``inliers`` (or None), ``n_matched``,
        ``tre_initial``, ``tre_fit``, ``tre_validation`` -- each a
        ``(mean, std)`` tuple, with ``tre_validation`` the one to trust.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

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


def register_centroid_files(
    source_file: str,
    target_file: str,
    source_cols: Sequence[str] = ("centroid_x", "centroid_y"),
    target_cols: Sequence[str] = ("x_centroid", "y_centroid"),
    output_path: Optional[str] = None,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
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
    result_df[x_col] = moved[:, 0]
    result_df[y_col] = moved[:, 1]

    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info("Saved transformed centroids to %s", output_path)

    return result_df, result
