"""
Coarse pre-registration transforms: scale, rotation and flips.

STalign's LDDMM converges from a landmark-initialised affine, but it still
expects the two datasets to start in roughly the same coordinate system and
orientation. Spatial metabolomics is acquired on its own pixel grid, often
rotated or mirrored relative to the Xenium section and at a completely
different scale, so a coarse pre-registration first makes the difference
between converging and wandering off.

This module builds that coarse transform as a single 3x3 affine, so it is
composable, inspectable and recordable -- the same matrix can be saved
alongside the outputs and replayed later, rather than living as a pile of
ad-hoc ``scale_xy=10.0, rotate_left_90=True`` flags scattered through a
notebook.

Order of operations
-------------------
Operations are applied in a fixed, documented order, about the source
centroid unless a centre is given::

    flip  ->  rotate  ->  scale  ->  translate

Fixing the order matters: rotation and flips do not commute, so "flip x then
rotate 90" is not "rotate 90 then flip x". Composing your own matrices with
:func:`compose` lets you depart from this order deliberately.
"""

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VALID_SCALE_MODES = ("extent", "max", "none")


def _as_xy(coords) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) array of (x, y) coordinates, got {arr.shape}")
    return arr


def compose(*matrices: np.ndarray) -> np.ndarray:
    """
    Compose 3x3 affine matrices, applied left to right.

    ``compose(A, B)`` returns a matrix equivalent to applying ``A`` then ``B``.
    Note this is ``B @ A`` in matrix terms -- the argument order follows the
    order operations happen, which is the one that matches how people describe
    them.
    """
    result = np.eye(3)
    for matrix in matrices:
        result = np.asarray(matrix, dtype=float) @ result
    return result


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """3x3 affine for a translation."""
    matrix = np.eye(3)
    matrix[0, 2] = dx
    matrix[1, 2] = dy
    return matrix


def scale_matrix(sx: float, sy: Optional[float] = None, center: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """3x3 affine scaling about ``center``; ``sy`` defaults to ``sx``."""
    sy = sx if sy is None else sy
    cx, cy = center
    matrix = np.eye(3)
    matrix[0, 0], matrix[1, 1] = sx, sy
    matrix[0, 2] = cx - sx * cx
    matrix[1, 2] = cy - sy * cy
    return matrix


def rotation_matrix(degrees: float, center: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """
    3x3 affine rotating counter-clockwise by ``degrees`` about ``center``.

    Counter-clockwise in standard xy orientation. Note that images displayed
    with an inverted y axis will appear to turn the other way.
    """
    theta = np.deg2rad(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = center
    matrix = np.eye(3)
    matrix[0, 0], matrix[0, 1] = cos_t, -sin_t
    matrix[1, 0], matrix[1, 1] = sin_t, cos_t
    matrix[0, 2] = cx - cos_t * cx + sin_t * cy
    matrix[1, 2] = cy - sin_t * cx - cos_t * cy
    return matrix


def flip_matrix(flip_x: bool = False, flip_y: bool = False, center: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """
    3x3 affine mirroring about ``center``.

    ``flip_x`` mirrors left-right (negates x), ``flip_y`` mirrors top-bottom.
    A flip is a scale by -1, so it is handled by the same machinery.
    """
    return scale_matrix(-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0, center=center)


def fit_scale_to_reference(
    source_xy, reference_xy, mode: str = "extent", preserve_aspect: bool = True
) -> Tuple[float, float]:
    """
    Scale factors bringing a source dataset onto a reference's coordinate system.

    Parameters
    ----------
    source_xy, reference_xy : array-like
        ``(N, 2)`` / ``(M, 2)`` coordinate arrays.
    mode : {'extent', 'max', 'none'}, optional
        ``'extent'`` (default) matches the bounding-box **span** of each axis,
        which is robust to the two datasets having different origins.
        ``'max'`` matches the maximum coordinate directly -- literal, but wrong
        whenever either dataset does not start near zero.
        ``'none'`` returns ``(1.0, 1.0)``.
    preserve_aspect : bool, optional
        If True (default), use a single isotropic factor -- the smaller of the
        two -- so the tissue is not distorted. Anisotropic scaling can force
        bounding boxes to agree while making the shapes match *worse*, which
        then misleads the landmark step.

    Returns
    -------
    (sx, sy)

    Notes
    -----
    Bounding boxes are sensitive to outliers: a handful of stray points
    inflates the span and shrinks the fitted factor. Check the result on a
    plot, and drop obvious debris before fitting.
    """
    if mode not in VALID_SCALE_MODES:
        raise ValueError(f"mode must be one of {VALID_SCALE_MODES}, got {mode!r}")
    if mode == "none":
        return 1.0, 1.0

    source_xy = _as_xy(source_xy)
    reference_xy = _as_xy(reference_xy)
    if source_xy.shape[0] == 0 or reference_xy.shape[0] == 0:
        raise ValueError("Cannot fit a scale from an empty coordinate set")

    if mode == "extent":
        src_span = source_xy.max(axis=0) - source_xy.min(axis=0)
        ref_span = reference_xy.max(axis=0) - reference_xy.min(axis=0)
    else:  # 'max'
        src_span = source_xy.max(axis=0)
        ref_span = reference_xy.max(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(src_span > 0, ref_span / src_span, 1.0)

    sx, sy = float(factors[0]), float(factors[1])
    if preserve_aspect:
        isotropic = min(sx, sy)
        logger.info(
            "Fitted isotropic scale %.4f (per-axis would be %.4f, %.4f)", isotropic, sx, sy
        )
        return isotropic, isotropic

    logger.info("Fitted anisotropic scale sx=%.4f sy=%.4f", sx, sy)
    return sx, sy


def build_pretransform(
    source_xy,
    reference_xy=None,
    scale: Optional[Tuple[float, float]] = None,
    scale_mode: str = "extent",
    preserve_aspect: bool = True,
    rotation: float = 0.0,
    flip_x: bool = False,
    flip_y: bool = False,
    align_centroids: bool = True,
    center: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Build the coarse pre-registration affine for a source dataset.

    Applies flip, then rotation, then scale, all about ``center`` (the source
    centroid by default), and finally an optional translation putting the
    source centroid on the reference centroid.

    Parameters
    ----------
    source_xy : array-like
        ``(N, 2)`` source coordinates.
    reference_xy : array-like, optional
        ``(M, 2)`` reference coordinates. Required for fitted scaling or
        centroid alignment.
    scale : tuple of float, optional
        Explicit ``(sx, sy)``. Overrides fitting against the reference.
    scale_mode : {'extent', 'max', 'none'}, optional
        Fitting mode when ``scale`` is not given. See
        :func:`fit_scale_to_reference`.
    preserve_aspect : bool, optional
        Isotropic scaling when fitting.
    rotation : float, optional
        Counter-clockwise degrees.
    flip_x, flip_y : bool, optional
        Mirror horizontally / vertically.
    align_centroids : bool, optional
        Translate so the source centroid lands on the reference centroid.
    center : sequence of float, optional
        Centre for flip/rotate/scale; defaults to the source centroid.

    Returns
    -------
    numpy.ndarray
        3x3 homogeneous affine, applicable with
        :func:`smint.alignment.centroid_registration.apply_affine`.
    """
    source_xy = _as_xy(source_xy)
    if source_xy.shape[0] == 0:
        raise ValueError("Cannot build a pre-transform from zero source points")

    pivot = np.asarray(center, dtype=float) if center is not None else source_xy.mean(axis=0)

    # Orientation first. Scale is fitted from bounding-box extents, and those
    # change under rotation -- fitting on the raw source would bake in the
    # error (measured 8.10 against a true 10.0 on a 30-degree rotated shape).
    orientation = compose(
        flip_matrix(flip_x, flip_y, center=pivot),
        rotation_matrix(rotation, center=pivot),
    )

    if scale is None:
        if reference_xy is not None and scale_mode != "none":
            sx, sy = fit_scale_to_reference(
                apply_pretransform(source_xy, orientation),
                reference_xy,
                mode=scale_mode,
                preserve_aspect=preserve_aspect,
            )
        else:
            sx, sy = 1.0, 1.0
    else:
        sx, sy = float(scale[0]), float(scale[1] if len(scale) > 1 else scale[0])

    matrix = compose(orientation, scale_matrix(sx, sy, center=pivot))

    if align_centroids and reference_xy is not None:
        reference_xy = _as_xy(reference_xy)
        moved_centroid = apply_pretransform(source_xy, matrix).mean(axis=0)
        offset = reference_xy.mean(axis=0) - moved_centroid
        matrix = compose(matrix, translation_matrix(*offset))

    return matrix


def apply_pretransform(coords, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 affine to ``(N, 2)`` ``(x, y)`` coordinates.

    Thin wrapper over :func:`smint.alignment.centroid_registration.apply_affine`
    so pre-registration reads naturally on its own.
    """
    from .centroid_registration import apply_affine

    return apply_affine(_as_xy(coords), matrix)


def describe_pretransform(matrix: np.ndarray) -> dict:
    """
    Decompose an affine into human-readable scale, rotation and shear.

    Useful for sanity-checking a transform before committing to it -- a
    negative determinant means the transform includes a reflection, which is
    easy to introduce accidentally by combining a flip with a rotation.

    Returns
    -------
    dict
        ``scale_x``, ``scale_y``, ``rotation_deg``, ``shear_deg``,
        ``translation``, ``determinant``, ``reflects``.
    """
    matrix = np.asarray(matrix, dtype=float)
    linear = matrix[:2, :2]

    scale_x = float(np.hypot(*linear[:, 0]))
    # Remove the x component from the y column to recover shear and true sy.
    shear_numerator = float(linear[:, 0] @ linear[:, 1])
    shear = shear_numerator / scale_x**2 if scale_x else 0.0
    y_orthogonal = linear[:, 1] - shear * linear[:, 0]
    scale_y = float(np.hypot(*y_orthogonal))

    determinant = float(np.linalg.det(linear))
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": float(np.rad2deg(np.arctan2(linear[1, 0], linear[0, 0]))),
        "shear_deg": float(np.rad2deg(np.arctan(shear))),
        "translation": (float(matrix[0, 2]), float(matrix[1, 2])),
        "determinant": determinant,
        "reflects": determinant < 0,
    }


def overlap_score(source_xy, reference_xy, pixel_size: float = 50.0) -> float:
    """
    Fraction of occupied reference bins that the source also occupies.

    A cheap, correspondence-free way to tell whether a coarse pre-registration
    is getting better or worse -- useful for driving an interactive rotate/flip
    control, where a full registration per adjustment would be far too slow.
    Ranges 0 to 1; higher is better.

    Parameters
    ----------
    source_xy, reference_xy : array-like
        ``(N, 2)`` coordinate sets in a shared frame.
    pixel_size : float, optional
        Bin size. Too small and nothing overlaps; too large and everything
        does. Roughly the tissue feature scale works well.
    """
    source_xy = _as_xy(source_xy)
    reference_xy = _as_xy(reference_xy)
    if source_xy.shape[0] == 0 or reference_xy.shape[0] == 0:
        return 0.0

    stacked = np.vstack([source_xy, reference_xy])
    origin = stacked.min(axis=0)

    def occupied(points):
        idx = np.floor((points - origin) / pixel_size).astype(np.int64)
        return set(map(tuple, idx))

    reference_bins = occupied(reference_xy)
    if not reference_bins:
        return 0.0
    return len(occupied(source_xy) & reference_bins) / len(reference_bins)
