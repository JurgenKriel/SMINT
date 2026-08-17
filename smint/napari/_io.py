"""
Data loading and landmark I/O for the SMINT napari plugin.

Deliberately free of napari and Qt imports so the logic can be exercised
headlessly; the widgets in :mod:`smint.napari._widgets` call into here.

Nothing in this module imports STalign. Density images are built with numpy
and scipy rather than ``STalign.rasterize``, because the plugin runs in the
napari environment where STalign is unavailable -- and because these images are
for display only, so they need not reproduce STalign's rasteriser. Landmarks are
picked directly on point coordinates in real units, so no pixel-to-micron
conversion enters the registration path.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from smint.alignment.columns import (  # re-exported for plugin callers
    X_CANDIDATES,
    Y_CANDIDATES,
    detect_coordinate_columns,
)

logger = logging.getLogger(__name__)


def load_points_table(
    path: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    max_points: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load ``(x, y)`` coordinates from a CSV for display.

    Only the coordinate columns are read. The metabolomics matrices run to
    ~450 MB with 460k rows and hundreds of m/z columns, none of which the
    viewer needs.

    Parameters
    ----------
    path : str
        CSV path.
    x_col, y_col : str, optional
        Coordinate columns; auto-detected when omitted.
    max_points : int, optional
        Randomly subsample to at most this many points for display. Does not
        affect registration, which always reads the full file in the worker.
    random_state : int, optional
        Subsampling seed.

    Returns
    -------
    coords : numpy.ndarray
        ``(N, 2)`` array of ``(x, y)``.
    frame : pandas.DataFrame
        The loaded coordinate columns, renamed to ``x``/``y``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    header = pd.read_csv(path, nrows=0).columns
    if x_col is None or y_col is None:
        detected_x, detected_y = detect_coordinate_columns(header)
        x_col = x_col or detected_x
        y_col = y_col or detected_y

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not detect coordinate columns in {os.path.basename(path)}. "
            f"Available: {', '.join(map(str, header[:20]))}. "
            "Pass x_col/y_col explicitly."
        )

    frame = pd.read_csv(path, usecols=[x_col, y_col]).rename(
        columns={x_col: "x", y_col: "y"}
    )
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["x", "y"])

    if max_points is not None and len(frame) > max_points:
        frame = frame.sample(max_points, random_state=random_state)
        logger.info("Subsampled %s to %d points for display", os.path.basename(path), max_points)

    logger.info("Loaded %d points from %s (%s, %s)", len(frame), os.path.basename(path), x_col, y_col)
    return frame[["x", "y"]].to_numpy(dtype=float), frame


def density_image(
    coords: np.ndarray, pixel_size: float = 30.0, smoothing: float = 1.0
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Build a smoothed density image from point coordinates, for display.

    Gives the tissue context that makes anatomical landmarks findable; the
    Points layer alone can be hard to read at low zoom.

    Parameters
    ----------
    coords : numpy.ndarray
        ``(N, 2)`` ``(x, y)`` coordinates.
    pixel_size : float, optional
        Bin size in coordinate units.
    smoothing : float, optional
        Gaussian sigma in pixels; 0 disables smoothing.

    Returns
    -------
    image : numpy.ndarray
        2D density array, indexed ``[row, col]`` = ``[y, x]``.
    origin : tuple of float
        ``(x_min, y_min)`` of the image.
    pixel_size : float
        Echoed back, for building the napari affine.

    Notes
    -----
    Returned in row-major ``[y, x]`` order to match napari's image indexing.
    Use ``origin`` and ``pixel_size`` to place it in world coordinates so it
    overlays the Points layers correctly.
    """
    if coords.shape[0] == 0:
        raise ValueError("Cannot build a density image from zero points")

    x, y = coords[:, 0], coords[:, 1]
    x_min, y_min = float(x.min()), float(y.min())
    n_x = max(int(np.ceil((x.max() - x_min) / pixel_size)) + 1, 1)
    n_y = max(int(np.ceil((y.max() - y_min) / pixel_size)) + 1, 1)

    image, _, _ = np.histogram2d(
        y, x, bins=[n_y, n_x],
        range=[[y_min, y_min + n_y * pixel_size], [x_min, x_min + n_x * pixel_size]],
    )

    if smoothing:
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=smoothing)

    return image, (x_min, y_min), pixel_size


# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------

def save_landmarks(points_xy: np.ndarray, path: str) -> Path:
    """
    Save landmarks in ``point_annotator.py``'s on-disk format.

    That format is a ``{label: [(x, y)]}`` dict pickled into a ``.npy``, with
    labels as ``'1'``, ``'2'``, ... in click order. Writing the same thing keeps
    existing Venture landmark files interchangeable with plugin output, and
    lets :func:`smint.alignment.st_sm_registration.load_landmarks` read either.

    Parameters
    ----------
    points_xy : numpy.ndarray
        ``(N, 2)`` landmarks as ``(x, y)`` in world coordinates.
    path : str
        Destination ``.npy``.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError(f"Expected an (N, 2) array of landmarks, got {points_xy.shape}")

    payload = {str(i + 1): [(float(x), float(y))] for i, (x, y) in enumerate(points_xy)}

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, payload, allow_pickle=True)
    logger.info("Saved %d landmarks to %s", len(points_xy), target)
    return target


def load_landmarks_xy(path: str) -> np.ndarray:
    """
    Load landmarks as ``(N, 2)`` ``(x, y)`` in click order.

    The registration code wants row-col ``(y, x)`` and has its own loader;
    this one returns xy for display in napari.
    """
    raw = np.load(path, allow_pickle=True).tolist()
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected landmark format in {path}: {type(raw)!r}")

    ordered = sorted(raw.keys(), key=lambda k: (len(str(k)), str(k)))
    return np.array([[raw[k][0][0], raw[k][0][1]] for k in ordered], dtype=float)


def landmark_pair_status(n_source: int, n_target: int) -> Tuple[bool, str]:
    """
    Whether a landmark pair is ready to register, and why not if it isn't.

    Returns
    -------
    (ok, message)
    """
    if n_source == 0 and n_target == 0:
        return False, "No landmarks placed yet."
    if n_source != n_target:
        return False, (
            f"Counts differ: {n_source} source vs {n_target} target. "
            "Each landmark needs a partner, in the same order."
        )
    if n_source < 3:
        return False, f"Need at least 3 pairs for an affine, have {n_source}."
    return True, f"{n_source} landmark pairs ready."
