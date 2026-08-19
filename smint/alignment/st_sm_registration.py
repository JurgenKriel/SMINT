"""
Spatial Transcriptomics <-> Spatial Metabolomics (ST/SM) registration for SMINT.

This module wraps the STalign LDDMM workflow used to register spatial
metabolomics (SM) sections onto spatial transcriptomics (ST) sections.

The workflow is deliberately split into two phases around a **manual**
landmark-annotation step, which is how the pipeline is actually run:

    Phase 1  prepare_landmark_inputs()   rasterize ST + SM, write .npz files
      ...    [manual] run point_annotator.py on each .npz, save *_points.npy
    Phase 2  register_sm_to_st()         load landmarks, run LDDMM, transform

Coordinate conventions
----------------------
STalign works in **row-column** order throughout, not xy. ``LDDMM`` takes
grids as ``[Y, X]``, and ``transform_points_source_to_target`` both accepts
and returns points as ``(row, col) == (y, x)`` -- it applies the transform
in place and does not transpose.

Registering through STalign therefore returns coordinates in a frame whose
axes are transposed relative to the source dataset. ``transform_points``
exposes this via the ``orientation`` argument:

``orientation="dataset"`` (default)
    Assign ``out[:, 0] -> x`` and ``out[:, 1] -> y``. This restores the
    original dataset's orientation and reproduces the historical Venture
    outputs (``x_transformed`` / ``y_transformed`` in
    ``venture_pt5/aligned_lddmm_2/``). Use this unless you know otherwise.

``orientation="stalign"``
    Assign ``out[:, 0] -> y`` and ``out[:, 1] -> x``, i.e. true xy in the ST
    target frame. Provided for callers that want STalign's native frame.

The two differ by a transpose. Mixing them silently produces a flipped
overlay, so the choice is always explicit and always recorded in the output.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Column names written by :func:`save_transformed_data`.
TRANSFORMED_X_COL = "x_transformed"
TRANSFORMED_Y_COL = "y_transformed"

_VALID_ORIENTATIONS = ("dataset", "stalign")


# --------------------------------------------------------------------------
# STalign import
# --------------------------------------------------------------------------

def _stalign():
    """
    Import and return the STalign *module*.

    The installed distribution ships an empty ``STalign/__init__.py`` with the
    implementation in ``STalign/STalign.py``. A bare ``import STalign`` therefore
    succeeds but yields an empty namespace, so every call fails later with a
    confusing ``AttributeError``. Always go through this helper.

    Returns
    -------
    module
        The ``STalign.STalign`` module.

    Raises
    ------
    ImportError
        If STalign (or a dependency) cannot be imported, with the underlying
        cause attached.
    """
    try:
        from STalign import STalign as _ST
    except Exception as exc:  # ImportError, AttributeError from numpy 2.x, ...
        raise ImportError(
            "Could not import STalign. This usually means the active "
            "environment has numpy>=2, which breaks STalign's `nptyping` "
            "dependency (`np.object0` was removed). Use an environment with "
            "numpy<2, e.g. STalign_env. Original error: %r" % (exc,)
        ) from exc

    if not hasattr(_ST, "rasterize"):
        raise ImportError(
            "Imported STalign but it has no `rasterize`; the package layout is "
            "not as expected (implementation should live in STalign/STalign.py)."
        )
    return _ST


def stalign_available() -> bool:
    """Return True if STalign can actually be imported and used."""
    try:
        _stalign()
        return True
    except ImportError:
        return False


def _preload_cuda_linalg() -> None:
    """
    Load torch's CUDA linalg backend eagerly, by absolute path.

    ``torch.linalg.inv`` -- which STalign's LDDMM calls once per iteration to
    invert the affine -- lazily ``dlopen``s ``libtorch_cuda_linalg.so`` by
    *bare name*, so the loader has to find it on a search path. Whether it does
    depends on which object's RPATH the dynamic loader attributes the call to,
    which in turn depends on the environment the worker inherited. A job
    submitted from the napari GUI can therefore die several minutes in with::

        RuntimeError: Error in dlopen: libtorch_cuda_linalg.so:
        cannot open shared object file: No such file or directory

    on the same node and the same install where an identical run from a shell
    succeeds. Loading the library here by absolute path puts it in the link map
    under its SONAME, so torch's later bare-name ``dlopen`` matches the open
    handle and never touches the filesystem search path.

    Best effort: a failure here is logged, not raised, so that CPU runs and any
    future torch layout keep working.
    """
    import ctypes
    import torch

    lib = Path(torch.__file__).parent / "lib" / "libtorch_cuda_linalg.so"
    if not lib.exists():
        logger.debug("No CUDA linalg backend to preload at %s", lib)
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        logger.debug("Preloaded %s", lib)
    except OSError as exc:
        logger.warning(
            "Could not preload %s (%s); torch.linalg on CUDA may fail with a "
            "dlopen error.", lib, exc
        )


def _resolve_device(device=None) -> str:
    """Return an explicit torch device string, preferring CUDA when present."""
    import torch

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda"):
        _preload_cuda_linalg()
    return device


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

def _is_numeric_like_header(name) -> bool:
    """True if a column name starts with a digit, or '.' followed by a digit."""
    if not isinstance(name, str):
        return False
    s = name.strip()
    if not s:
        return False
    return s[0].isdigit() or (s[0] == "." and len(s) > 1 and s[1].isdigit())


def add_x_prefix_to_numeric_headers(
    df: pd.DataFrame,
    exclude: Optional[Iterable[str]] = None,
    already_prefixed_ok: bool = True,
) -> pd.DataFrame:
    """
    Prefix numeric-like column headers with ``X`` (e.g. ``72.0796`` -> ``X72.0796``).

    m/z columns are read from CSV as bare numbers, which are awkward as
    identifiers and collide with R's own ``X`` prefixing on round-trip.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose columns should be renamed.
    exclude : iterable of str, optional
        Column names to leave untouched (e.g. ``{'x', 'y', 'is_edge'}``).
    already_prefixed_ok : bool, optional
        If True, leave headers that already start with ``X`` unchanged.

    Returns
    -------
    pandas.DataFrame
        Renamed copy, or the original frame if nothing needed renaming.
    """
    exclude_set: Set[str] = set(exclude or [])

    def new_name(col):
        if col in exclude_set:
            return col
        if _is_numeric_like_header(col):
            if already_prefixed_ok and col.startswith("X"):
                return col
            return f"X{col}"
        return col

    mapping = {col: new_name(col) for col in df.columns}
    if any(mapping[k] != k for k in mapping):
        return df.rename(columns=mapping)
    return df


def _sniff_delimiter(path: str, default: str = ",") -> str:
    """
    Detect a CSV delimiter from the header line.

    ``pd.read_csv(sep=None)`` also sniffs, but forces the pure-Python parser,
    which is roughly an order of magnitude slower. These matrices run to
    ~450 MB / 460k rows, so we sniff once and hand the C parser an explicit
    separator.
    """
    import csv as _csv

    with open(path, "r", newline="") as fh:
        sample = fh.readline()
    if not sample:
        return default
    try:
        return _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except _csv.Error:
        counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else default


def read_sm_matrix(
    mtx_file: str,
    scale_xy: float = 10.0,
    rotate_left_90: bool = False,
    keep_positive: bool = True,
    origin: Optional[Tuple[float, float]] = None,
    usecols: Optional[Sequence[str]] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read a spatial metabolomics matrix file and normalise it for registration.

    Applies, in order: delimiter sniffing, ``x``/``y`` name standardisation,
    ``X``-prefixing of numeric m/z headers, coordinate scaling, optional 90 deg
    CCW rotation, and an optional shift to keep coordinates non-negative.

    The SM grid is stored in pixel units; ``scale_xy`` converts it to the same
    micron scale as the ST centroids, so registration starts near the right
    magnitude. For the Venture cohort this is ``10.0``.

    Parameters
    ----------
    mtx_file : str
        Path to the ``*_information_matrix_tissue_only_full_with_xy.csv`` file.
    scale_xy : float, optional
        Multiplier applied to ``x`` and ``y``. Use ``1.0`` to disable.
    rotate_left_90 : bool, optional
        If True, apply ``(x, y) -> (-y, x)`` *after* scaling.
    keep_positive : bool, optional
        If True, translate so coordinates are >= ``origin`` (default ``(0, 0)``).
    origin : tuple of float, optional
        Target lower bound ``(ox, oy)`` when ``keep_positive`` is True.
    usecols : sequence of str, optional
        Restrict to these columns. Passing ``['x', 'y']`` reads coordinates
        only, which is much faster when the m/z intensities aren't needed
        (e.g. when rasterizing for landmark annotation).
    x_col, y_col : str, optional
        Which columns hold the coordinates. By default the columns literally
        named ``x`` and ``y`` are used; pass these when the matrix carries
        several coordinate pairs and you need a specific one.
    verbose : bool, optional
        Log progress.

    Returns
    -------
    pandas.DataFrame
        Full frame with ``x``/``y`` columns plus ``X``-prefixed m/z features.

    Notes
    -----
    This returns the whole DataFrame rather than a ``(coords, data, mz)``
    tuple; use :func:`sm_coordinates` to pull the coordinate array out.
    """
    if not os.path.exists(mtx_file):
        raise FileNotFoundError(f"Metabolomics file not found: {mtx_file}")

    if verbose:
        logger.info("Reading matrix file: %s", os.path.basename(mtx_file))

    sep = _sniff_delimiter(mtx_file)
    read_kwargs = {"sep": sep}
    if usecols is not None:
        # Match case-insensitively so callers can ask for 'x'/'y' regardless
        # of how the header is cased.
        header = pd.read_csv(mtx_file, sep=sep, nrows=0).columns
        wanted = {str(c).lower() for c in usecols}
        read_kwargs["usecols"] = [c for c in header if str(c).lower() in wanted]

    df = pd.read_csv(mtx_file, **read_kwargs)

    if x_col is not None or y_col is not None:
        missing = [c for c in (x_col, y_col) if c is not None and c not in df.columns]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in {os.path.basename(mtx_file)}. "
                f"Available: {', '.join(map(str, df.columns[:30]))}"
            )
    if x_col is None:
        x_col = next((c for c in df.columns if str(c).lower() == "x"), None)
    if y_col is None:
        y_col = next((c for c in df.columns if str(c).lower() == "y"), None)
    is_edge_col = next((c for c in df.columns if str(c).lower() == "is_edge"), None)

    if x_col is None or y_col is None:
        raise ValueError(
            "Could not find 'x' and 'y' columns in "
            f"{os.path.basename(mtx_file)}. Available: "
            f"{', '.join(map(str, df.columns[:30]))}. Pass x_col / y_col explicitly."
        )

    rename_xy = {}
    if x_col != "x":
        rename_xy[x_col] = "x"
    if y_col != "y":
        rename_xy[y_col] = "y"
    if rename_xy:
        df = df.rename(columns=rename_xy)

    exclude = {"x", "y"}
    if is_edge_col is not None:
        exclude.add(is_edge_col)
    df = add_x_prefix_to_numeric_headers(df, exclude=exclude, already_prefixed_ok=True)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    if scale_xy not in (None, 1, 1.0):
        df["x"] = df["x"] * float(scale_xy)
        df["y"] = df["y"] * float(scale_xy)

    if rotate_left_90:
        x_old, y_old = df["x"].copy(), df["y"].copy()
        df["x"], df["y"] = -y_old, x_old

    if keep_positive:
        ox, oy = (0.0, 0.0) if origin is None else origin
        min_x, min_y = df["x"].min(), df["y"].min()
        shift_x = ox - min_x if min_x < ox else 0.0
        shift_y = oy - min_y if min_y < oy else 0.0
        if shift_x or shift_y:
            df["x"] = df["x"] + shift_x
            df["y"] = df["y"] + shift_y

    if verbose:
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    return df


def sm_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Return the ``(N, 2)`` ``[x, y]`` array from an SM frame."""
    return df[["x", "y"]].to_numpy(dtype=float)


def list_columns(csv_file: str) -> list:
    """
    Column names of a CSV, without reading its contents.

    Lets a caller offer a coordinate-column picker before committing to reading
    a 450 MB matrix.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    sep = _sniff_delimiter(csv_file)
    return list(pd.read_csv(csv_file, sep=sep, nrows=0).columns)


def resolve_st_columns(
    st_file: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Decide which ST columns hold the coordinates to register.

    Explicit names win. When omitted, columns are auto-detected in the priority
    order in :mod:`smint.alignment.columns`, and the choice is logged.

    Be deliberate here. ST tables often carry several plausible coordinate
    pairs from successive processing stages -- ``x_centroid`` alongside
    ``x_new`` and ``x_new_add``, say -- and picking the wrong one produces a
    silently misregistered result rather than an error. Pass the names
    explicitly whenever more than one pair exists.

    Parameters
    ----------
    st_file : str
        Path to the ST CSV.
    x_col, y_col : str, optional
        Explicit column names; auto-detected when omitted.

    Returns
    -------
    (x_col, y_col)
    """
    from .columns import detect_coordinate_columns

    columns = list_columns(st_file)

    missing = [c for c in (x_col, y_col) if c is not None and c not in columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in {os.path.basename(st_file)}. "
            f"Available: {', '.join(map(str, columns))}"
        )
    if x_col and y_col:
        return x_col, y_col

    detected_x, detected_y = detect_coordinate_columns(columns)
    x_col = x_col or detected_x
    y_col = y_col or detected_y
    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not detect coordinate columns in {os.path.basename(st_file)}. "
            f"Available: {', '.join(map(str, columns))}. "
            "Pass st_x_col / st_y_col explicitly."
        )

    logger.info(
        "Using ST coordinate columns '%s' / '%s' from %s",
        x_col, y_col, os.path.basename(st_file),
    )
    return x_col, y_col


def read_st_annotations(
    st_file: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read an ST annotation table (e.g. ``Z2_final_aligned_annos.csv``).

    Parameters
    ----------
    st_file : str
        Path to the ST CSV.
    x_col, y_col : str, optional
        Coordinate column names. Auto-detected when omitted; see
        :func:`resolve_st_columns` for why explicit names are safer.
    verbose : bool, optional
        Log progress.

    Returns
    -------
    pandas.DataFrame
        The frame, with the coordinate columns guaranteed present and numeric.
    """
    if not os.path.exists(st_file):
        raise FileNotFoundError(f"ST file not found: {st_file}")

    x_col, y_col = resolve_st_columns(st_file, x_col, y_col)

    df = pd.read_csv(st_file)
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    if verbose:
        logger.info(
            "Read %d ST cells from %s (%s, %s)",
            len(df), os.path.basename(st_file), x_col, y_col,
        )
    return df


def st_coordinates(
    df: pd.DataFrame,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
) -> np.ndarray:
    """
    Return the ``(N, 2)`` ``[x, y]`` array from an ST frame.

    Columns are auto-detected when omitted.
    """
    from .columns import require_coordinate_columns

    if x_col is None or y_col is None:
        detected_x, detected_y = require_coordinate_columns(df.columns, "ST frame")
        x_col = x_col or detected_x
        y_col = y_col or detected_y
    return df[[x_col, y_col]].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Rasterization + the manual landmark stage
# --------------------------------------------------------------------------

def rasterize_coordinates(x, y, dx: float = 30.0, draw: int = 0, **kwargs):
    """
    Rasterize point coordinates into a density image via ``STalign.rasterize``.

    Parameters
    ----------
    x, y : array-like
        Point coordinates.
    dx : float, optional
        Pixel size, in the same units as ``x``/``y``.
    draw : int, optional
        Passed to STalign; ``0`` suppresses the figure (the default here, so
        this is safe to call in batch/SLURM jobs).
    **kwargs
        Forwarded to ``STalign.rasterize`` (``blur``, ``expand``, ...).

    Returns
    -------
    X, Y : numpy.ndarray
        Pixel locations along each axis.
    I : numpy.ndarray
        Rasterized image, channels first.
    fig : matplotlib figure or None
        The drawn figure when ``draw`` is truthy, otherwise None.

    Notes
    -----
    ``STalign.rasterize`` returns a 3-tuple when ``draw`` is falsy and a
    4-tuple when it is truthy. This wrapper always returns 4 values so callers
    can unpack unconditionally.
    """
    ST = _stalign()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    logger.info("Rasterizing %d points at dx=%s", len(x), dx)

    result = ST.rasterize(x, y, dx=dx, draw=draw, **kwargs)
    if len(result) == 4:
        X, Y, I, fig = result
    else:
        (X, Y, I), fig = result, None

    logger.info("Rasterized grid shape: %s", (I.shape,))
    return X, Y, I, fig


def save_rasterized(output_path: str, X, Y, I) -> str:
    """
    Save a rasterized image as ``.npz`` for the manual point annotator.

    ``point_annotator.py`` expects arrays named ``x``, ``y`` and ``I``, and
    writes its landmarks alongside as ``<stem>_points.npy``.

    Parameters
    ----------
    output_path : str
        Destination path; ``.npz`` is appended by numpy if absent.
    X, Y, I : numpy.ndarray
        Outputs of :func:`rasterize_coordinates`.

    Returns
    -------
    str
        The path written.
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(output_path, x=X, y=Y, I=I)
    written = output_path if output_path.endswith(".npz") else output_path + ".npz"
    logger.info("Wrote rasterized image for annotation: %s", written)
    return written


def load_landmarks(points_file: str) -> np.ndarray:
    """
    Load landmarks written by ``point_annotator.py`` into row-col order.

    The annotator stores a ``{label: [(x, y)]}`` dict. STalign's
    ``L_T_from_points`` and ``LDDMM`` both expect points in **row-col**
    ``(y, x)`` order, so the pair is swapped on load.

    Parameters
    ----------
    points_file : str
        Path to a ``*_points.npy`` file.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array in ``(row, col) == (y, x)`` order, ordered by label.

    Raises
    ------
    FileNotFoundError
        If the landmark file is missing -- usually meaning the manual
        annotation step has not been run for this section yet.
    """
    if not os.path.exists(points_file):
        raise FileNotFoundError(
            f"Landmark file not found: {points_file}. Run point_annotator.py on "
            "the matching .npz first (this step is intentionally manual)."
        )

    raw = np.load(points_file, allow_pickle=True).tolist()
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected landmark format in {points_file}: {type(raw)!r}")

    points = []
    for key in sorted(raw.keys(), key=lambda k: (len(str(k)), str(k))):
        entry = raw[key][0]
        # stored (x, y) -> emit (y, x) == (row, col)
        points.append([entry[1], entry[0]])

    arr = np.asarray(points, dtype=float)
    logger.info("Loaded %d landmarks from %s", len(arr), os.path.basename(points_file))
    return arr


def check_landmark_pair(points_source: np.ndarray, points_target: np.ndarray) -> None:
    """
    Validate a source/target landmark pair before running LDDMM.

    Raises
    ------
    ValueError
        If the counts differ or there are fewer than 3 points (an affine fit
        needs 3), which otherwise surfaces as an opaque failure inside LDDMM.
    """
    if points_source.shape != points_target.shape:
        raise ValueError(
            "Landmark count/shape mismatch: source %s vs target %s. The two "
            "annotations must have the same number of correspondences, in the "
            "same order." % (points_source.shape, points_target.shape)
        )
    if points_source.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 landmarks for an affine initialisation, got "
            f"{points_source.shape[0]}."
        )


def affine_from_landmarks(points_source: np.ndarray, points_target: np.ndarray):
    """
    Compute an affine matrix from landmark correspondences.

    Parameters
    ----------
    points_source, points_target : numpy.ndarray
        ``(N, 2)`` arrays in row-col order, as returned by :func:`load_landmarks`.

    Returns
    -------
    torch.Tensor
        3x3 affine matrix.

    Notes
    -----
    :func:`run_lddmm_alignment` derives its own initialisation from
    ``points_source``/``points_target`` internally, so this is **not** needed
    for the standard workflow. It is kept for inspecting or seeding a
    landmark-only (affine) registration.
    """
    import torch

    ST = _stalign()
    check_landmark_pair(points_source, points_target)
    L, T = ST.L_T_from_points(points_source, points_target)
    return ST.to_A(torch.tensor(L), torch.tensor(T))


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def run_lddmm_alignment(
    source_grid: Sequence,
    source_image,
    target_grid: Sequence,
    target_image,
    points_source: Optional[np.ndarray] = None,
    points_target: Optional[np.ndarray] = None,
    niter: int = 1000,
    epV: float = 200.0,
    device: Optional[str] = None,
    **params,
):
    """
    Run STalign LDDMM to register a source image onto a target image.

    Parameters
    ----------
    source_grid, target_grid : sequence
        Grids as ``[Y, X]`` (row-col), matching STalign's convention.
    source_image, target_image : numpy.ndarray
        Rasterized images from :func:`rasterize_coordinates`.
    points_source, points_target : numpy.ndarray, optional
        Landmark correspondences in row-col order. When supplied, LDDMM uses
        them to initialise the affine, which is what makes registration
        converge on ST/SM pairs; without them it starts from identity and
        typically fails to find the tissue.
    niter : int, optional
        Optimisation iterations.
    epV : float, optional
        Velocity-field regularisation weight.
    device : str, optional
        Torch device; defaults to CUDA when available, else CPU.
    **params
        Forwarded to ``STalign.LDDMM``. Note that ``sigmaM``/``sigmaB``/
        ``sigmaA`` are best left at their defaults when landmarks are given.

    Returns
    -------
    dict
        STalign LDDMM output, including ``A``, ``v`` and ``xv``.
    """
    ST = _stalign()

    if (points_source is None) != (points_target is None):
        raise ValueError(
            "points_source and points_target must be supplied together."
        )
    if points_source is not None:
        check_landmark_pair(points_source, points_target)

    device = _resolve_device(device)
    if device == "cpu":
        logger.warning(
            "Running LDDMM on CPU with niter=%d; this is slow. Submit via "
            "sbatch rather than running on the login node.", niter
        )

    call = dict(niter=niter, epV=epV, device=device)
    if points_source is not None:
        call["pointsI"] = points_source
        call["pointsJ"] = points_target
    call.update(params)

    logger.info(
        "Running LDDMM (niter=%d, epV=%s, device=%s, landmarks=%s)",
        niter, epV, device,
        0 if points_source is None else len(points_source),
    )
    out = ST.LDDMM(source_grid, source_image, target_grid, target_image, **call)
    logger.info("LDDMM alignment complete")
    return out


def transform_points(
    lddmm_output: dict,
    x,
    y,
    orientation: str = "dataset",
) -> np.ndarray:
    """
    Apply an LDDMM result to source point coordinates.

    Parameters
    ----------
    lddmm_output : dict
        Output of :func:`run_lddmm_alignment` (needs ``A``, ``v``, ``xv``).
    x, y : array-like
        Source coordinates, as plain x and y vectors.
    orientation : {'dataset', 'stalign'}, optional
        Which frame to return. See the module docstring. ``'dataset'``
        reproduces the historical Venture outputs and keeps the source
        dataset's orientation; ``'stalign'`` returns true xy in the ST frame.
        The two differ by a transpose.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array of transformed coordinates, columns ordered to match
        ``orientation``.
    """
    import torch

    if orientation not in _VALID_ORIENTATIONS:
        raise ValueError(
            f"orientation must be one of {_VALID_ORIENTATIONS}, got {orientation!r}"
        )

    ST = _stalign()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must be the same length, got {x.shape} and {y.shape}")

    A = lddmm_output["A"]
    v = lddmm_output["v"]
    xv = lddmm_output["xv"]

    # STalign takes and returns (row, col) == (y, x), applying the transform
    # in place without transposing.
    source_rowcol = np.stack([y, x], axis=1)

    device = A.device if hasattr(A, "device") else _resolve_device()
    # Scope the default-device change so we don't mutate global torch state.
    previous = torch.tensor(0.0).device
    try:
        torch.set_default_device(device)
        out = ST.transform_points_source_to_target(xv, v, A, source_rowcol)
        out = out.detach().cpu().numpy()
    finally:
        torch.set_default_device(previous)

    if orientation == "dataset":
        # out[:, 0] -> x, out[:, 1] -> y: restores the source dataset's frame.
        return np.column_stack([out[:, 0], out[:, 1]])
    # 'stalign': out is (row, col) == (y, x); emit true (x, y).
    return np.column_stack([out[:, 1], out[:, 0]])


def save_transformed_data(
    source_data: pd.DataFrame,
    transformed_coords: np.ndarray,
    output_path: Optional[str] = None,
    x_col: str = TRANSFORMED_X_COL,
    y_col: str = TRANSFORMED_Y_COL,
    orientation: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach transformed coordinates to the source table and optionally save it.

    Parameters
    ----------
    source_data : pandas.DataFrame
        The SM frame the coordinates were derived from.
    transformed_coords : numpy.ndarray
        ``(N, 2)`` output of :func:`transform_points`.
    output_path : str, optional
        CSV destination. Parent directories are created.
    x_col, y_col : str, optional
        Output column names.
    orientation : str, optional
        Recorded in ``df.attrs['orientation']`` so the convention travels with
        the frame in-process.

    Returns
    -------
    pandas.DataFrame
        Copy of ``source_data`` with the two coordinate columns appended.
    """
    if len(source_data) != len(transformed_coords):
        raise ValueError(
            f"Row count mismatch: {len(source_data)} data rows vs "
            f"{len(transformed_coords)} transformed coordinates."
        )

    result = source_data.copy()
    result[x_col] = transformed_coords[:, 0]
    result[y_col] = transformed_coords[:, 1]
    if orientation is not None:
        result.attrs["orientation"] = orientation

    if output_path:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Saved transformed data to %s", output_path)

    return result


# --------------------------------------------------------------------------
# Phase 1 / Phase 2 orchestration
# --------------------------------------------------------------------------

def prepare_landmark_inputs(
    st_file: str,
    sm_file: str,
    output_prefix: str,
    dx: float = 30.0,
    st_x_col: Optional[str] = None,
    st_y_col: Optional[str] = None,
    sm_kwargs: Optional[dict] = None,
) -> dict:
    """
    Phase 1: rasterize an ST/SM pair and write ``.npz`` files for annotation.

    After this returns, run ``point_annotator.py`` on each ``.npz`` and pick the
    same anatomical landmarks, in the same order, in both. The annotator writes
    ``<prefix>_st_points.npy`` / ``<prefix>_sm_points.npy`` next to them, which
    :func:`register_sm_to_st` then consumes.

    Parameters
    ----------
    st_file : str
        ST annotation CSV.
    sm_file : str
        SM matrix CSV.
    output_prefix : str
        Path prefix. Writes ``<prefix>_st.npz`` / ``<prefix>_sm.npz``; the
        annotator then appends ``_points.npy`` to each stem, yielding the
        ``<prefix>_st_points.npy`` / ``<prefix>_sm_points.npy`` files that
        :func:`register_sm_to_st` expects.
    dx : float, optional
        Rasterization pixel size, in microns. Must match between the pair.
    st_x_col, st_y_col : str, optional
        ST coordinate columns.
    sm_kwargs : dict, optional
        Forwarded to :func:`read_sm_matrix` (e.g. ``rotate_left_90``).

    Returns
    -------
    dict
        Paths written and the rasterized arrays, keyed ``st_npz``, ``sm_npz``,
        ``st_raster``, ``sm_raster``, plus ``st_points``/``sm_points`` giving
        the landmark paths the annotator will produce.
    """
    sm_kwargs = dict(sm_kwargs or {})
    st_x_col, st_y_col = resolve_st_columns(st_file, st_x_col, st_y_col)

    st_df = read_st_annotations(st_file, x_col=st_x_col, y_col=st_y_col)
    sm_df = read_sm_matrix(sm_file, **sm_kwargs)

    st_xy = st_coordinates(st_df, st_x_col, st_y_col)
    sm_xy = sm_coordinates(sm_df)

    X_st, Y_st, I_st, _ = rasterize_coordinates(st_xy[:, 0], st_xy[:, 1], dx=dx)
    X_sm, Y_sm, I_sm, _ = rasterize_coordinates(sm_xy[:, 0], sm_xy[:, 1], dx=dx)

    st_npz = save_rasterized(f"{output_prefix}_st", X_st, Y_st, I_st)
    sm_npz = save_rasterized(f"{output_prefix}_sm", X_sm, Y_sm, I_sm)

    st_points = st_npz.replace(".npz", "_points.npy")
    sm_points = sm_npz.replace(".npz", "_points.npy")

    logger.info(
        "Phase 1 complete. Annotate both .npz files with point_annotator.py, "
        "picking the same landmarks in the same order in each:\n"
        "  python point_annotator.py %s %s\n"
        "This writes %s and %s, which register_sm_to_st() consumes.",
        st_npz, sm_npz, st_points, sm_points,
    )
    return {
        "st_npz": st_npz,
        "sm_npz": sm_npz,
        "st_points": st_points,
        "sm_points": sm_points,
        "st_raster": (X_st, Y_st, I_st),
        "sm_raster": (X_sm, Y_sm, I_sm),
    }


def register_sm_to_st(
    st_file: str,
    sm_file: str,
    st_points_file: str,
    sm_points_file: str,
    output_path: Optional[str] = None,
    dx: float = 30.0,
    niter: int = 1000,
    epV: float = 200.0,
    device: Optional[str] = None,
    orientation: str = "dataset",
    st_x_col: Optional[str] = None,
    st_y_col: Optional[str] = None,
    sm_kwargs: Optional[dict] = None,
    lddmm_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Phase 2: register an SM section onto an ST section using manual landmarks.

    Rasterizes both modalities, runs landmark-initialised LDDMM with SM as
    source and ST as target, transforms every SM pixel, and returns the SM
    table with transformed coordinates appended.

    Parameters
    ----------
    st_file, sm_file : str
        Input tables for the same physical section.
    st_points_file, sm_points_file : str
        ``*_points.npy`` landmark files from the manual annotation step.
    output_path : str, optional
        CSV destination for the transformed SM table.
    dx : float, optional
        Rasterization pixel size; must match what was annotated.
    niter, epV : optional
        LDDMM optimisation settings.
    device : str, optional
        Torch device; defaults to CUDA when available.
    orientation : {'dataset', 'stalign'}, optional
        Output coordinate frame. See the module docstring.
    st_x_col, st_y_col : str, optional
        Which ST columns to register. Auto-detected when omitted; pass them
        explicitly when the table carries several coordinate pairs, since the
        wrong choice misregisters silently rather than raising.
    sm_kwargs : dict, optional
        Forwarded to :func:`read_sm_matrix`, including ``x_col``/``y_col``.
    lddmm_params : dict, optional
        Extra parameters forwarded to ``STalign.LDDMM``.

    Returns
    -------
    pandas.DataFrame
        SM data with ``x_transformed`` / ``y_transformed`` columns.
    """
    sm_kwargs = dict(sm_kwargs or {})
    lddmm_params = dict(lddmm_params or {})
    st_x_col, st_y_col = resolve_st_columns(st_file, st_x_col, st_y_col)

    st_df = read_st_annotations(st_file, x_col=st_x_col, y_col=st_y_col)
    sm_df = read_sm_matrix(sm_file, **sm_kwargs)

    st_xy = st_coordinates(st_df, st_x_col, st_y_col)
    sm_xy = sm_coordinates(sm_df)

    X_st, Y_st, I_st, _ = rasterize_coordinates(st_xy[:, 0], st_xy[:, 1], dx=dx)
    X_sm, Y_sm, I_sm, _ = rasterize_coordinates(sm_xy[:, 0], sm_xy[:, 1], dx=dx)

    points_st = load_landmarks(st_points_file)
    points_sm = load_landmarks(sm_points_file)
    check_landmark_pair(points_sm, points_st)

    # Source is SM, target is ST; grids are [Y, X] (row-col).
    out = run_lddmm_alignment(
        [Y_sm, X_sm], I_sm,
        [Y_st, X_st], I_st,
        points_source=points_sm,
        points_target=points_st,
        niter=niter,
        epV=epV,
        device=device,
        **lddmm_params,
    )

    transformed = transform_points(out, sm_xy[:, 0], sm_xy[:, 1], orientation=orientation)

    return save_transformed_data(
        sm_df, transformed, output_path=output_path, orientation=orientation
    )


def find_sm_matrix_files(base_path: str, layers: Sequence[int]) -> dict:
    """
    Locate SM matrix files for the given layers.

    Looks under ``base_path/layer_{n}/matrix/*information_matrix_tissue_only_full_with_xy.csv``.

    Parameters
    ----------
    base_path : str
        Root of the ``integrated`` directory for a patient.
    layers : sequence of int
        Layer numbers to search.

    Returns
    -------
    dict
        ``{layer: [paths]}``.
    """
    found = {}
    for layer in layers:
        matrix_dir = os.path.join(base_path, f"layer_{layer}", "matrix")
        found[layer] = sorted(
            glob.glob(
                os.path.join(
                    matrix_dir, "*information_matrix_tissue_only_full_with_xy.csv"
                )
            )
        )
    return found
