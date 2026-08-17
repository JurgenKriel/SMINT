"""
Coordinate-column detection for spatial tables.

The project uses several naming conventions depending on where a table came
from -- ``x_final``/``y_final`` for ST annotations, ``x_centroid``/
``y_centroid`` for Xenium, ``centroid_x``/``centroid_y`` for segmented nuclei,
plain ``x``/``y`` for metabolomics matrices. Rather than making every caller
guess, resolve them in one place.
"""

from typing import Optional, Sequence, Tuple

#: Column-name candidates, in priority order.
X_CANDIDATES = ("x_final", "x_centroid", "centroid_x", "x_transformed", "x", "X")
Y_CANDIDATES = ("y_final", "y_centroid", "centroid_y", "y_transformed", "y", "Y")


def detect_coordinate_columns(
    columns: Sequence[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Guess the ``(x, y)`` column names in a table.

    Candidates are tried in priority order, so a table carrying both
    ``x_final`` and ``x`` resolves to the more specific one.

    Parameters
    ----------
    columns : sequence of str
        The table's column names.

    Returns
    -------
    (x_col, y_col)
        Either may be None if nothing matched.
    """
    lookup = {str(c).lower(): c for c in columns}
    x_col = next((lookup[c.lower()] for c in X_CANDIDATES if c.lower() in lookup), None)
    y_col = next((lookup[c.lower()] for c in Y_CANDIDATES if c.lower() in lookup), None)
    return x_col, y_col


def require_coordinate_columns(columns: Sequence[str], source: str = "table"):
    """
    Detect coordinate columns, raising a useful error when they are missing.

    Raises
    ------
    ValueError
        Naming the source and listing what was actually available, since the
        usual cause is pointing at the wrong file.
    """
    x_col, y_col = detect_coordinate_columns(columns)
    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not detect coordinate columns in {source}. "
            f"Available: {', '.join(map(str, list(columns)[:20]))}. "
            "Pass column names explicitly."
        )
    return x_col, y_col
