"""
Polygon stitching and cell / nucleus reconciliation.

A Python port of ``stitching/stitching.Rmd`` (Joel Moffet), which resolves the
artefacts left when an image is segmented in overlapping tiles or fields of
view. It runs in two stages.

**Stage 1 -- stitch each object type** ([`stitch_polygons`][smint.segmentation.stitching.stitch_polygons]). For every
intersecting pair, the *maximum overlap* is the intersection area divided by
the smaller polygon's area (the larger of the two fractions).

1. Merge: union every pair whose maximum overlap exceeds ``merge_threshold``
   (the same object seen by two tiles), repeating until nothing more merges, so
   chains of fragments collapse into one object.
2. Trim: for pairs below ``trim_threshold`` (a minor collision), remove the
   overlap from both. A polygon trimmed by several neighbours loses every
   overlap.
3. Remove: recompute overlaps; for pairs at or above ``trim_threshold``, drop
   the smaller polygon.

The phases run in this order, each on the result of the last. Cells are first
filtered to those touching a stitched nucleus ([`filter_by_nuclei`][smint.segmentation.stitching.filter_by_nuclei]).

**Stage 2 -- reconcile cells and nuclei** ([`reconcile_cells_and_nuclei`][smint.segmentation.stitching.reconcile_cells_and_nuclei]):
drop cells without a credible nucleus, grow cells over nuclei they mostly
contain, split nuclei straddling several cells, remove duplicate and contained
objects, then assign every nucleus to exactly one cell.

Polygons are Shapely 2 geometries in any consistent unit; overlaps smaller than
``min_overlap_area`` (in squared units) are treated as touching edges and
ignored. The R workflow keeps a polygon split in two by a trim or cut as a
multi-part geometry; here the largest part is kept, so every object stays a
single polygon with one outline.
"""

import logging

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Tolerance for "fully contained": intersection / area >= 1 - CONTAINED_TOL
CONTAINED_TOL = 1e-6


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _largest_polygon(geom):
    """Largest Polygon part of any geometry, or None if it has no area."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    parts += [p for g in getattr(geom, "geoms", []) if g.geom_type == "MultiPolygon" for p in g.geoms]
    return max(parts, key=lambda p: p.area) if parts else None


def clean_polygons(geoms, min_area=0.0):
    """
    Make geometries valid single polygons and drop empty or tiny ones.

    Args:
        geoms (sequence): Shapely geometries.
        min_area (float): Drop polygons with area <= this.

    Returns:
        numpy.ndarray: Object array of valid Polygons.
    """
    geoms = np.asarray(list(geoms), dtype=object)
    if len(geoms) == 0:
        return geoms
    geoms = shapely.make_valid(geoms)
    out = [_largest_polygon(g) for g in geoms]
    return np.array([g for g in out if g is not None and g.area > min_area], dtype=object)


def polygons_from_outlines(outlines):
    """
    Build polygons from outline dicts with ``x_coords`` / ``y_coords``.

    Args:
        outlines (list): Dicts with ``x_coords`` and ``y_coords`` arrays.

    Returns:
        numpy.ndarray: Valid Polygons (outlines with < 3 vertices are skipped).
    """
    polys = [shapely.Polygon(np.column_stack([o['x_coords'], o['y_coords']]))
             for o in outlines if len(o['x_coords']) >= 3]
    return clean_polygons(polys)


def _pairs(a, b=None, min_overlap_area=1.0):
    """
    Intersecting pairs with their intersection area.

    With one array, returns pairs ``i < j`` within it; with two, pairs
    ``(i in a, j in b)``.

    Returns:
        tuple: ``(i, j, intersection_area)`` arrays.
    """
    same = b is None
    b = a if same else b
    if len(a) == 0 or len(b) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float)
    i, j = shapely.STRtree(b).query(a, predicate="intersects")
    if same:
        keep = i < j
        i, j = i[keep], j[keep]
    inter = shapely.area(shapely.intersection(a[i], b[j]))
    keep = inter >= min_overlap_area
    return i[keep], j[keep], inter[keep]


def _max_overlap(geoms, min_overlap_area):
    areas = shapely.area(geoms)
    i, j, inter = _pairs(geoms, min_overlap_area=min_overlap_area)
    return i, j, np.maximum(inter / areas[i], inter / areas[j]), areas


def _subtract_neighbours(geoms, targets, neighbours, originals=None):
    """``geoms[k] -= union(originals[n] for n in neighbours of k)`` for each target k."""
    originals = geoms if originals is None else originals
    out = geoms.copy()
    by_target = {}
    for k, n in zip(targets, neighbours):
        by_target.setdefault(int(k), []).append(int(n))
    for k, ns in by_target.items():
        out[k] = shapely.difference(geoms[k], shapely.union_all(originals[ns]))
    return out


# ---------------------------------------------------------------------------
# Stage 1: stitching
# ---------------------------------------------------------------------------

def merge_overlapping(geoms, merge_threshold=0.5, max_rounds=10, min_overlap_area=1.0):
    """
    Union polygons whose maximum overlap exceeds ``merge_threshold``.

    Overlapping pairs are joined into connected groups and each group is
    unioned; this repeats on the result until no pair qualifies (or
    ``max_rounds`` is reached), since a merged polygon can newly overlap
    another by more than the threshold.

    Returns:
        numpy.ndarray: Polygons after merging.
    """
    geoms = clean_polygons(geoms)
    for round_ in range(max_rounds):
        i, j, overlap, _ = _max_overlap(geoms, min_overlap_area)
        merge = overlap > merge_threshold
        if not merge.any():
            break
        n = len(geoms)
        graph = coo_matrix((np.ones(merge.sum()), (i[merge], j[merge])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        order = np.argsort(labels, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
        merged = [geoms[g[0]] if len(g) == 1 else shapely.union_all(geoms[g]) for g in groups]
        logger.debug("Merge round %d: %d -> %d", round_ + 1, n, len(groups))
        geoms = clean_polygons(merged)
    else:
        logger.warning("Merging still changing after %d rounds", max_rounds)
    return geoms


def trim_collisions(geoms, trim_threshold=0.25, min_overlap_area=1.0):
    """
    Remove small overlaps from both polygons of each pair.

    For pairs with maximum overlap below ``trim_threshold``, each polygon loses
    its intersection with the other (computed against the untrimmed
    neighbours).

    Returns:
        numpy.ndarray: Polygons after trimming.
    """
    i, j, overlap, _ = _max_overlap(geoms, min_overlap_area)
    trim = overlap < trim_threshold
    if not trim.any():
        return geoms
    targets = np.concatenate([i[trim], j[trim]])
    neighbours = np.concatenate([j[trim], i[trim]])
    return clean_polygons(_subtract_neighbours(geoms, targets, neighbours))


def remove_conflicts(geoms, trim_threshold=0.25, min_overlap_area=1.0):
    """
    Drop the smaller polygon of each pair overlapping by ``trim_threshold`` or more.

    Returns:
        numpy.ndarray: Remaining polygons.
    """
    i, j, overlap, areas = _max_overlap(geoms, min_overlap_area)
    conflict = overlap >= trim_threshold
    if not conflict.any():
        return geoms
    i, j = i[conflict], j[conflict]
    drop = np.where(areas[i] < areas[j], i, j)
    keep = np.ones(len(geoms), dtype=bool)
    keep[drop] = False
    return geoms[keep]


def filter_by_nuclei(cells, nuclei):
    """Keep only cells that touch at least one nucleus."""
    if len(cells) == 0 or len(nuclei) == 0:
        return cells[:0] if len(nuclei) == 0 else cells
    hit, _ = shapely.STRtree(nuclei).query(cells, predicate="intersects")
    return cells[np.unique(hit)]


def stitch_polygons(geoms, merge_threshold=0.5, trim_threshold=0.25,
                    max_merge_rounds=10, min_overlap_area=1.0):
    """
    Resolve duplicate and colliding polygons from overlapping tiles.

    Runs three phases in order, each on the result of the last. The *maximum
    overlap* of a pair is their intersection area divided by the smaller
    polygon's area.

    1. Merge: union pairs whose maximum overlap exceeds ``merge_threshold``
       (the same object seen by two tiles), repeating until nothing merges.
    2. Trim: for pairs below ``trim_threshold`` (a minor collision), remove
       the overlap from both.
    3. Remove: for pairs at or above ``trim_threshold``, drop the smaller
       polygon.

    Args:
        geoms (sequence): Shapely polygons.
        merge_threshold (float): Merge pairs whose maximum overlap exceeds this.
        trim_threshold (float): Trim pairs below this; remove the smaller of
            pairs at or above it (and not merged).
        max_merge_rounds (int): Cap on merge iterations.
        min_overlap_area (float): Ignore intersections smaller than this.

    Returns:
        numpy.ndarray: Stitched polygons.
    """
    n_in = len(geoms)
    geoms = merge_overlapping(geoms, merge_threshold, max_merge_rounds, min_overlap_area)
    n_merged = len(geoms)
    geoms = trim_collisions(geoms, trim_threshold, min_overlap_area)
    geoms = remove_conflicts(geoms, trim_threshold, min_overlap_area)
    logger.info("Stitched %d polygons: %d after merging, %d after trim/remove",
                n_in, n_merged, len(geoms))
    return geoms


# ---------------------------------------------------------------------------
# Stage 2: cell / nucleus reconciliation
# ---------------------------------------------------------------------------

def _resolve_overlaps(geoms, min_overlap_area=1.0):
    """
    Make a set of polygons non-overlapping.

    Identical and fully contained polygons are dropped (the contained one),
    then any polygon more than half covered by another, then remaining partial
    overlaps are cut out of the larger polygon.
    """
    areas = shapely.area(geoms)
    i, j, inter = _pairs(geoms, min_overlap_area=min_overlap_area)
    fi, fj = inter / areas[i], inter / areas[j]
    full_i, full_j = fi >= 1 - CONTAINED_TOL, fj >= 1 - CONTAINED_TOL

    drop = np.zeros(len(geoms), dtype=bool)
    drop[j[full_i & full_j]] = True                  # identical: keep the first
    drop[i[full_i & ~full_j]] = True                 # i inside j
    drop[j[full_j & ~full_i]] = True                 # j inside i

    live = ~drop[i] & ~drop[j]
    i, j, fi, fj = i[live], j[live], fi[live], fj[live]
    mostly = (fi > 0.5) | (fj > 0.5)
    smaller = np.where(areas[i] < areas[j], i, j)
    covered = np.where((fi > 0.5) & (fj > 0.5), smaller, np.where(fi > 0.5, i, j))
    drop[covered[mostly]] = True

    live = ~drop[i] & ~drop[j]
    i, j = i[live], j[live]
    larger = np.where(areas[i] >= areas[j], i, j)
    other = np.where(areas[i] >= areas[j], j, i)
    geoms = _subtract_neighbours(geoms, larger, other)
    keep = ~drop
    return clean_polygons(geoms[keep]) if keep.any() else geoms[:0]


def reconcile_cells_and_nuclei(cells, nuclei, min_overlap_area=1.0,
                               min_nucleus_fraction=0.1,
                               split_keep_fraction_2=0.4,
                               split_keep_fraction_3plus=0.3):
    """
    Make cells and nuclei consistent and assign each nucleus to one cell.

    Steps, following ``stitching.Rmd``:

    1. Drop cells touching no nucleus, and cells whose only nucleus overlaps
       them by less than ``min_nucleus_fraction`` of the nucleus.
    2. Grow a cell over a nucleus that touches only that cell and is more than
       half (but not fully) inside it.
    3. Split nuclei straddling several cells into per-cell pieces. A piece is
       kept if it holds at least 50% of the nucleus, or -- for a cell with no
       other nucleus -- at least ``split_keep_fraction_2`` (nucleus in 2 cells)
       or ``split_keep_fraction_3plus`` (3+ cells); a single-nucleus cell
       holding less than that is dropped.
    4. Remove identical, contained and mostly-covered duplicates among cells
       and among nuclei; cut remaining overlaps out of the larger object.
    5. Assign: a nucleus more than half inside a cell is absorbed into it; one
       less than half inside is cut out of the cell (and the cell out of it).
       Nuclei left without a cell become nucleus-only cells; nuclei still
       straddling a boundary are dropped; cells with no nucleus are dropped.

    Args:
        cells (sequence): Stitched cell polygons.
        nuclei (sequence): Stitched nucleus polygons.
        min_overlap_area (float): Ignore intersections smaller than this.
        min_nucleus_fraction (float): Step 1 threshold.
        split_keep_fraction_2 (float): Step 3 threshold, nucleus in 2 cells.
        split_keep_fraction_3plus (float): Step 3 threshold, nucleus in 3+ cells.

    Returns:
        tuple: ``(cells, nuclei, nucleus_cell)`` where ``nucleus_cell[k]`` is
            the index into ``cells`` of the cell containing nucleus ``k``.
    """
    cells, nuclei = clean_polygons(cells), clean_polygons(nuclei)
    n_cells_in, n_nuclei_in = len(cells), len(nuclei)

    def pairs():
        ci, ni, inter = _pairs(cells, nuclei, min_overlap_area)
        return ci, ni, inter, inter / shapely.area(nuclei)[ni]

    # 1. Cells need a credible nucleus
    ci, ni, inter, frac = pairs()
    n_per_cell = np.bincount(ci, minlength=len(cells))
    weak = np.zeros(len(cells), dtype=bool)
    weak[ci[(n_per_cell[ci] == 1) & (frac < min_nucleus_fraction)]] = True
    cells = cells[(n_per_cell > 0) & ~weak]

    # 2. Grow cells over nuclei they mostly contain
    ci, ni, inter, frac = pairs()
    cells_per_nucleus = np.bincount(ni, minlength=len(nuclei))
    grow = (cells_per_nucleus[ni] == 1) & (frac > 0.5) & (frac < 1 - CONTAINED_TOL)
    for c in np.unique(ci[grow]):
        cells[c] = shapely.union_all(np.append(nuclei[ni[grow & (ci == c)]], cells[c]))
    cells = clean_polygons(cells)

    # 3. Split nuclei straddling several cells
    ci, ni, inter, frac = pairs()
    n_per_cell = np.bincount(ci, minlength=len(cells))
    cells_per_nucleus = np.bincount(ni, minlength=len(nuclei))
    multi = cells_per_nucleus[ni] >= 2
    threshold = np.where(cells_per_nucleus[ni] == 2, split_keep_fraction_2, split_keep_fraction_3plus)
    lone = n_per_cell[ci] == 1
    drop_cell = multi & lone & (frac < threshold)
    keep_piece = multi & ~((frac < 0.5) & ~lone) & ~drop_cell
    pieces = shapely.intersection(cells[ci[keep_piece]], nuclei[ni[keep_piece]])
    whole = np.ones(len(nuclei), dtype=bool)
    whole[ni[multi]] = False
    nuclei = clean_polygons(np.concatenate([nuclei[whole], pieces]), min_overlap_area)
    keep = np.ones(len(cells), dtype=bool)
    keep[ci[drop_cell]] = False
    cells = cells[keep]

    # 4. No duplicates or overlaps within each type
    cells = _resolve_overlaps(cells, min_overlap_area)
    nuclei = _resolve_overlaps(nuclei, min_overlap_area)

    # 5. Assign nuclei to cells
    ci, ni, inter, frac = pairs()
    partial = (frac > 0.001) & (frac < 0.999)
    absorb, cut = partial & (frac > 0.5), partial & (frac <= 0.5)
    original_cells = cells.copy()
    cells = _subtract_neighbours(cells, ci[cut], ni[cut], originals=nuclei)
    for c in np.unique(ci[absorb]):
        cells[c] = shapely.union_all(np.append(nuclei[ni[absorb & (ci == c)]], cells[c]))
    nuclei = _subtract_neighbours(nuclei, ni[cut], ci[cut], originals=original_cells)
    cells, nuclei = clean_polygons(cells), clean_polygons(nuclei, min_overlap_area)

    ci, ni, inter, frac = pairs()
    straddling = np.unique(ni[(frac > 0.001) & (frac < 0.999)])
    keep = np.ones(len(nuclei), dtype=bool)
    keep[straddling] = False
    nuclei = nuclei[keep]

    ci, ni, inter, frac = pairs()
    orphan = np.setdiff1d(np.arange(len(nuclei)), ni)
    cells = np.concatenate([cells[np.unique(ci)], nuclei[orphan]])

    ci, ni, inter, frac = pairs()
    order = np.lexsort((-inter, ni))
    first = order[np.r_[True, np.diff(ni[order]) != 0]] if len(order) else order
    nucleus_cell = np.full(len(nuclei), -1)
    nucleus_cell[ni[first]] = ci[first]

    has_cell = nucleus_cell >= 0
    nuclei, nucleus_cell = nuclei[has_cell], nucleus_cell[has_cell]
    used = np.unique(nucleus_cell)
    remap = np.full(len(cells), -1)
    remap[used] = np.arange(len(used))
    cells, nucleus_cell = cells[used], remap[nucleus_cell]

    logger.info("Reconciled %d cells / %d nuclei -> %d cells / %d nuclei",
                n_cells_in, n_nuclei_in, len(cells), len(nuclei))
    return cells, nuclei, nucleus_cell
