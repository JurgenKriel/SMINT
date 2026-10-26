import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from smint.segmentation.stitching import (
    filter_by_nuclei,
    merge_overlapping,
    reconcile_cells_and_nuclei,
    remove_conflicts,
    stitch_polygons,
    trim_collisions,
)


def disk(x, y, r):
    return Point(x, y).buffer(r, quad_segs=32)


def arr(*geoms):
    return np.array(geoms, dtype=object)


def total_overlap(geoms):
    tree = shapely.STRtree(geoms)
    i, j = tree.query(geoms, predicate="intersects")
    keep = i < j
    return float(shapely.area(shapely.intersection(geoms[i[keep]], geoms[j[keep]])).sum())


# --- Stage 1 -----------------------------------------------------------------

def test_tile_duplicates_merge_into_one_cell():
    full = disk(100, 100, 30)
    left_half = full.intersection(box(0, 0, 105, 200))  # seen by the neighbouring tile
    out = stitch_polygons(arr(full, left_half))
    assert len(out) == 1
    assert out[0].area == pytest.approx(full.area, rel=1e-3)


def test_overlapping_fragments_of_one_cell_do_not_delete_it():
    # A full cell plus two edge slivers from neighbouring tiles that overlap
    # each other by 25-50%: deciding removals before merging would delete the
    # merged cell along with the smaller sliver.
    full = disk(100, 100, 30)
    top = full.intersection(box(60, 60, 140, 85))
    left = full.intersection(box(60, 60, 82, 140))
    assert len(stitch_polygons(arr(full, top, left))) == 1


def test_merge_chains_until_stable():
    a, b, c = box(0, 0, 10, 10), box(4, 0, 14, 10), box(8, 0, 18, 10)
    assert len(merge_overlapping(arr(a, b, c))) == 1


def test_small_collision_is_trimmed_from_both():
    a, b = box(0, 0, 10, 10), box(9, 0, 19, 10)  # 10% overlap
    out = trim_collisions(arr(a, b))
    assert len(out) == 2
    assert total_overlap(out) == 0
    assert all(g.area == pytest.approx(90) for g in out)


def test_trimmed_by_several_neighbours_loses_every_overlap():
    centre = box(0, 0, 20, 20)
    out = trim_collisions(arr(centre, box(18, 0, 38, 20), box(0, 18, 20, 38)))
    assert total_overlap(out) == 0
    assert out[0].area == pytest.approx(400 - 40 - 40 + 4)


def test_intermediate_overlap_removes_smaller():
    big, small = box(0, 0, 20, 20), box(14, 0, 24, 10)  # 40% of small
    out = remove_conflicts(arr(big, small))
    assert len(out) == 1 and out[0].equals(big)


def test_touching_edges_are_not_overlaps():
    out = stitch_polygons(arr(box(0, 0, 10, 10), box(10, 0, 20, 10)))
    assert len(out) == 2


def test_cells_without_nuclei_are_filtered():
    cells = arr(disk(0, 0, 10), disk(100, 0, 10))
    nuclei = arr(disk(0, 0, 3))
    assert len(filter_by_nuclei(cells, nuclei)) == 1


# --- Stage 2 -----------------------------------------------------------------

def test_reconcile_assigns_each_nucleus_to_its_cell():
    cells = arr(disk(0, 0, 20), disk(100, 0, 20))
    nuclei = arr(disk(100, 0, 6), disk(0, 0, 6))
    c, n, match = reconcile_cells_and_nuclei(cells, nuclei)
    assert len(c) == 2 and len(n) == 2
    for k in range(2):
        assert c[match[k]].contains(n[k])


def test_reconcile_grows_cell_over_mostly_contained_nucleus():
    cell = box(0, 0, 20, 20)
    nucleus = box(14, 5, 24, 15)  # 60% inside
    c, n, match = reconcile_cells_and_nuclei(arr(cell), arr(nucleus))
    assert len(c) == 1 and c[0].buffer(1e-9).contains(n[0])
    assert c[0].area == pytest.approx(400 + 40)


def test_reconcile_cuts_mostly_outside_nucleus_out_of_cell():
    cell = box(0, 0, 20, 20)
    inner = box(5, 5, 10, 10)
    edge = box(17, 5, 27, 15)  # 30% inside: cut out of the cell, cell out of it
    c, n, match = reconcile_cells_and_nuclei(arr(cell), arr(inner, edge))
    # The cell loses the overlap; the part of the nucleus outside it has no
    # cell left, so it becomes a nucleus-only cell of its own.
    assert len(c) == 2 and total_overlap(c) == 0
    assert any(g.equals(inner) for g in n)
    for k in range(len(n)):
        assert c[match[k]].buffer(1e-9).contains(n[k])
    assert c[match[1 - [g.equals(inner) for g in n].index(True)]].equals(box(20, 5, 27, 15))


def test_reconcile_splits_nucleus_between_two_cells():
    cells = arr(box(0, 0, 20, 20), box(20, 0, 40, 20))
    nuclei = arr(box(15, 5, 25, 15), box(2, 2, 6, 6), box(34, 2, 38, 6))
    c, n, match = reconcile_cells_and_nuclei(cells, nuclei)
    assert len(c) == 2
    assert len(n) == 4  # two whole nuclei + the shared one split in half
    for k in range(len(n)):
        assert c[match[k]].buffer(1e-9).contains(n[k])


def test_reconcile_orphan_nucleus_becomes_a_cell():
    c, n, match = reconcile_cells_and_nuclei(arr(disk(0, 0, 20)), arr(disk(0, 0, 5), disk(200, 0, 5)))
    assert len(c) == 2 and len(n) == 2
    assert c[match[1]].equals(n[1])


def test_reconcile_drops_duplicate_and_contained_cells():
    cell = disk(0, 0, 20)
    cells = arr(cell, cell, disk(0, 0, 10))
    c, n, match = reconcile_cells_and_nuclei(cells, arr(disk(0, 0, 4)))
    assert len(c) == 1 and c[0].area == pytest.approx(cell.area, rel=1e-6)


def test_reconcile_output_has_no_overlaps():
    rng = np.random.default_rng(1)
    centres = rng.uniform(0, 300, size=(60, 2))
    cells = arr(*[disk(x, y, rng.uniform(12, 20)) for x, y in centres])
    nuclei = arr(*[disk(x + rng.normal(0, 3), y + rng.normal(0, 3), 5) for x, y in centres])
    c, n, match = reconcile_cells_and_nuclei(stitch_polygons(cells), stitch_polygons(nuclei))
    assert total_overlap(c) < 1.0
    assert total_overlap(n) < 1.0
    assert (match >= 0).all()
    covered = shapely.area(shapely.intersection(c[match], n)) / shapely.area(n)
    assert (covered > 0.99).all()
