"""
Figures for docs/concepts/tiling_and_stitching.md.

Every geometric operation here is SMINT's own code -- ``generate_tiles`` and
``stitch_and_reconcile`` from :mod:`smint.segmentation.distributed_seg`, and
the merge / trim / remove / reconcile functions of
:mod:`smint.segmentation.stitching` -- run on synthetic polygons. Only the
"segmentation" of each tile is simulated: a tile reports every cell it can
see inside its expanded window, clipped to that window and slightly jittered,
the way two Cellpose runs on overlapping crops disagree by a pixel or two.

Run from the repository root (no Cellpose or GPU needed):

    PYTHONPATH=. python docs/assets/concepts/src/make_tiling_figures.py

Writes PNGs to docs/assets/concepts/ and prints the numbers quoted on the page.
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.patches import Patch, Rectangle
from shapely import affinity

from smint.segmentation.distributed_seg import generate_tiles, stitch_and_reconcile
from smint.segmentation.stitching import (
    merge_overlapping,
    polygons_from_outlines,
    reconcile_cells_and_nuclei,
    remove_conflicts,
    stitch_polygons,
    trim_collisions,
)

logging.basicConfig(level=logging.WARNING)

OUT = Path(__file__).resolve().parents[1]
CELL = "#2b6cb0"      # cells / ST
NUC = "#d53f8c"       # nuclei / moving data
ALT = "#dd8a2e"       # second tile / comparison
GREY = "#718096"
LIGHT = "#e2e8f0"
DPI = 130

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Synthetic tissue
# ---------------------------------------------------------------------------

def make_tissue(width, height, rng, n_try=4000, r_range=(11, 19)):
    """Non-overlapping elliptical cells, each with an off-centre nucleus."""
    cells, nuclei = [], []
    for _ in range(n_try):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        a, b = rng.uniform(*r_range), rng.uniform(*r_range)
        ang = rng.uniform(0, 180)
        cell = affinity.rotate(affinity.scale(shapely.Point(cx, cy).buffer(1, 24), a, b), ang)
        if any(cell.distance(c) < 1.5 for c in cells if c.centroid.distance(cell.centroid) < 50):
            continue
        cells.append(cell)
        off = rng.normal(0, 2.0, 2)
        nuc = affinity.rotate(affinity.scale(
            shapely.Point(cx + off[0], cy + off[1]).buffer(1, 18), 0.45 * a, 0.45 * b), ang)
        nuclei.append(nuc)
    return np.array(cells, dtype=object), np.array(nuclei, dtype=object)


def simulate_tile(objects, window, rng, jitter=0.8, min_area=25.0):
    """What one tile 'segments': every object it sees, clipped and jittered."""
    win = shapely.box(*window)
    out = []
    for obj in objects:
        if not obj.intersects(win):
            continue
        part = obj.intersection(win)
        if part.area < min_area:
            continue
        part = affinity.translate(part.buffer(rng.uniform(-jitter, jitter)),
                                  *rng.normal(0, jitter / 2, 2))
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        x, y = np.asarray(part.exterior.coords).T
        out.append({"x_coords": x, "y_coords": y})
    return out


def expanded(coords, overlap, shape):
    """The window segment_tile reads: core expanded by overlap, clipped to the image."""
    y0, x0, y1, x1 = coords
    h, w = shape
    return (max(0, x0 - overlap), max(0, y0 - overlap), min(w, x1 + overlap), min(h, y1 + overlap))


def draw(ax, geoms, color, lw=1.0, fill=0.0, ls="-", z=2):
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        for p in parts:
            x, y = p.exterior.xy
            if fill:
                ax.fill(x, y, color=color, alpha=fill, lw=0, zorder=z)
            ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=z + 0.1)


def iou_matches(found, truth):
    """Number of true objects recovered with IoU > 0.5 (one-to-one)."""
    tree = shapely.STRtree(found)
    hit = 0
    used = set()
    for t in truth:
        for j in tree.query(t, predicate="intersects"):
            if j in used:
                continue
            inter = found[j].intersection(t).area
            if inter / (found[j].union(t).area) > 0.5:
                used.add(j)
                hit += 1
                break
    return hit


# ---------------------------------------------------------------------------
# Figure 1: tile grid, overlap and the duplicates it creates
# ---------------------------------------------------------------------------

def figure_tiles(numbers):
    rng = np.random.default_rng(3)
    H, W, TILE, OV = 400, 600, 200, 40
    cells, nuclei = make_tissue(W, H, rng)
    tiles = generate_tiles((H, W), TILE)

    raw_cells, raw_nuclei, origin = [], [], []
    for tid, coords in tiles:
        win = expanded(coords, OV, (H, W))
        c = simulate_tile(cells, win, rng)
        n = simulate_tile(nuclei, win, rng)
        raw_cells += c
        raw_nuclei += n
        origin += [tid] * len(c)

    raw = polygons_from_outlines(raw_cells)
    merged = merge_overlapping(raw)
    trimmed = trim_collisions(merged)
    removed = remove_conflicts(trimmed)
    final_cells, final_nuclei, nucleus_cell = stitch_and_reconcile(raw_cells, raw_nuclei)

    numbers["tiles"] = {
        "image_px": [H, W], "tile_size": TILE, "tile_overlap": OV, "n_tiles": len(tiles),
        "true_cells": int(len(cells)), "true_nuclei": int(len(nuclei)),
        "raw_cell_outlines": int(len(raw)), "raw_nucleus_outlines": len(raw_nuclei),
        "after_merge": int(len(merged)), "after_trim": int(len(trimmed)),
        "after_remove": int(len(removed)),
        "final_cells": int(len(final_cells)), "final_nuclei": int(len(final_nuclei)),
        "cells_recovered_iou50": iou_matches(final_cells, cells),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # (a) the grid
    ax = axes[0]
    draw(ax, cells, LIGHT, lw=0.6, fill=0.0)
    for tid, (y0, x0, y1, x1) in tiles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=1.4, zorder=4))
        ax.text(x0 + 8, y0 + 18, f"tile {tid}", fontsize=8, zorder=5)
    tid, coords = tiles[1]
    ex0, ey0, ex1, ey1 = expanded(coords, OV, (H, W))
    ax.add_patch(Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=True, fc=CELL, alpha=0.10,
                           ec=CELL, lw=1.6, ls="--", zorder=3))
    seen = [c for c in cells if c.intersects(shapely.box(ex0, ey0, ex1, ey1))]
    draw(ax, seen, CELL, lw=0.8)
    ax.annotate("", xy=(ex1, 120), xytext=(400, 120),
                arrowprops=dict(arrowstyle="<->", color=CELL, lw=1.2), zorder=6)
    ax.text(420, 112, "overlap", color=CELL, fontsize=8, ha="center", va="bottom", zorder=6,
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_title("(a) Core tiles; tile 1 reads its core + overlap")
    ax.set_xlim(-5, W + 5)
    ax.set_ylim(H + 5, -5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    # (b) zoom on the boundary between tile 0 and tile 1: raw outlines per tile
    zoom = (150, 40, 250, 160)
    ax = axes[1]
    zb = shapely.box(*zoom)
    raw_by_tile = {}
    for o, t in zip(raw_cells, origin):
        if t in (0, 1):
            p = shapely.Polygon(np.column_stack([o["x_coords"], o["y_coords"]]))
            if p.intersects(zb):
                raw_by_tile.setdefault(t, []).append(p)
    draw(ax, raw_by_tile.get(0, []), CELL, lw=1.3)
    draw(ax, raw_by_tile.get(1, []), ALT, lw=1.3, ls="--")
    ax.axvline(200, color="black", lw=1.2)
    ax.axvspan(200 - OV, 200 + OV, color=GREY, alpha=0.10)
    ax.set_xlim(zoom[0], zoom[2])
    ax.set_ylim(zoom[3], zoom[1])
    ax.set_aspect("equal")
    ax.set_title("(b) Raw outlines near the boundary")
    ax.legend(handles=[Patch(fc="none", ec=CELL, label="from tile 0"),
                       Patch(fc="none", ec=ALT, ls="--", label="from tile 1"),
                       Patch(fc=GREY, alpha=0.2, label="overlap band")],
              loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_xticks([])
    ax.set_yticks([])

    # (c) after stitch_and_reconcile
    ax = axes[2]
    fc = [c for c in final_cells if c.intersects(zb)]
    fn = [n for n in final_nuclei if n.intersects(zb)]
    draw(ax, fc, CELL, lw=1.3, fill=0.12)
    draw(ax, fn, NUC, lw=1.0, fill=0.25)
    ax.axvline(200, color="black", lw=1.2)
    ax.set_xlim(zoom[0], zoom[2])
    ax.set_ylim(zoom[3], zoom[1])
    ax.set_aspect("equal")
    ax.set_title("(c) After stitch_and_reconcile")
    ax.legend(handles=[Patch(fc=CELL, alpha=0.3, ec=CELL, label="cell"),
                       Patch(fc=NUC, alpha=0.4, ec=NUC, label="nucleus")],
              loc="lower left", fontsize=8, framealpha=0.95)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(OUT / "tiling_overlap.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the three stitching phases on minimal pairs
# ---------------------------------------------------------------------------

def figure_phases(numbers):
    def disc(x, y, r):
        return shapely.Point(x, y).buffer(r, 48)

    cases = [
        ("Merge", "same cell seen by two tiles\nmax overlap > merge_threshold (0.5)",
         [disc(0, 0, 20), disc(4, 1, 19)], merge_overlapping),
        ("Trim", "neighbours touching\nmax overlap < trim_threshold (0.25)",
         [disc(-14, 0, 18), disc(14, 0, 18)], trim_collisions),
        ("Remove", "real conflict\n0.25 <= max overlap <= 0.5",
         [disc(-6, 0, 22), disc(18, 0, 10)], remove_conflicts),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    numbers["phases"] = {}
    for col, (name, why, polys, fn) in enumerate(cases):
        geoms = np.array(polys, dtype=object)
        a, b = geoms
        inter = a.intersection(b).area
        overlap = max(inter / a.area, inter / b.area)
        result = fn(geoms)
        assert len(stitch_polygons(geoms)) == len(result), name
        numbers["phases"][name] = {"max_overlap": round(float(overlap), 2),
                                   "n_in": 2, "n_out": int(len(result))}
        ax = axes[0, col]
        draw(ax, [a], CELL, lw=1.6, fill=0.15)
        draw(ax, [b], ALT, lw=1.6, fill=0.15, ls="--")
        ax.set_title(f"{name}: before", loc="center")
        ax.text(0.5, -0.08, f"{why}\nmeasured max overlap = {overlap:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#2d3748")
        ax = axes[1, col]
        colours = [CELL, ALT, NUC]
        for k, g in enumerate(result):
            draw(ax, [g], colours[k % 3], lw=1.6, fill=0.15)
        ax.set_title(f"after {fn.__name__}()", loc="center", fontsize=10, fontweight="normal",
                     family="monospace")
        ax.text(0.5, -0.08, f"{len(result)} polygon{'s' if len(result) != 1 else ''}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        for r in (0, 1):
            axes[r, col].set_xlim(-42, 42)
            axes[r, col].set_ylim(-28, 28)
            axes[r, col].set_aspect("equal")
            axes[r, col].axis("off")
    fig.tight_layout(h_pad=3.0)
    fig.savefig(OUT / "stitching_phases.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: cell / nucleus reconciliation
# ---------------------------------------------------------------------------

def figure_reconcile(numbers):
    def disc(x, y, r):
        return shapely.Point(x, y).buffer(r, 48)

    cells = [
        disc(0, 0, 20),        # A: normal cell
        disc(60, 0, 20),       # B: no nucleus
        disc(120, 0, 20),      # C: nucleus sticking out
        disc(178, 0, 18),      # D1: its only nucleus is shared with D2
        disc(214, 0, 18),      # D2: has its own nucleus too
    ]
    nuclei = [
        disc(2, 1, 8),         # A
        disc(132, 6, 10),      # C: mostly inside, partly outside
        disc(220, 4, 6),       # D2 own nucleus
        disc(191, 2, 8),       # straddles D1 / D2, mostly in D1
        disc(260, 0, 8),       # E: nucleus with no cell
    ]
    cells_out, nuclei_out, nucleus_cell = reconcile_cells_and_nuclei(
        np.array(cells, dtype=object), np.array(nuclei, dtype=object))
    numbers["reconcile"] = {"cells_in": len(cells), "nuclei_in": len(nuclei),
                            "cells_out": int(len(cells_out)), "nuclei_out": int(len(nuclei_out)),
                            "nucleus_cell": nucleus_cell.tolist()}

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 4.6))
    for ax, (cs, ns, title) in zip(axes, [(cells, nuclei, "Stitched cells and nuclei (input)"),
                                          (cells_out, nuclei_out,
                                           "After reconcile_cells_and_nuclei(): every nucleus in exactly one cell")]):
        draw(ax, cs, CELL, lw=1.5, fill=0.12)
        draw(ax, ns, NUC, lw=1.2, fill=0.35)
        ax.set_title(title, loc="left")
        ax.set_xlim(-25, 285)
        ax.set_ylim(28, -26)
        ax.set_aspect("equal")
        ax.axis("off")
    labels = [(0, "kept"), (60, "no nucleus:\ndropped"), (120, "nucleus mostly inside:\ncell grows over it"),
              (196, "nucleus straddling two cells:\nsplit, each piece kept only\nwhere it is most of the nucleus"), (260, "nucleus without a cell:\nnucleus-only cell")]
    for x, text in labels:
        axes[0].text(x, 34, text, ha="center", va="top", fontsize=8.5, color="#2d3748",
                     transform=axes[0].transData, clip_on=False)
    fig.tight_layout(h_pad=3.2)
    fig.savefig(OUT / "reconcile_cells_nuclei.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: why tile_overlap must exceed the largest cell radius
# ---------------------------------------------------------------------------

def figure_overlap_choice(numbers):
    H, W, TILE = 200, 400, 200
    radius = 60
    OVERLAPS = (3, 10, 80)
    cell = affinity.scale(shapely.Point(200, 100).buffer(1, 64), radius, 0.8 * radius)
    neighbours = [shapely.Point(x, y).buffer(16, 32) for x, y in
                  [(60, 50), (90, 150), (330, 45), (345, 160), (105, 100), (300, 105)]]
    truth = np.array([cell] + neighbours, dtype=object)
    tiles = generate_tiles((H, W), TILE)
    rng = np.random.default_rng(1)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    numbers["overlap_choice"] = {"cell_radius_px": radius}
    for ax, ov in zip(axes, OVERLAPS):
        raw = []
        for _, coords in tiles:
            raw += simulate_tile(truth, expanded(coords, ov, (H, W)), rng, jitter=0.3)
        out = stitch_polygons(polygons_from_outlines(raw))
        big = max(out, key=lambda p: p.intersection(cell).area)
        recovered = big.intersection(cell).area / cell.area
        n_frag = sum(1 for p in out if p.intersection(cell).area > 0.05 * p.area)
        numbers["overlap_choice"][f"overlap_{ov}"] = {
            "polygons_on_big_cell": n_frag, "big_cell_area_recovered": round(float(recovered), 2)}
        draw(ax, [cell], GREY, lw=1.0, ls=":")
        cols = [CELL, ALT, NUC]
        k = 0
        for p in out:
            if p.intersection(cell).area > 0.05 * p.area:
                draw(ax, [p], cols[k % 3], lw=1.6, fill=0.18)
                k += 1
            else:
                draw(ax, [p], CELL, lw=0.8, fill=0.05)
        ax.axvline(200, color="black", lw=1.2)
        ax.axvspan(200 - ov, 200 + ov, color=GREY, alpha=0.10)
        verdict = (f"split in {n_frag}" if n_frag > 1 else
                   "truncated" if recovered < 0.95 else "complete")
        ax.set_title(f"tile_overlap = {ov} px  ({'<' if ov < radius else '>'} radius {radius})")
        ax.text(0.5, -0.06, f"large cell: {n_frag} polygon{'s' if n_frag != 1 else ''}, "
                f"{100 * recovered:.0f}% of its area -> {verdict}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "tile_overlap_choice.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    numbers = {}
    figure_tiles(numbers)
    figure_phases(numbers)
    figure_reconcile(numbers)
    figure_overlap_choice(numbers)
    print(json.dumps(numbers, indent=2))
