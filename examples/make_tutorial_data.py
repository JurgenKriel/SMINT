#!/usr/bin/env python
"""
Generate the small synthetic datasets used by the SMINT tutorials.

Everything here is made up, deterministic (fixed seeds) and small, so the
tutorials run anywhere and every result can be scored against a known truth:

``synthetic_wsi.ome.tif`` + ``synthetic_wsi_truth.csv``
    A 1024 x 1024 px, 2-channel (C, Y, X) uint8 image that looks like a
    fluorescence tile scan: channel 0 is a nuclear stain, channel 1 a
    cytoplasm / membrane stain. The truth table lists every cell and nucleus.

``synthetic_cells.csv`` / ``synthetic_nuclei.csv`` + ``synthetic_same_section_truth.csv``
    A same-section pair, as after imaging a Xenium slide and then
    post-staining it. Target "Xenium cells" (``x_centroid``/``y_centroid``) and
    source "nuclei" (``centroid_x``/``centroid_y``) moved by a known offset,
    rotation, scale and smooth warp. About 10% of cells have no nucleus and
    there are 25% extra nuclei with no cell, as in real data.

``synthetic_st.csv`` / ``synthetic_sm.csv`` + landmarks + ``synthetic_sm_truth.csv``
    A sequential-section pair: spatial transcriptomics cells (``x_final``/
    ``y_final``, microns) and a spatial metabolomics pixel grid (``x``/``y`` as
    raw 10 um pixel indices, read with ``scale_xy=10, rotate_left_90=True`` as
    for the real MALDI data) of the same tissue shape, with a further known
    8 degree rotation, 8% scale and smooth warp. ``synthetic_st_points.npy`` /
    ``synthetic_sm_points.npy`` hold 8 landmark pairs in the format STalign's
    ``point_annotator.py`` writes (a ``{label: [(x, y)]}`` dict), in the
    frames :func:`smint.alignment.register_sm_to_st` reads them in.

Usage::

    python examples/make_tutorial_data.py                     # -> docs/tutorials/data/
    python examples/make_tutorial_data.py --out my_data/

Needs numpy, scipy, pandas and tifffile only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spaced_points(rng, n, lo, hi, min_dist, accept=None, max_tries=200_000):
    """Up to ``n`` random points in the box [lo, hi) at least ``min_dist`` apart.

    ``accept(points) -> bool array`` optionally thins candidates (tissue
    shape, density). Dart throwing in batches; good enough at these sizes.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    points = np.empty((0, 2))
    tries = 0
    while len(points) < n and tries < max_tries:
        batch = rng.uniform(lo, hi, (4096, 2))
        tries += len(batch)
        if accept is not None:
            batch = batch[accept(batch)]
        for p in batch:
            if len(points) and np.min(np.sum((points - p) ** 2, axis=1)) < min_dist ** 2:
                continue
            points = np.vstack([points, p])
            if len(points) >= n:
                break
    return points


def smooth_warp(points, amplitude, wavelength, phase=(0.3, 1.1)):
    """A smooth, invertible-looking displacement field (sum of sinusoids)."""
    x, y = points[:, 0], points[:, 1]
    k = 2 * np.pi / wavelength
    dx = amplitude * np.sin(k * y + phase[0]) * np.cos(0.7 * k * x)
    dy = amplitude * np.cos(k * x + phase[1]) * np.sin(0.6 * k * y + 0.5)
    return np.column_stack([dx, dy])


def rotation(degrees):
    t = np.deg2rad(degrees)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def save_landmarks(path, points_xy):
    """Write landmarks the way STalign's point_annotator.py does."""
    payload = {str(i + 1): [(float(x), float(y))] for i, (x, y) in enumerate(points_xy)}
    np.save(path, payload, allow_pickle=True)


# ---------------------------------------------------------------------------
# 1. Whole-slide-like image
# ---------------------------------------------------------------------------

def make_image(out, size=1024, seed=1):
    import tifffile

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)

    # Two empty "vessels" so the tissue is not a uniform carpet
    holes = [(300, 700, 90), (780, 260, 60)]

    def in_tissue(p):
        ok = np.ones(len(p), bool)
        for hx, hy, r in holes:
            ok &= np.hypot(p[:, 0] - hx, p[:, 1] - hy) > r + 20
        return ok

    seeds = spaced_points(rng, 2000, [8, 8], [size - 8, size - 8], 30.0, in_tissue)
    n = len(seeds)
    cell_radius = rng.uniform(15, 21, n)

    # Cells: nearest seed within its radius, with a 2 px gap between neighbours
    tree = cKDTree(seeds)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    d, j = tree.query(grid, k=2)
    own = j[:, 0]
    inside = (d[:, 0] <= cell_radius[own]) & (d[:, 1] - d[:, 0] > 2.0)
    cells = np.where(inside, own + 1, 0).reshape(size, size)

    # Nuclei: an ellipse near each seed, kept inside its cell
    nuclei = np.zeros_like(cells)
    nuc_centres = np.zeros((n, 2))
    for i, (sx, sy) in enumerate(seeds):
        a, b = rng.uniform(6.5, 9.0), rng.uniform(5.0, 7.0)
        theta = rng.uniform(0, np.pi)
        cx, cy = sx + rng.normal(0, 1.5), sy + rng.normal(0, 1.5)
        x0, x1 = int(max(cx - 12, 0)), int(min(cx + 13, size))
        y0, y1 = int(max(cy - 12, 0)), int(min(cy + 13, size))
        lx, ly = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
        u = lx * np.cos(theta) + ly * np.sin(theta)
        v = -lx * np.sin(theta) + ly * np.cos(theta)
        mask = ((u / a) ** 2 + (v / b) ** 2 <= 1) & (cells[y0:y1, x0:x1] == i + 1)
        nuclei[y0:y1, x0:x1][mask] = i + 1
        nuc_centres[i] = cx, cy

    # Channel 0: nuclear stain with chromatin texture
    texture = ndimage.gaussian_filter(rng.normal(0, 1, (size, size)), 1.2)
    level = np.concatenate([[0], rng.uniform(150, 230, n)])
    nuc_img = level[nuclei] * (1 + 0.15 * texture) * (nuclei > 0)
    nuc_img = ndimage.gaussian_filter(nuc_img, 1.0) + 12

    # Channel 1: cytoplasm, brighter at the membrane, dimmer over the nucleus
    level = np.concatenate([[0], rng.uniform(55, 110, n)])
    body = level[cells] * (cells > 0)
    edge = ndimage.binary_dilation(cells > 0) & ~ndimage.binary_erosion(cells > 0, iterations=2)
    edge |= ndimage.morphological_gradient(cells, size=3) > 0
    cyto = body * (1 - 0.45 * (nuclei > 0)) + 70 * (edge & (cells > 0))
    cyto = ndimage.gaussian_filter(cyto * (1 + 0.1 * texture), 1.5) + 15

    image = np.stack([nuc_img, cyto])
    image = image + rng.normal(0, 4, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    tifffile.imwrite(out / "synthetic_wsi.ome.tif", image, photometric="minisblack",
                     compression="zlib", metadata={"axes": "CYX"})

    labels = np.arange(1, n + 1)
    truth = pd.DataFrame({
        "cell_id": labels,
        "x": ndimage.mean(xx, cells, labels), "y": ndimage.mean(yy, cells, labels),
        "cell_area_px": ndimage.sum(np.ones_like(xx), cells, labels).astype(int),
        "nucleus_x": nuc_centres[:, 0], "nucleus_y": nuc_centres[:, 1],
        "nucleus_area_px": ndimage.sum(np.ones_like(xx), nuclei, labels).astype(int),
    })
    truth = truth[truth.cell_area_px > 0]
    truth.round(2).to_csv(out / "synthetic_wsi_truth.csv", index=False)
    return len(truth)


# ---------------------------------------------------------------------------
# 2. Same-section centroid pair
# ---------------------------------------------------------------------------

def make_same_section(out, side=2000.0, seed=2):
    rng = np.random.default_rng(seed)

    # Target: "Xenium cells", ~1000 per mm^2 and >= 8 um apart, like real tissue
    n_cells = int(side * side / 1e6 * 1000)
    cells = spaced_points(rng, n_cells, [0, 0], [side, side], 8.0)

    # Each cell has a nucleus a little off-centre; 10% have none
    has_nucleus = rng.random(len(cells)) > 0.10
    true_nuc = cells[has_nucleus] + rng.normal(0, 2.0, (has_nucleus.sum(), 2))
    true_cell = np.flatnonzero(has_nucleus)

    # Extra nuclei with no Xenium cell (25% as many as cells)
    extra = rng.uniform([0, 0], [side, side], (int(0.25 * len(cells)), 2))
    true_nuc = np.vstack([true_nuc, extra])
    true_cell = np.concatenate([true_cell, np.full(len(extra), -1)])

    # Known misalignment of the post-stain frame: ~17 um shift (about one cell
    # spacing), 0.5 degree rotation, 0.6% scale and a smooth +-3 um warp
    centre = np.array([side / 2, side / 2])
    moved = (true_nuc - centre) @ (1.006 * rotation(0.5)).T + centre + [14.0, -9.5]
    moved += smooth_warp(true_nuc, 3.0, 900.0)

    order = rng.permutation(len(moved))
    moved, true_nuc, true_cell = moved[order], true_nuc[order], true_cell[order]

    pd.DataFrame({"x_centroid": cells[:, 0], "y_centroid": cells[:, 1]}).round(3) \
        .to_csv(out / "synthetic_cells.csv", index=False)
    pd.DataFrame({"centroid_x": moved[:, 0], "centroid_y": moved[:, 1]}).round(3) \
        .to_csv(out / "synthetic_nuclei.csv", index=False)
    pd.DataFrame({"x_true": true_nuc[:, 0], "y_true": true_nuc[:, 1], "true_cell": true_cell}) \
        .round(3).to_csv(out / "synthetic_same_section_truth.csv", index=False)
    return len(cells), len(moved)


# ---------------------------------------------------------------------------
# 3. Sequential-section ST / SM pair
# ---------------------------------------------------------------------------

def tissue_mask(p):
    """An asymmetric section: two lobes, a notch and a ventricle-like hole."""
    x, y = p[:, 0], p[:, 1]
    lobe1 = ((x - 1700) / 1500) ** 2 + ((y - 1500) / 1100) ** 2 <= 1
    lobe2 = ((x - 3000) / 900) ** 2 + ((y - 1900) / 800) ** 2 <= 1
    hole = ((x - 1500) / 380) ** 2 + ((y - 1350) / 200) ** 2 <= 1
    notch = (np.abs(x - 2300) < 120) & (y < 900)
    return (lobe1 | lobe2) & ~hole & ~notch


def st_density(p):
    """Denser band of cells, like a cortical layer, so the raster has texture."""
    x, y = p[:, 0], p[:, 1]
    band = np.exp(-((np.hypot(x - 1700, y - 1500) - 800) / 180) ** 2)
    return 0.35 + 0.65 * band


def make_sequential(out, seed=3):
    rng = np.random.default_rng(seed)

    # ST cells in the truth frame (microns)
    st = spaced_points(rng, 9000, [0, 0], [4000, 3000], 9.0,
                       accept=lambda p: tissue_mask(p) & (rng.random(len(p)) < st_density(p)))

    # SM pixels: a regular 10 um grid in the frame registration works in (what
    # read_sm_matrix(scale_xy=10, rotate_left_90=True) returns), mapped to the
    # truth frame by a known residual rotation, scale and smooth warp
    angle, scale = 8.0, 1.08
    sm_centre = np.array([2000.0, 1500.0])
    truth_centre = np.array([2100.0, 1550.0])

    def sm_to_truth(q):
        return (q - sm_centre) @ (scale * rotation(angle)).T + truth_centre + smooth_warp(q, 45.0, 2500.0)

    gx, gy = np.meshgrid(np.arange(0, 450), np.arange(0, 350))
    q = 10.0 * np.column_stack([gx.ravel(), gy.ravel()])
    truth_xy = sm_to_truth(q)
    keep = tissue_mask(truth_xy)
    q, truth_xy = q[keep], truth_xy[keep]

    # Undo what read_sm_matrix does, so the file holds raw pixel indices like a
    # MALDI export: it maps raw (x, y) to (-10 y, 10 x), then shifts x to start
    # at 0. Raw x = q_y / 10 and raw y = (max q_x - q_x) / 10 invert that, and
    # the frame it rebuilds is q shifted by min q_x along x.
    qx_min, qx_max = q[:, 0].min(), q[:, 0].max()
    raw = np.column_stack([q[:, 1] / 10, (qx_max - q[:, 0]) / 10]).round().astype(int)
    frame_shift = np.array([qx_min, 0.0])

    # Landmarks: recognisable points of the shape, picked in both frames with
    # a little clicking noise; SM positions found by inverting sm_to_truth
    from scipy.optimize import least_squares

    features = np.array([
        [1500, 1150], [1880, 1350],              # ends of the hole
        [2180, 920], [2420, 920],                # corners of the notch
        [3880, 1900], [2440, 2480],              # far tip of lobe 2, upper junction
        [220, 1500], [1700, 2580],               # far tip of lobe 1, top of lobe 1
    ], dtype=float)
    sm_points = np.array([
        least_squares(lambda q, f=f: sm_to_truth(q[None])[0] - f, x0=sm_centre).x
        for f in features
    ])
    st_points = features + rng.normal(0, 15.0, features.shape)
    sm_points = sm_points - frame_shift + rng.normal(0, 15.0, sm_points.shape)

    pd.DataFrame({"x_final": st[:, 0], "y_final": st[:, 1]}).round(2) \
        .to_csv(out / "synthetic_st.csv", index=False)
    pd.DataFrame({"x": raw[:, 0], "y": raw[:, 1]}).to_csv(out / "synthetic_sm.csv", index=False)
    pd.DataFrame({"x_true": truth_xy[:, 0], "y_true": truth_xy[:, 1]}).round(2) \
        .to_csv(out / "synthetic_sm_truth.csv", index=False)
    save_landmarks(out / "synthetic_st_points.npy", st_points)
    save_landmarks(out / "synthetic_sm_points.npy", sm_points)
    return len(st), len(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    n = make_image(args.out)
    print(f"synthetic_wsi.ome.tif      1024 x 1024 px, 2 channels, {n} cells")
    n_cells, n_nuclei = make_same_section(args.out)
    print(f"synthetic_cells/nuclei.csv {n_cells} cells, {n_nuclei} nuclei")
    n_st, n_sm = make_sequential(args.out)
    print(f"synthetic_st/sm.csv        {n_st} ST cells, {n_sm} SM pixels, 8 landmark pairs")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
