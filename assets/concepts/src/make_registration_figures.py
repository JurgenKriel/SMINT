"""
Figures for docs/concepts/registration_methods.md and coordinate_frames.md
that need no STalign.

* pretransform_order.png -- flip -> rotate -> scale -> translate, built from
  SMINT's own matrix helpers and checked against ``build_pretransform``.
* registration_metrics.png -- why fit TRE is circular and why a snapped
  residual of zero proves nothing, on a synthetic same-section pair where the
  true nucleus -> cell partner of every point is known.

Run from the repository root:

    PYTHONPATH=. python docs/assets/concepts/src/make_registration_figures.py

Writes PNGs to docs/assets/concepts/ and prints the numbers quoted on the pages.
"""

import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from smint.alignment import (
    apply_pretransform,
    build_pretransform,
    compose,
    describe_pretransform,
    fit_scale_to_reference,
    flip_matrix,
    overlap_score,
    refine_by_assignment,
    register_centroids,
    rotation_matrix,
    scale_matrix,
    translation_matrix,
)

logging.basicConfig(level=logging.WARNING)

OUT = Path(__file__).resolve().parents[1]
CELL = "#2b6cb0"
NUC = "#d53f8c"
ALT = "#dd8a2e"
GREY = "#a0aec0"
DPI = 130

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic tissue: an asymmetric "F", so mirror images and transposes show
# ---------------------------------------------------------------------------

def in_tissue(p):
    x, y = p[:, 0], p[:, 1]
    return (((x > 0) & (x < 600) & (y > 0) & (y < 2400))
            | ((x > 0) & (x < 1800) & (y > 1800) & (y < 2400))
            | ((x > 0) & (x < 1200) & (y > 900) & (y < 1400)))


def tissue_points(n, rng):
    p = rng.uniform([0, 0], [1800, 2400], (n * 3, 2))
    return p[in_tissue(p)][:n]


# ---------------------------------------------------------------------------
# Figure: pre-transform composition order
# ---------------------------------------------------------------------------

def figure_pretransform(numbers):
    rng = np.random.default_rng(0)
    st = tissue_points(6000, rng)                       # reference, microns
    # A source that is mirrored, turned and on a 10x smaller grid
    truth = compose(flip_matrix(flip_x=True, center=st.mean(0)),
                    rotation_matrix(30, center=st.mean(0)))
    inverse = np.linalg.inv(truth)
    sm = apply_pretransform(st, inverse) / 10.0 + [40, 25]

    pivot = sm.mean(0)
    steps = [("source", np.eye(3))]
    flip = flip_matrix(flip_x=True, center=pivot)
    rot = rotation_matrix(30, center=pivot)
    oriented = compose(flip, rot)
    sx, sy = fit_scale_to_reference(apply_pretransform(sm, oriented), st)
    scale = scale_matrix(sx, sy, center=pivot)
    moved = apply_pretransform(sm, compose(flip, rot, scale))
    shift = translation_matrix(*(st.mean(0) - moved.mean(0)))
    steps += [("1. flip_x", flip), ("2. rotate 30°", compose(flip, rot)),
              (f"3. scale ×{sx:.2f}", compose(flip, rot, scale)),
              ("4. translate", compose(flip, rot, scale, shift))]

    built = build_pretransform(sm, st, rotation=30, flip_x=True)
    assert np.allclose(built, steps[-1][1]), "manual composition differs from build_pretransform"
    rot_then_flip = compose(rotation_matrix(30, center=pivot), flip_matrix(flip_x=True, center=pivot))
    rtf_scale = fit_scale_to_reference(apply_pretransform(sm, rot_then_flip), st)
    rtf = compose(rot_then_flip, scale_matrix(*rtf_scale, center=pivot))
    rtf = compose(rtf, translation_matrix(*(st.mean(0) - apply_pretransform(sm, rtf).mean(0))))

    final = apply_pretransform(sm, built)
    numbers["pretransform"] = {
        "fitted_scale": round(sx, 3),
        "overlap_flip_then_rotate": round(overlap_score(final, st), 3),
        "overlap_rotate_then_flip": round(overlap_score(apply_pretransform(sm, rtf), st), 3),
        "overlap_source_raw": round(overlap_score(sm, st), 3),
        "described": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in describe_pretransform(built).items() if k != "translation"},
    }

    fig, axes = plt.subplots(1, 6, figsize=(14.6, 3.3))
    for ax, (label, m) in zip(axes[:5], steps):
        pts = apply_pretransform(sm, m)
        if label == "4. translate":
            ax.scatter(st[:, 0], st[:, 1], s=1, c=CELL, alpha=0.35, lw=0, label="reference (ST)")
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c=NUC, alpha=0.5, lw=0, label="source (SM)")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    axes[0].set_title("source (SM pixels)")
    ax = axes[5]
    ax.scatter(st[:, 0], st[:, 1], s=1, c=CELL, alpha=0.35, lw=0)
    p = apply_pretransform(sm, rtf)
    ax.scatter(p[:, 0], p[:, 1], s=1, c=ALT, alpha=0.5, lw=0)
    ax.set_title("rotate → flip instead")
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    axes[4].text(0.5, -0.2, f"overlap_score {numbers['pretransform']['overlap_flip_then_rotate']:.2f}",
                 transform=axes[4].transAxes, ha="center", fontsize=9)
    axes[5].text(0.5, -0.2, f"overlap_score {numbers['pretransform']['overlap_rotate_then_flip']:.2f}",
                 transform=axes[5].transAxes, ha="center", fontsize=9)
    for ax in axes[1:]:
        ax.annotate("", xy=(-0.12, 0.5), xytext=(-0.3, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#4a5568"))
    save(fig, "pretransform_order.png")


# ---------------------------------------------------------------------------
# Figure: validation metrics and their traps (same-section, synthetic)
# ---------------------------------------------------------------------------

def poisson_disc(width, height, min_dist, rng, n_candidates=60000):
    """Greedy dart throwing: cells at least ``min_dist`` apart."""
    cand = rng.uniform([0, 0], [width, height], (n_candidates, 2))
    kept = []
    grid = {}
    cell = min_dist
    for p in cand:
        key = (int(p[0] // cell), int(p[1] // cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in grid.get((key[0] + dx, key[1] + dy), ()):
                    if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_dist ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            kept.append(p)
            grid.setdefault(key, []).append(p)
    return np.array(kept)


def figure_metrics(numbers):
    rng = np.random.default_rng(7)
    cells = poisson_disc(1500, 1500, 13.0, rng)
    spacing = float(np.median(cKDTree(cells).query(cells, k=2)[0][:, 1]))

    has_nucleus = rng.random(len(cells)) < 0.9
    partner = np.flatnonzero(has_nucleus)
    nuclei_true = cells[partner] + rng.normal(0, 2.0, (len(partner), 2))    # off-centre nuclei
    extra = rng.uniform([0, 0], [1500, 1500], (int(0.05 * len(cells)), 2))   # no Xenium cell
    nuclei_true = np.vstack([nuclei_true, extra])
    partner = np.concatenate([partner, np.full(len(extra), -1)])

    # The post-stain scan sits in its own frame: small rotation, scale, shift and a gentle warp
    c = nuclei_true.mean(0)
    m = compose(rotation_matrix(0.6, center=c), scale_matrix(1.01, center=c), translation_matrix(14, -6))
    nuclei = apply_pretransform(nuclei_true, m)
    nuclei = nuclei + np.c_[3 * np.sin(nuclei[:, 1] / 300), 3 * np.cos(nuclei[:, 0] / 260)]

    has = partner >= 0

    def true_error(moved):
        return float(np.median(np.linalg.norm(moved[has] - cells[partner[has]], axis=1)))

    runs = {}
    for label, kw in [("ransac+tps", dict(method="ransac+tps")),
                      ("tps, no regularisation", dict(method="tps", tps_regularization=0.0, n_control=None)),
                      ("multiround", dict(method="multiround"))]:
        t = time.time()
        r = register_centroids(nuclei, cells, **kw)
        moved = r["transform"](nuclei)
        runs[label] = {
            "tre_initial": round(r["tre_initial"][0], 2),
            "tre_fit": round(r["tre_fit"][0], 2),
            "tre_validation": round(r["tre_validation"][0], 2),
            "true_median_error": round(true_error(moved), 2),
            "seconds": round(time.time() - t, 1),
            "_moved": moved,
        }
    before = true_error(nuclei)

    ref = refine_by_assignment(runs["multiround"]["_moved"], cells)
    status = ref["status"]
    snapped = (status == "snapped")
    assigned = (status == "assigned")
    resid = np.linalg.norm(ref["refined_xy"][snapped] - cells[ref["target_idx"][snapped]], axis=1)
    right_snap = float(np.mean(ref["target_idx"][snapped] == partner[snapped]))
    right_assign = float(np.mean(ref["target_idx"][assigned] == partner[assigned])) if assigned.any() else float("nan")

    numbers["metrics"] = {
        "n_cells": int(len(cells)), "n_nuclei": int(len(nuclei)), "cell_spacing_median_nn": round(spacing, 1),
        "true_median_error_before": round(before, 2),
        "runs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in runs.items()},
        "refinement": {
            "n_snapped": int(snapped.sum()), "n_assigned": int(assigned.sum()),
            "snapped_residual_max": float(resid.max()) if len(resid) else None,
            "snapped_on_true_partner": round(right_snap, 3),
            "assigned_on_true_partner": round(right_assign, 3),
            "validation_median_error": round(ref["validation"]["median_error"], 2),
            "median_deviation": {k: round(v, 2) for k, v in ref["median_deviation"].items()},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    labels = ["no transform"] + list(runs)
    x = np.arange(len(labels))
    w = 0.26
    fit = [np.nan] + [runs[k]["tre_fit"] for k in runs]
    val = [runs["ransac+tps"]["tre_initial"]] + [runs[k]["tre_validation"] for k in runs]
    true = [before] + [runs[k]["true_median_error"] for k in runs]
    ax.bar(x - w, fit, w, color=GREY, label="tre_fit: pairs used to fit")
    ax.bar(x, val, w, color=ALT, label="tre_initial / tre_validation: held out")
    ax.bar(x + w, true, w, color=NUC, label="true error to the real partner (median)")
    for xs, vals in ((x - w, fit), (x, val), (x + w, true)):
        for xi, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.axhline(spacing, color="#4a5568", ls=":", lw=1)
    ax.text(len(labels) - 0.55, spacing + 0.3, f"median cell spacing {spacing:.0f} µm", fontsize=8,
            color="#4a5568", ha="right")
    ax.set_xticks(x, ["no transform", "ransac+tps", "tps,\nno regularisation", "multiround"])
    ax.set_ylabel("µm")
    ax.set_title("(a) Fit TRE can look perfect while the registration is not")
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, max(true) * 1.35)

    ax = axes[1]
    cats = ["snapped\n(residual)", "snapped\n(held-out check)", "snapped\non right cell", "assigned\non right cell"]
    vals = [float(resid.max()) if len(resid) else 0.0, ref["validation"]["median_error"],
            100 * right_snap, 100 * right_assign]
    colors = [GREY, ALT, NUC, NUC]
    ax2 = ax.twinx()
    ax.bar([0, 1], vals[:2], color=colors[:2], width=0.6)
    ax2.bar([2, 3], vals[2:], color=colors[2:], width=0.6, alpha=0.85)
    ax.text(0, vals[0] + 0.25, f"{vals[0]:.1f} µm\n(by construction)", ha="center", fontsize=8)
    ax.text(1, vals[1] + 0.1, f"{vals[1]:.1f} µm", ha="center", fontsize=8)
    ax2.text(2, vals[2] + 2, f"{vals[2]:.0f}%", ha="center", fontsize=8)
    ax2.text(3, vals[3] + 2, f"{vals[3]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(4), cats, fontsize=8.5)
    ax.set_ylabel("µm")
    ax.set_ylim(0, max(vals[:2]) * 1.6 + 1)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("% of pairs")
    ax2.spines["right"].set_visible(True)
    ax.set_title("(b) Snapping: zero residual, so check differently")
    save(fig, "registration_metrics.png")


if __name__ == "__main__":
    numbers = {}
    figure_pretransform(numbers)
    figure_metrics(numbers)
    print(json.dumps(numbers, indent=2))
