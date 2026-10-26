"""
Render the registration figures used in the README and documentation.

Two figures, one per registration regime:

* ``registration_sequential.png`` -- spatial metabolomics (SM) registered onto
  spatial transcriptomics (ST) from a *different* section with STalign LDDMM,
  guided by manual landmarks.
* ``registration_same_section.png`` -- segmented nuclei from a post-stain of
  the *same* section, coarsely pre-registered by an orientation search and
  then fitted onto Xenium cells, with TRE on held-out pairs.

The sequential figure needs STalign (numpy < 2); run it from an STalign
environment. Results are cached in ``--cache`` so the plots can be re-rendered
without re-registering.

    python examples/registration_figures.py \\
        --st Z2_final_aligned_annos.csv \\
        --sm Ven5B_information_matrix_tissue_only_full_with_xy.csv \\
        --st-points ven5_z2_st_points_B_points.npy \\
        --sm-points ven5_z2_sm_points_B_points.npy \\
        --nuclei nuc_centroids_xy.csv \\
        --outdir docs/assets --cache registration_cache
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smint.alignment import (
    apply_pretransform,
    build_pretransform,
    describe_pretransform,
    load_landmarks,
    overlap_score,
    read_sm_matrix,
    register_centroids,
    register_sm_to_st,
)

ST_COLOR = "#2b6cb0"
SM_COLOR = "#d53f8c"
POINT_SIZE = 0.15


def _cached(path, compute):
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    result = compute()
    np.savez(path, **result)
    return result


def _style(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _subsample(xy, n, seed=0):
    if len(xy) <= n:
        return xy
    return xy[np.random.default_rng(seed).choice(len(xy), n, replace=False)]


def sequential(args):
    def compute():
        st = pd.read_csv(args.st, usecols=["x_final", "y_final"])
        sm = read_sm_matrix(args.sm, scale_xy=args.scale_xy, rotate_left_90=args.rotate_left_90,
                            usecols=["x", "y"], verbose=False)
        # orientation="stalign" returns true xy in the ST frame, so the result
        # overlays the ST cells; the default ("dataset") is transposed.
        reg = register_sm_to_st(args.st, args.sm, args.st_points, args.sm_points,
                                niter=args.niter, orientation="stalign",
                                sm_kwargs=dict(scale_xy=args.scale_xy,
                                               rotate_left_90=args.rotate_left_90))
        return {
            "st": st[["x_final", "y_final"]].to_numpy(float),
            "sm": sm[["x", "y"]].to_numpy(float),
            "sm_reg": reg[["x_transformed", "y_transformed"]].to_numpy(float),
            "st_pts": load_landmarks(args.st_points)[:, ::-1],
            "sm_pts": load_landmarks(args.sm_points)[:, ::-1],
        }

    r = _cached(os.path.join(args.cache, "sequential.npz"), compute)
    st, sm, sm_reg = r["st"], _subsample(r["sm"], 120_000), _subsample(r["sm_reg"], 120_000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    ax = axes[0]
    ax.scatter(*st.T, s=POINT_SIZE, c=ST_COLOR, rasterized=True)
    ax.scatter(*sm.T, s=POINT_SIZE, c=SM_COLOR, alpha=0.5, rasterized=True)
    _style(ax, "Before: SM and ST from different sections")

    ax = axes[1]
    ax.scatter(*st.T, s=POINT_SIZE, c=ST_COLOR, alpha=0.25, rasterized=True)
    ax.scatter(*sm.T, s=POINT_SIZE, c=SM_COLOR, alpha=0.15, rasterized=True)
    for i, ((sx, sy), (tx, ty)) in enumerate(zip(r["sm_pts"], r["st_pts"]), 1):
        ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", color="0.15", lw=0.8))
        ax.text(tx, ty, str(i), fontsize=8, color="0.1", ha="left", va="bottom")
    _style(ax, f"{len(r['st_pts'])} paired landmarks (SM → ST)")

    ax = axes[2]
    ax.scatter(*st.T, s=POINT_SIZE, c=ST_COLOR, rasterized=True)
    ax.scatter(*sm_reg.T, s=POINT_SIZE, c=SM_COLOR, alpha=0.5, rasterized=True)
    _style(ax, f"After: STalign LDDMM ({args.niter} iterations)")

    handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=l)
               for c, l in ((ST_COLOR, "Xenium ST cells"), (SM_COLOR, "MALDI SM pixels"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Sequential sections: spatial metabolomics → spatial transcriptomics",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = os.path.join(args.outdir, "registration_sequential.png")
    fig.savefig(out, dpi=args.dpi)
    print("wrote", out)


def same_section(args):
    def compute():
        nuc = pd.read_csv(args.nuclei, usecols=args.nuclei_cols)[args.nuclei_cols].to_numpy(float)
        xen = pd.read_csv(args.st, usecols=args.xenium_cols)[args.xenium_cols].to_numpy(float)
        best = (-1.0, 0.0, False, False)
        for fx in (False, True):
            for fy in (False, True):
                for rot in range(-180, 180, 15):
                    m = build_pretransform(nuc, xen, rotation=float(rot), flip_x=fx, flip_y=fy)
                    score = overlap_score(apply_pretransform(nuc, m), xen)
                    if score > best[0]:
                        best = (score, float(rot), fx, fy)
        score, rot, fx, fy = best
        matrix = build_pretransform(nuc, xen, rotation=rot, flip_x=fx, flip_y=fy)
        pre = apply_pretransform(nuc, matrix)
        res = register_centroids(pre, xen, method=args.method,
                                 max_distance=args.max_distance, validation_fraction=0.25)
        print("pretransform:", describe_pretransform(matrix), "overlap", round(score, 3))
        print("TRE initial / fit / held-out:", res["tre_initial"], res["tre_fit"], res["tre_validation"])
        return {
            "nuc": nuc, "xen": xen, "pre": pre, "reg": res["transform"](pre),
            "overlap": score, "n_matched": res["n_matched"],
            "tre": np.array([res["tre_initial"], res["tre_fit"], res["tre_validation"]]),
        }

    r = _cached(os.path.join(args.cache, "same_section.npz"), compute)
    xen, nuc, pre, reg = r["xen"], r["nuc"], r["pre"], r["reg"]
    tre = r["tre"]

    # Zoom window: a square in the middle of the tissue
    cx, cy = np.median(xen, axis=0)
    half = args.zoom / 2
    window = (cx - half, cx + half, cy - half, cy + half)

    def zoom(ax, moving, title):
        for xy, c, a in ((xen, ST_COLOR, 0.9), (moving, SM_COLOR, 0.9)):
            sel = ((xy[:, 0] > window[0]) & (xy[:, 0] < window[1])
                   & (xy[:, 1] > window[2]) & (xy[:, 1] < window[3]))
            ax.scatter(*xy[sel].T, s=6, c=c, alpha=a, linewidths=0, rasterized=True)
        ax.set_xlim(window[:2])
        ax.set_ylim(window[2:])
        _style(ax, title)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.75])
    ax = fig.add_subplot(gs[0])
    ax.scatter(*xen.T, s=POINT_SIZE, c=ST_COLOR, rasterized=True)
    ax.scatter(*nuc.T, s=POINT_SIZE, c=SM_COLOR, alpha=0.5, rasterized=True)
    _style(ax, "Before: nuclei in their own frame")

    ax = fig.add_subplot(gs[1])
    ax.scatter(*xen.T, s=POINT_SIZE, c=ST_COLOR, rasterized=True)
    ax.scatter(*pre.T, s=POINT_SIZE, c=SM_COLOR, alpha=0.5, rasterized=True)
    ax.add_patch(plt.Rectangle((window[0], window[2]), args.zoom, args.zoom,
                               fill=False, ec="0.1", lw=1))
    _style(ax, f"Pre-registered: orientation search (overlap {float(r['overlap']):.2f})")

    zoom(fig.add_subplot(gs[2]), reg, f"After {args.method} (zoom)")

    ax = fig.add_subplot(gs[3])
    labels = ["Initial", "Fitted pairs", "Held-out pairs"]
    colors = ["0.65", "0.65", SM_COLOR]
    ax.bar(labels, tre[:, 0], yerr=tre[:, 1], color=colors, capsize=3)
    for i, v in enumerate(tre[:, 0]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Target registration error (µm)")
    ax.set_title(f"TRE over {int(r['n_matched']):,} matched pairs", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=l)
               for c, l in ((ST_COLOR, "Xenium cells"), (SM_COLOR, "Post-stain nuclei"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Same section: post-stain nuclei → Xenium cells", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = os.path.join(args.outdir, "registration_same_section.png")
    fig.savefig(out, dpi=args.dpi)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--st", required=True, help="Xenium/ST annotation CSV")
    p.add_argument("--sm", help="SM information matrix CSV (sequential figure)")
    p.add_argument("--st-points", help="ST landmarks .npy")
    p.add_argument("--sm-points", help="SM landmarks .npy")
    p.add_argument("--nuclei", help="Nuclei centroid CSV (same-section figure)")
    p.add_argument("--nuclei-cols", nargs=2, default=["x", "y"])
    p.add_argument("--xenium-cols", nargs=2, default=["x_centroid", "y_centroid"])
    p.add_argument("--scale-xy", type=float, default=10.0)
    p.add_argument("--rotate-left-90", action="store_true", default=True)
    p.add_argument("--niter", type=int, default=1000)
    p.add_argument("--method", default="multiround")
    p.add_argument("--max-distance", type=float, default=100.0)
    p.add_argument("--zoom", type=float, default=600.0, help="Zoom window size (µm)")
    p.add_argument("--outdir", default="docs/assets")
    p.add_argument("--cache", default="registration_cache")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--only", choices=["sequential", "same_section"])
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    if args.only in (None, "same_section") and args.nuclei:
        same_section(args)
    if args.only in (None, "sequential") and args.sm:
        sequential(args)


if __name__ == "__main__":
    main()
