"""
Figure for docs/concepts/coordinate_frames.md: the STalign transpose.

Runs SMINT's sequential-section path end to end on a synthetic pair --
``smint-alignment pretransform`` (via ``smint.cli.align.main``), landmarks
written with the napari plugin's ``save_landmarks``, then
``register_sm_to_st`` once per ``orientation`` -- and plots both outputs over
the ST cells.

Needs an STalign environment (numpy<2, torch). LDDMM runs on the CPU in a few
seconds at this size:

    PYTHONPATH=. <stalign-env>/bin/python docs/assets/concepts/src/make_stalign_figure.py

Writes docs/assets/concepts/stalign_orientation.png and prints the numbers
quoted on the page.
"""

import json
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smint.alignment import load_transform, register_sm_to_st
from smint.alignment.pretransform import apply_pretransform, rotation_matrix
from smint.cli.align import main as smint_alignment
from smint.napari._io import save_landmarks

logging.basicConfig(level=logging.WARNING)

OUT = Path(__file__).resolve().parents[1]
CELL = "#2b6cb0"
NUC = "#d53f8c"
DPI = 130

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def in_tissue(p):
    x, y = p[:, 0], p[:, 1]
    return (((x > 0) & (x < 600) & (y > 0) & (y < 2400))
            | ((x > 0) & (x < 1800) & (y > 1800) & (y < 2400))
            | ((x > 0) & (x < 1200) & (y > 900) & (y < 1400)))


def warp(p):
    """A gentle non-linear distortion between the two sections."""
    return p + np.c_[40 * np.sin(p[:, 1] / 500), 30 * np.cos(p[:, 0] / 400)]


def main():
    rng = np.random.default_rng(0)
    work = Path(tempfile.mkdtemp(prefix="smint_frames_"))

    # ST: cell centroids in microns
    st = rng.uniform([0, 0], [1800, 2400], (60000, 2))
    st = st[in_tissue(st)][:15000]
    pd.DataFrame({"cell_id": np.arange(len(st)), "x_centroid": st[:, 0], "y_centroid": st[:, 1]}) \
        .to_csv(work / "st.csv", index=False)

    # SM: a 10 um MALDI-like grid of the neighbouring section, warped, turned 25 deg,
    # stored in pixel units -- with one fake m/z column
    grid = np.mgrid[0:1800:10, 0:2400:10].reshape(2, -1).T.astype(float)
    grid = grid[in_tissue(grid)]
    turn = rotation_matrix(25, center=grid.mean(0))
    sm_px = apply_pretransform(warp(grid), turn) / 10.0 + [20, 20]
    pd.DataFrame({"x": sm_px[:, 0], "y": sm_px[:, 1], "123.4567": rng.random(len(sm_px))}) \
        .to_csv(work / "sm.csv", index=False)

    # 1. Coarse pre-registration with the CLI (scales pixels -> microns, turns back 25 deg)
    smint_alignment(["pretransform", "--source", str(work / "sm.csv"), "--target", str(work / "st.csv"),
                     "--rotation", "-25", "--output", str(work / "sm_pre.csv")])
    matrix = load_transform(str(work / "sm_pre.csv") + ".transform.json")
    sm_pre = pd.read_csv(work / "sm_pre.csv")[["x", "y"]].to_numpy()

    # 2. Landmarks, saved the way the napari plugin saves them: (x, y) per label
    lm_st = np.array([[300, 100], [300, 2300], [1700, 2100], [1100, 1150], [500, 1600]], float)
    lm_sm = apply_pretransform(apply_pretransform(warp(lm_st), turn) / 10.0 + [20, 20], matrix)
    save_landmarks(lm_st, work / "z_st_points.npy")
    save_landmarks(lm_sm, work / "z_sm_points.npy")

    # 3. LDDMM, once per output orientation. The SM table is already in microns,
    #    so scale_xy must be 1.0 here -- the default 10.0 would scale it again.
    out = {}
    for orientation in ("dataset", "stalign"):
        df = register_sm_to_st(
            st_file=str(work / "st.csv"), sm_file=str(work / "sm_pre.csv"),
            st_points_file=str(work / "z_st_points.npy"), sm_points_file=str(work / "z_sm_points.npy"),
            niter=1000, device="cpu", orientation=orientation, sm_kwargs={"scale_xy": 1.0},
        )
        out[orientation] = df[["x_transformed", "y_transformed"]].to_numpy()

    truth = grid   # where each SM pixel really lies in the ST frame
    err = {k: float(np.median(np.linalg.norm(v - truth, axis=1))) for k, v in out.items()}
    err["pretransform_only"] = float(np.median(np.linalg.norm(sm_pre - truth, axis=1)))
    numbers = {"st_cells": int(len(st)), "sm_pixels": int(len(sm_px)),
               "median_error_um": {k: round(v, 1) for k, v in err.items()}}

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.3))
    panels = [
        ("(a) Input: SM in pixels", None, sm_px, "SM pixel grid (x, y in px)"),
        ("(b) After pretransform", st, sm_pre, f"median error {err['pretransform_only']:.0f} µm"),
        ('(c) orientation="dataset"', st, out["dataset"], "x and y swapped: a mirrored overlay"),
        ('(d) orientation="stalign"', st, out["stalign"], f"median error {err['stalign']:.0f} µm"),
    ]
    for ax, (title, ref, moving, note) in zip(axes, panels):
        if ref is not None:
            ax.scatter(ref[:, 0], ref[:, 1], s=1, c=CELL, alpha=0.35, lw=0)
        ax.scatter(moving[:, 0], moving[:, 1], s=0.6, c=NUC, alpha=0.3, lw=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()       # image convention: y grows downwards
        ax.tick_params(labelsize=7)
        ax.text(0.5, -0.19, note, transform=ax.transAxes, ha="center", fontsize=9)
    axes[0].set_xlabel("px", fontsize=8)
    for ax in axes[1:]:
        ax.set_xlabel("µm (ST frame)", fontsize=8)
    axes[1].scatter([], [], s=12, c=CELL, label="ST cells")
    axes[1].scatter([], [], s=12, c=NUC, label="SM pixels")
    axes[1].legend(loc="lower right", fontsize=8, markerscale=1.5, framealpha=0.9)
    fig.savefig(OUT / "stalign_orientation.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
