"""
Render docs/assets/registration_multiround.png: how multi-round (displacement
voting) centroid registration works, on Venture 5 z5 post-stain nuclei and
Xenium cells. Edit XEN / NUC below for other data.

    python examples/multiround_figure.py
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from smint.alignment import centroid_registration as cr

XEN = "/vast/projects/BCRL_Multi_Omics/venture_pt5/z5_pre_aligned_coordinates.csv"
NUC = "/vast/projects/BCRL_Multi_Omics/venture_pt5/segmentations/wsi/transformed_nuc_points_in_global.csv"
xen = pd.read_csv(XEN, usecols=["x_centroid", "y_centroid"]).to_numpy(float)
nuc = pd.read_csv(NUC, usecols=["x_centroid", "y_centroid"]).to_numpy(float)
held = np.random.default_rng(0).random(len(nuc)) < 0.25
fit, val = nuc[~held], nuc[held]

multi = cr.register_centroids(nuc, xen)                      # default multiround
single = cr.register_centroids(fit, xen, method="ransac+tps")
CELL, NUCL, SINGLE = "#2b6cb0", "#d53f8c", "#dd8a2e"

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.3, wspace=0.25)

# (a) one block's votes vs its nearest-neighbour offsets
k = 64
d, j = cKDTree(xen).query(fit, k=k, distance_upper_bound=100)
cx, cy = np.median(fit, axis=0)
in_block = (np.abs(fit[:, 0] - cx) < 250) & (np.abs(fit[:, 1] - cy) < 250)
valid = np.isfinite(d[in_block])
votes = (xen[np.where(valid, j[in_block], 0)] - fit[in_block][:, None, :])[valid]
edges = np.arange(-60, 62, 2.0)
h = gaussian_filter(np.histogram2d(votes[:, 0], votes[:, 1], bins=[edges, edges])[0], 1.0)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(h.T, origin="lower", extent=(edges[0], edges[-1], edges[0], edges[-1]), cmap="magma")
ix, iy = np.unravel_index(np.argmax(h), h.shape)
peak = ((edges[ix] + edges[ix + 1]) / 2, (edges[iy] + edges[iy + 1]) / 2)
nn = xen[cKDTree(xen).query(fit[in_block])[1]] - fit[in_block]
ax.scatter(*nn[::3].T, s=3, c="white", alpha=0.35, label="nearest-cell offsets")
ax.scatter(*peak, s=140, marker="+", c="#7fff7f", linewidths=2.5, label=f"vote peak ({peak[0]:.0f}, {peak[1]:.0f}) µm")
ax.scatter(0, 0, s=50, marker="o", facecolors="none", edgecolors="white", label="no shift")
ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
ax.set_xlabel("dx (µm)"); ax.set_ylabel("dy (µm)")
ax.set_title("(a) Votes in one 500 µm block", fontsize=11)
ax.legend(loc="lower left", fontsize=7.5, framealpha=0.6)

# (b) block displacement field across the section
centres, peaks = cr.vote_block_displacements(fit, xen, max_distance=100, block_size=500)
_, consistent = cr.estimate_affine_ransac(centres, centres + peaks, residual_threshold=6.0)
ax = fig.add_subplot(gs[0, 1])
ax.scatter(*xen[::10].T, s=0.3, c="0.85")
c_ok = centres[consistent]; p_ok = peaks[consistent]
q = ax.quiver(c_ok[:, 0], c_ok[:, 1], p_ok[:, 0], p_ok[:, 1], np.hypot(*p_ok.T), cmap="viridis",
              clim=(0, 40), angles="xy", scale_units="xy", scale=0.06, width=0.005)
ax.scatter(*centres[~consistent].T, s=10, marker="x", c="#c53030", linewidths=1, label="inconsistent block (ignored)")
fig.colorbar(q, ax=ax, fraction=0.046, pad=0.02, label="offset (µm)")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower left", fontsize=7.5, frameon=False)
ax.set_title(f"(b) Vote peaks of {len(centres)} blocks", fontsize=11)

# (c) same window: one-shot vs multiround
half = 120
win = (cx - half, cx + half, cy - half, cy + half)
def zoom(ax, pts, colour, title):
    for xy, c, s in ((xen, CELL, 26), (pts, colour, 26)):
        sel = (xy[:, 0] > win[0]) & (xy[:, 0] < win[1]) & (xy[:, 1] > win[2]) & (xy[:, 1] < win[3])
        ax.scatter(*xy[sel].T, s=s, c=c, alpha=0.85, linewidths=0)
    ax.set_xlim(win[:2]); ax.set_ylim(win[2:]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
zoom(fig.add_subplot(gs[0, 2]), single["transform"](nuc), SINGLE, "(c) One-shot ransac+tps")
zoom(fig.add_subplot(gs[0, 3]), multi["transform"](nuc), NUCL, "multiround (240 µm window)")

# (d) held-out distances to nearest cell
ax = fig.add_subplot(gs[1, 0:2])
tree = cKDTree(xen)
bins = np.linspace(0, 40, 81)
for pts, c, lab in ((val, "0.55", "no transform"), (single["transform"](val), SINGLE, "one-shot ransac+tps"),
                    (multi["transform"](val), NUCL, "multiround")):
    dd = tree.query(pts)[0]
    ax.hist(dd, bins=bins, histtype="step", lw=2, color=c, label=f"{lab}: median {np.median(dd):.1f} µm")
ax.set_xlabel("held-out nucleus → nearest Xenium cell (µm)"); ax.set_ylabel("nuclei")
ax.set_title("(d) Held-out nuclei: distance to nearest cell", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)

# (e) rounds
ax = fig.add_subplot(gs[1, 2:4])
r = multi["rounds"]
labels = ["start"] + [f"round {i}: {x['stage']}\n{x['block_size']:.0f} µm blocks" for i, x in enumerate(r[1:], 1)]
vals = [x["median_nn"] for x in r]
colours = ["0.55"] + [NUCL if x["accepted"] else "0.8" for x in r[1:]]
bars = ax.bar(labels, vals, color=colours)
for b, x in zip(bars[1:], r[1:]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "kept" if x["accepted"] else "rejected",
            ha="center", va="bottom", fontsize=9)
ax.axhline(np.median(tree.query(single["transform"](fit))[0]), color=SINGLE, ls="--", lw=1.5,
           label="one-shot ransac+tps")
ax.set_ylabel("fitting nuclei: median NN (µm)")
ax.set_title("(e) Each round is kept only if it helps", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper right")

handles = [plt.Line2D([], [], ls="", marker="o", color=c, label=l) for c, l in
           ((CELL, "Xenium cells"), (SINGLE, "nuclei, one-shot"), (NUCL, "nuclei, multiround"))]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
fig.suptitle("Multi-round registration: post-stain nuclei → Xenium cells (Venture 5 z5)", fontsize=14, y=0.995)
fig.savefig("docs/assets/registration_multiround.png", dpi=110, bbox_inches="tight")
print("ok")
