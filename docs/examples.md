# Examples

Worked examples on real data from the BCRL Venture cohort, run on the WEHI
Milton cluster. Timings and results are from those runs; your numbers will
depend on the data and hardware.

To try the API without any of these datasets, run the synthetic examples
shipped with the repository:

```bash
python examples/alignment_example.py    # pre-registration, centroid fitting, job specs
```

---

## 1. Spatial metabolomics → spatial transcriptomics

MALDI spatial metabolomics (SM) acquired on a section neighbouring a Xenium
spatial transcriptomics (ST) section. The tissue differs between sections, so
this uses STalign LDDMM guided by manual landmarks.

![Sequential-section registration: SM onto ST, before, landmarks and after LDDMM](assets/registration_sequential.png)

*Venture 5, Xenium section Z2 and MALDI section Ven5B. 459,969 SM pixels,
53,622 ST cells, 11 landmark pairs.*

**Inputs**

| File | Contents |
|---|---|
| `Z2_final_aligned_annos.csv` | Xenium cells; registration uses `x_final` / `y_final` (auto-detected) |
| `Ven5B_information_matrix_tissue_only_full_with_xy.csv` | MALDI pixels: `x`, `y` and one column per m/z (442 MB) |
| `ven5_z2_st_points_B_points.npy`, `ven5_z2_sm_points_B_points.npy` | 11 paired landmarks |

### Step 1 — rasterise both modalities for landmarking

```python
from smint.alignment import prepare_landmark_inputs

out = prepare_landmark_inputs(
    st_file="Z2_final_aligned_annos.csv",
    sm_file="Ven5B_information_matrix_tissue_only_full_with_xy.csv",
    output_prefix="ven5_z2",
    sm_kwargs=dict(scale_xy=10.0, rotate_left_90=True),   # MALDI pixels -> microns
)
```

### Step 2 — place landmarks

Open the two datasets in the [napari plugin](napari_plugin.md) (**Plugins →
SMINT Registration → 3. Landmarks**), or run STalign's `point_annotator.py` on
the two `.npz` files. Pick the **same features in the same order** in both;
correspondence is positional.

### Step 3 — register on a GPU node

```bash
smint-alignment st-sm \
    --st Z2_final_aligned_annos.csv \
    --sm Ven5B_information_matrix_tissue_only_full_with_xy.csv \
    --st-points ven5_z2_st_points_B_points.npy \
    --sm-points ven5_z2_sm_points_B_points.npy \
    --rotate-left-90 --orientation stalign \
    --output-dir /vast/scratch/$USER/ven5_z2 \
    --submit --watch --partition gpuq --gpus 1 --memory 64G \
    --worker-python /path/to/STalign_env/bin/python
```

```text
submitted st_sm via sbatch (30527655)
state: completed -- st_sm round finished
{"transformed_csv": "/vast/scratch/.../ven5_z2/sm_transformed.csv"}
```

The worker ran LDDMM (1,000 iterations) on one A30; from submission to
finished output, including queueing, took 72 s. The output keeps every m/z column and adds
`x_transformed` / `y_transformed`.

The same run from Python:

```python
from smint.alignment import register_sm_to_st

sm = register_sm_to_st(
    st_file="Z2_final_aligned_annos.csv",
    sm_file="Ven5B_information_matrix_tissue_only_full_with_xy.csv",
    st_points_file="ven5_z2_st_points_B_points.npy",
    sm_points_file="ven5_z2_sm_points_B_points.npy",
    sm_kwargs=dict(scale_xy=10.0, rotate_left_90=True),
    orientation="stalign",
    output_path="sm_transformed.csv",
)
```

!!! warning "Pass `orientation=\"stalign\"` to overlay on ST"
    The default, `"dataset"`, reproduces historical outputs but is transposed
    relative to the ST cells. Plotted over ST it looks like a failed
    registration. See [Coordinate orientation](alignment.md#coordinate-orientation).

### Step 4 — check the result

```python
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

st = pd.read_csv("Z2_final_aligned_annos.csv", usecols=["x_final", "y_final"]).to_numpy()
sm = pd.read_csv("sm_transformed.csv", usecols=["x_transformed", "y_transformed"]).to_numpy()
d, _ = cKDTree(st).query(sm)
print(f"median distance to nearest ST cell: {np.median(d):.1f} um")
```

```text
median distance to nearest ST cell: 16.4 um
```

That is about half a 30 µm MALDI pixel. Because the sections differ, there
is no ground truth to score against; this distance and the overlay are the
practical checks.

---

## 2. Coarse pre-registration with an orientation search

Before landmarking or fitting, bring a dataset roughly into the target frame.
Here 92,610 segmented nuclei are fitted to the same Xenium cells:

```bash
smint-alignment pretransform \
    --source nuclei.csv --target Z2_final_aligned_annos.csv \
    --search --output nuclei_pre.csv
```

```text
scale        : 0.9554, 0.9554
rotation     : -15.00 deg
reflects     : False
overlap      : 0.832
wrote        : nuclei_pre.csv
transform    : nuclei_pre.csv.transform.json
```

`--search` tries every 15° rotation with and without each flip and keeps the
best overlap score; it took 13 s. The matrix and its decomposition are saved
next to the output, so the step can be reproduced exactly. The napari
plugin's **2. Pre-register** widget does the same interactively.

---

## 3. Same-section registration, and reading TRE

When a second modality is acquired on the *same* section (for example
post-staining), centroids correspond one-to-one and a global transform can be
fitted:

```bash
smint-alignment centroid \
    --source nuclei_pre.csv --source-cols x y \
    --target Z2_final_aligned_annos.csv \
    --method ransac+tps \
    --output-dir /vast/scratch/$USER/nuc_to_xenium \
    --submit --watch --partition regular
```

```text
{
  "method": "ransac+tps",
  "n_matched": 21238,
  "tre_initial": 10.17,
  "tre_fit": 11.62,
  "tre_validation": 12.31,
  "inlier_fraction": 0.498
}
```

**How to read this.** `tre_validation` is the error on pairs held out from
fitting, and the only number to trust. Here it is *worse* than `tre_initial`:
the transform did not improve on the starting alignment. With a residual of
~10 µm — about one cell spacing — and only half the matches accepted as
inliers, the remaining distance is ambiguity in which cell matches which, not
misregistration. These nuclei were already as well aligned as a global
transform can make them. On data that needs registering, `tre_validation`
falls well below `tre_initial`. See
[Measuring quality honestly](alignment.md#measuring-quality-honestly).

---

## 4. Segmenting a whole-slide image on one GPU

A fine-tuned Cellpose-SAM model segments cells, and Cellpose's base model
segments nuclei, chunk by chunk:

```python
from smint.segmentation import process_large_image

results = process_large_image(
    image_path="r6_crop.ome.tif",            # (C, Y, X): ch0 nuclear, ch1 cell
    csv_base_path="seg/r6",
    chunk_size=(1024, 1024),
    cell_model_path="/path/to/models/cpsam_20260225_153049",
    cells_diameter=120, cells_flow_threshold=0.6, cells_cellprob_threshold=-2.5,
    cells_channels=[2, 1],                  # main = channel 2 (cell), nuclear = channel 1
    nuclei_model_path="nuclei",             # Cellpose 4: falls back to cpsam, with a warning
    nuclei_diameter=25, nuclei_flow_threshold=0.3, nuclei_cellprob_threshold=0.2,
    nuclei_channels=[1, 0],
    visualize=True, visualize_output_dir="seg/vis",
)
print(results["total_cells"], results["total_nuclei"])
```

```text
33 29
```

![Cell (red) and nucleus (cyan) outlines on one 1024 px chunk](assets/segmentation_example.jpg)

*Venture 6, 2048 × 2048 px crop, Cellpose 4 on one A30: 23 s. Green: nuclear
channel; blue: cell channel.*

`channels` follows Cellpose 3's convention (`[main, nuclear]`, 1-based, 0 =
none) and selects the same input on Cellpose 3 and 4.

---

## 5. Distributed segmentation on SLURM GPUs

For a whole slide, tiles are spread across GPU workers that SMINT launches
through SLURM, then stitched and reconciled:

```python
from smint.segmentation import run_distributed_segmentation

results = run_distributed_segmentation(
    image_path="Venture6_tile_scan_R6.ome.tif",       # OME-TIFF or OME-Zarr, on shared storage
    output_dir="/vast/scratch/me/r6_seg",
    cell_model="/path/to/models/cpsam_20260225_153049",
    nuclei_model="nuclei",
    cell_params={"diameter": 120, "flow_threshold": 0.6,
                 "cellprob_threshold": -2.5, "channels": [2, 1]},
    nuclei_params={"diameter": 25, "flow_threshold": 0.3,
                   "cellprob_threshold": 0.2, "channels": [1, 0]},
    chunk_size=1048, tile_overlap=256,
    backend="slurm", n_workers=4,
    slurm={"queue": "gpuq", "gpu_type": "A30", "memory": "64GB"},
)
```

```text
Tiles: 16 of 1048 px, 256 px overlap
4 worker(s) ready
Stitched 66 polygons: 44 after merging, 44 after trim/remove
30 of 103 cell outlines touch no nucleus and were dropped
Stitched 73 polygons: 43 after merging, 43 after trim/remove
Reconciled 43 cells / 44 nuclei -> 44 cells / 44 nuclei
Done in 0.6 min
```

Run the coordinator itself as a small CPU job (`sbatch` on `regular`); it
submits the GPU workers and only stitches. The output directory then holds
`cells_outlines.csv` / `nuclei_outlines.csv` (outline vertices),
`cells_summary.csv` / `nuclei_summary.csv` (centroids, areas, and which cell
each nucleus belongs to) and full-resolution label masks. See
[Segmentation](segmentation.md#stitching-tiles) for how tiles are stitched.
