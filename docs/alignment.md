# Spatial Alignment

SMINT aligns spatial omics modalities in two quite different situations, and
picking the right one matters more than any parameter you will tune afterwards.

| Situation | What is true | Use |
|---|---|---|
| **Sequential sections** — the modalities come from different physical sections | The tissue genuinely differs between them; no cell corresponds one-to-one | STalign LDDMM ([`register_sm_to_st`](#st-sm-registration)) |
| **Same section, post-staining** — a second modality is acquired on the *exact same* section | Centroids really do correspond one-to-one | Correspondence fitting ([`register_centroids`](#centroid-registration)) |

Using the same-section tools on sequential sections will happily produce a
number, but it will be fitting noise: nearest-neighbour "correspondences"
between different sections pair cells that are merely nearby, not the same.

---

## Coarse pre-registration

Spatial metabolomics arrives on its own pixel grid, frequently rotated or
mirrored relative to the Xenium section and at a completely different scale.
LDDMM will not recover from a gross orientation mismatch, so bring the datasets
roughly together first.

```python
from smint.alignment import build_pretransform, apply_pretransform, overlap_score

matrix = build_pretransform(
    sm_xy, xen_xy,
    scale_mode="extent",   # match bounding-box spans
    rotation=30,           # degrees counter-clockwise
    flip_x=True,
)
moved = apply_pretransform(sm_xy, matrix)
print(overlap_score(moved, xen_xy))     # 0-1, higher is better
```

Operations are applied in a fixed order — **flip → rotate → scale → translate**
— about the source centroid. The order is fixed because rotations and flips do
not commute.

!!! note "Scale is fitted after orientation is corrected"
    Bounding-box extents change under rotation, so fitting scale on the raw
    source bakes in an error. On a 30° rotated test shape, fitting first gave
    8.10 against a true 10.0. `build_pretransform` handles this for you by
    applying flip and rotation before fitting.

`scale_mode` options:

- **`"extent"`** (default) — match the bounding-box *span* per axis. Robust to
  the two datasets having different origins.
- **`"max"`** — match maximum coordinates directly. Literal, but wrong whenever
  either dataset does not start near zero.
- **`"none"`** — no scaling.

Scaling is isotropic by default (`preserve_aspect=True`). Anisotropic scaling
can force bounding boxes to agree while making the shapes match *worse*, which
then misleads the landmark step.

Use `describe_pretransform(matrix)` to decompose a transform into scale,
rotation, shear and whether it reflects — a negative determinant means a
reflection crept in, which is easy to introduce by combining a flip with a
rotation.

---

## ST-SM registration

For sequential sections, using STalign's LDDMM. The workflow is deliberately
split into two phases around a **manual landmark step**, because that step is
where the biological judgement lives.

### Phase 1 — prepare landmark inputs

```python
from smint.alignment import prepare_landmark_inputs

out = prepare_landmark_inputs(
    st_file="Z2_final_aligned_annos.csv",
    sm_file="Ven5B_information_matrix_tissue_only_full_with_xy.csv",
    output_prefix="/path/to/ven5_z2",
    dx=30.0,
    sm_kwargs=dict(scale_xy=10.0, rotate_left_90=True),
)
```

This writes `<prefix>_st.npz` and `<prefix>_sm.npz`.

### Phase 2 — annotate, then register

Annotate both `.npz` files, picking the **same anatomical features in the same
order** in each — correspondence is positional, so an ordering mismatch
silently produces a wrong registration. Either use the
[napari plugin](napari_plugin.md) or the standalone annotator:

```bash
python point_annotator.py ven5_z2_st.npz ven5_z2_sm.npz
```

That writes `<prefix>_st_points.npy` and `<prefix>_sm_points.npy`. Then:

```python
from smint.alignment import register_sm_to_st

df = register_sm_to_st(
    st_file=..., sm_file=...,
    st_points_file=out["st_points"],
    sm_points_file=out["sm_points"],
    output_path="sm_transformed.csv",
    niter=1000, epV=200.0,
)
```

### Choosing which columns to register

ST tables routinely carry several coordinate pairs from successive processing
stages — `x_centroid` alongside `x_new` and `x_new_add`, say. Which one you
register against is a real choice, and the wrong one **misregisters silently
rather than raising**.

Inspect first:

```bash
smint-alignment columns Ven5_z2_pre_aligned_1.csv
```

Then be explicit:

```python
register_sm_to_st(
    ..., st_x_col="x_new_add", st_y_col="y_new_add",
    sm_kwargs=dict(x_col="x", y_col="y"),
)
```

```bash
smint-alignment st-sm ... --st-x-col x_new_add --st-y-col y_new_add
```

Omitting them auto-detects in the priority order `x_final`, `x_centroid`,
`centroid_x`, `x_transformed`, `x` — and logs the choice, so it appears in the
job log. In the napari plugin the columns are dropdowns populated from the file
you pick, with the detected pair preselected.

### Coordinate orientation

STalign works in **row-column** order throughout. `LDDMM` takes grids as
`[Y, X]`, and `transform_points_source_to_target` both accepts and returns
points as `(row, col)` — it applies the transform in place and does not
transpose.

Registering through STalign therefore returns coordinates whose axes are
transposed relative to the source dataset. `transform_points` exposes this:

- **`orientation="dataset"`** (default) — restores the original dataset's
  orientation. This is what you want, and it reproduces historical outputs.
- **`orientation="stalign"`** — true xy in the ST target frame.

The two differ by a transpose. Mixing them silently produces a flipped overlay,
so the choice is always explicit and is recorded on the output frame.

---

## Centroid registration

For post-staining on the same section, where centroids correspond one-to-one.

```python
from smint.alignment import register_centroids

result = register_centroids(
    nucleus_xy, xenium_xy,
    method="ransac+tps",
    max_distance=100.0,
    validation_fraction=0.25,
)
print(result["tre_validation"])    # the number that matters
```

| Method | What it does |
|---|---|
| `affine` | Least-squares affine on all matched pairs |
| `ransac` | RANSAC-robust affine; rejects bad correspondences |
| `tps` | Thin-plate spline after an affine pre-alignment |
| `ransac+tps` | RANSAC affine, then TPS on the inliers |

### Measuring quality honestly

Target Registration Error is easy to compute in a way that is **circular**.
Correspondences here are established by nearest neighbour, so a sufficiently
flexible transform can drive TRE to ~0 on the pairs it was fitted to,
regardless of whether those correspondences are correct. An unregularised TPS
with as many control points as pairs will *always* report TRE ≈ 0 on those
pairs — that is interpolation, not accuracy.

`register_centroids` therefore always holds out a fraction of pairs and reports
TRE on both:

```
tre_initial      before any transform
tre_fit          on the pairs used for fitting     <- not evidence
tre_validation   on held-out pairs                 <- the real number
```

A large gap between the two means the transform is memorising
correspondences. The function warns when it detects this.

!!! warning "Matching is mutual by default"
    `mutual=True` keeps only pairs that are each other's nearest neighbour.
    One-directional matching lets many source points collapse onto a single
    popular target, inflating the pair count. On a real Venture 5 section this
    was the difference between 75,146 apparent pairs and 40,135 real ones.

### Interpreting the result

If no method beats the initial TRE, the datasets are already as aligned as a
global transform can make them, and the remaining distance is correspondence
ambiguity rather than misregistration. Two useful diagnostics:

- Compare against the target's own inter-cell spacing. If the residual is a
  large fraction of that, many "matches" are neighbouring cells rather than the
  same cell.
- Sweep `max_distance`. If TRE shrinks in proportion rather than reaching a
  plateau, the threshold is setting the residual, not alignment error.

---

## Running on HPC

Registration can be submitted to SLURM or run locally through a job spec; see
[HPC Deployment](hpc_deployment.md) and the [napari plugin](napari_plugin.md).

```python
from smint.alignment.jobs import JobSpec, SlurmResources, submit, poll

spec = JobSpec(
    round="st_sm",
    inputs={"st_file": ..., "sm_file": ..., "st_points": ..., "sm_points": ...},
    params={"niter": 1000},
    output_dir="/vast/scratch/you/run1",
    backend="sbatch",
    resources=SlurmResources(partition="gpuq", gpus=1, gpu_type="A30", memory="64G"),
)
info = submit(spec)
print(poll(spec.output_dir)["state"])
```

!!! danger "Batch jobs need shared storage"
    An sbatch job runs on a different machine, so node-local paths (`/tmp`,
    `/var/tmp`, `/dev/shm`) are invisible to it — the job dies with no logs at
    all, because the log directory does not exist there. `JobSpec.validate()`
    rejects these up front. Use `/vast/scratch` or `/vast/projects`.

### GPU

LDDMM runs on GPU automatically when one is available. Request it with
`SlurmResources(partition="gpuq", gpus=1)`. GPU and CPU results are
bit-identical; on a Venture 5 Z2 section the GPU run took 82 s against 112 s on
8 CPU cores, so the benefit is modest at typical raster sizes.

`gpuq` mixes A30, A100 and P100 cards, and a bare `--gres=gpu:1` takes whatever
is free — so an untyped request can land on hardware generations apart from the
one a run was timed on. `gpu_type` defaults to `"A30"`, the card registration is
validated against, and renders as `--gres=gpu:A30:1`. Pass another name to pick
a different card, or `gpu_type=None` to accept any:

```python
SlurmResources(partition="gpuq", gpus=1, gpu_type="A100")  # --gres=gpu:A100:1
SlurmResources(partition="gpuq", gpus=1, gpu_type=None)    # --gres=gpu:1
```

Types are checked against the partition before submission, so a card that
partition does not have is reported by name rather than as SLURM's "Invalid
generic resource".

---

## Environments

STalign requires **numpy < 2** — the `nptyping` dependency it pulls in via
`nrrd` uses `np.object0`, which numpy 2 removed. The centroid and
pre-registration paths have no STalign dependency and run anywhere.

If `stalign_available()` returns `False`, check numpy first:

```python
from smint.alignment import stalign_available
print(stalign_available())
```
