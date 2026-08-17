# Alignment API

Everything below is importable from `smint.alignment`.

```python
from smint.alignment import register_sm_to_st, register_centroids, build_pretransform
```

---

## Pre-registration

`smint.alignment.pretransform` — coarse scale, rotation and flips, composed
into a single 3×3 affine so the transform is inspectable, savable and
replayable.

### `build_pretransform`

```python
build_pretransform(
    source_xy, reference_xy=None, scale=None, scale_mode="extent",
    preserve_aspect=True, rotation=0.0, flip_x=False, flip_y=False,
    align_centroids=True, center=None,
) -> np.ndarray            # 3x3 affine
```

Applies **flip → rotate → scale → translate** about the source centroid.
Scale is fitted *after* orientation is corrected, because bounding-box extents
change under rotation.

| Parameter | Meaning |
|---|---|
| `scale_mode` | `"extent"` (match bounding-box spans), `"max"` (match maxima), `"none"` |
| `preserve_aspect` | Isotropic scaling; avoids distorting tissue shape |
| `rotation` | Degrees counter-clockwise |
| `align_centroids` | Translate source centroid onto reference centroid |

### `fit_scale_to_reference`

```python
fit_scale_to_reference(source_xy, reference_xy, mode="extent",
                       preserve_aspect=True) -> (sx, sy)
```

Bounding boxes are sensitive to outliers — a few stray points inflate the span
and shrink the fitted factor. Check on a plot before trusting it.

### `overlap_score`

```python
overlap_score(source_xy, reference_xy, pixel_size=50.0) -> float
```

Fraction of occupied reference bins the source also occupies, 0–1. Needs no
correspondences, so it is fast enough to drive an interactive control.

### `describe_pretransform`

```python
describe_pretransform(matrix) -> dict
```

Returns `scale_x`, `scale_y`, `rotation_deg`, `shear_deg`, `translation`,
`determinant`, `reflects`. A negative determinant means the transform includes
a reflection.

### `save_transform` / `load_transform`

```python
save_transform(matrix, path, **metadata) -> Path
load_transform(path) -> np.ndarray
```

Persists a 3×3 affine as JSON together with its decomposition and any metadata
worth recording, so a coarse transform can be reproduced later rather than
being an unrepeatable manual step.

### Matrix helpers

`compose`, `rotation_matrix`, `flip_matrix`, `scale_matrix`,
`translation_matrix`, `apply_pretransform`. `compose(A, B)` applies `A` then
`B`.

---

## ST↔SM registration

`smint.alignment.st_sm_registration` — STalign LDDMM for sequential sections.

### `prepare_landmark_inputs`

```python
prepare_landmark_inputs(st_file, sm_file, output_prefix, dx=30.0,
                        st_x_col="x_final", st_y_col="y_final",
                        sm_kwargs=None) -> dict
```

Phase 1. Rasterises both modalities and writes `<prefix>_st.npz` /
`<prefix>_sm.npz` for annotation. Returns those paths plus the
`<prefix>_*_points.npy` paths the annotator will produce.

### `register_sm_to_st`

```python
register_sm_to_st(st_file, sm_file, st_points_file, sm_points_file,
                  output_path=None, dx=30.0, niter=1000, epV=200.0,
                  device=None, orientation="dataset",
                  sm_kwargs=None, lddmm_params=None) -> pd.DataFrame
```

Phase 2. Rasterises, runs landmark-initialised LDDMM with SM as source and ST
as target, transforms every SM pixel, and returns the SM table with
`x_transformed` / `y_transformed` appended.

`device` defaults to CUDA when available.

### `resolve_st_columns` / `list_columns`

```python
resolve_st_columns(st_file, x_col=None, y_col=None) -> (x_col, y_col)
list_columns(csv_file) -> list
```

Decide which ST columns to register. Explicit names win; otherwise columns are
auto-detected and the choice is logged. `list_columns` reads only the header,
so it is cheap on a 450 MB matrix.

### `read_sm_matrix`

```python
read_sm_matrix(mtx_file, scale_xy=10.0, rotate_left_90=False,
               keep_positive=True, origin=None, usecols=None,
               verbose=True) -> pd.DataFrame
```

Sniffs the delimiter, standardises `x`/`y`, prefixes numeric m/z headers with
`X`, then scales, optionally rotates 90° CCW, and shifts to keep coordinates
non-negative.

!!! tip
    Pass `usecols=["x", "y"]` when you only need coordinates. On a 445 MB
    matrix this cuts load time to ~2 s.

### `transform_points`

```python
transform_points(lddmm_output, x, y, orientation="dataset") -> np.ndarray
```

`orientation="dataset"` restores the source dataset's frame (default);
`"stalign"` returns true xy in the ST frame. The two differ by a transpose.

### Landmarks

```python
load_landmarks(points_file) -> np.ndarray     # (N, 2) row-col
check_landmark_pair(points_source, points_target) -> None
affine_from_landmarks(points_source, points_target)
```

`load_landmarks` reads `point_annotator.py`'s `{label: [(x, y)]}` format and
swaps to the row-col order STalign requires.

!!! note
    `affine_from_landmarks` is **not** needed for the standard workflow —
    `LDDMM` derives its own initialisation from `pointsI`/`pointsJ`.

### Other

`read_st_annotations`, `st_coordinates`, `sm_coordinates`,
`rasterize_coordinates`, `save_rasterized`, `run_lddmm_alignment`,
`save_transformed_data`, `find_sm_matrix_files`, `stalign_available`.

`rasterize_coordinates` always returns a 4-tuple `(X, Y, I, fig)`, with
`fig=None` when `draw=0` — STalign's own function returns 3 or 4 values
depending on that flag.

---

## Centroid registration

`smint.alignment.centroid_registration` — correspondence fitting for
post-staining on the same section.

### `register_centroids`

```python
register_centroids(source_xy, target_xy, method="ransac+tps",
                   max_distance=100.0, mutual=True,
                   validation_fraction=0.25, residual_threshold=10.0,
                   max_trials=1000, tps_regularization=1.0,
                   n_control=1000, random_state=0) -> dict
```

Returns `transform` (callable), `affine`, `tps`, `source_idx`, `target_idx`,
`match_distances`, `inliers`, `n_matched`, and `tre_initial`, `tre_fit`,
`tre_validation` as `(mean, std)` tuples.

**Only `tre_validation` is meaningful** — see
[Measuring quality honestly](../alignment.md#measuring-quality-honestly).

### `register_centroid_files`

```python
register_centroid_files(source_file, target_file,
                        source_cols=("centroid_x", "centroid_y"),
                        target_cols=("x_centroid", "y_centroid"),
                        output_path=None, **kwargs) -> (pd.DataFrame, dict)
```

Every source row is transformed, not just matched ones — matching only selects
the pairs used to *fit*.

### `match_centroids`

```python
match_centroids(source_xy, target_xy, max_distance=100.0,
                mutual=True) -> (source_idx, target_idx, distances)
```

`mutual=True` keeps only reciprocal nearest neighbours.

### `fit_tps` / `apply_tps`

```python
fit_tps(source_xy, target_xy, regularization=1.0, n_control=1000) -> dict|None
apply_tps(points_xy, model, chunk_size=50000) -> np.ndarray
```

!!! warning "Control points cost cubically"
    TPS solves an `(n_control + 3)` square system. 74,000 control points needs
    a ~44 GB solve, plus as much again for the kernel. `n_control=1000` is a
    sane default; `regularization=0` gives exact interpolation, which is
    almost never what you want.

### Other

`estimate_affine`, `estimate_affine_ransac`, `apply_affine`,
`target_registration_error`, `split_pairs`.

---

## Jobs

`smint.alignment.jobs` — run registration in a separate process, locally or via
SLURM.

### `JobSpec`

```python
JobSpec(round, inputs, output_dir, params={}, backend="sbatch",
        resources=SlurmResources(), worker_python=None)
```

`round` is `"st_sm"` or `"centroid"`. Required `inputs`:

| Round | Keys |
|---|---|
| `st_sm` | `st_file`, `sm_file`, `st_points`, `sm_points` |
| `centroid` | `source_file`, `target_file` |

`worker_python` defaults to `SMINT_WORKER_PYTHON`, then an `STalign_env`
beside the package, then the current interpreter.

`validate()` checks required keys exist on disk and — for sbatch — rejects
node-local paths a compute node cannot see.

### `SlurmResources`

```python
SlurmResources(partition="regular", cpus_per_task=8, memory="64G",
               time_limit="04:00:00", job_name="smint_register", gpus=0)
```

`check()` **raises** if the partition does not exist on this cluster — turning
SLURM's terse "Invalid partition name specified" into a message listing the
valid options — and warns when `gpus > 0` on a non-GPU partition (the job would
sit pending rather than fail) and vice versa. `available_partitions()` returns
the cluster's partitions via `sinfo`, or an empty list off-cluster.

### `submit` / `poll`

```python
submit(spec) -> dict      # job_id (sbatch) or pid (local)
poll(output_dir) -> dict  # state, message, metrics, outputs
```

`poll` reconciles the worker's status file with SLURM, so a job killed by OOM
or timeout is reported as failed rather than appearing to run forever.

States: `submitted`, `queued`, `running`, `completed`, `failed`, `cancelled`,
`unknown`.

### Worker CLI

```bash
smint-register /path/to/job_spec.json
python -m smint.cli.register /path/to/job_spec.json
```

The only place STalign is imported — which is what lets a napari front end in
a numpy≥2 environment drive a registration that needs numpy<2.

---

## Command line

```bash
smint-alignment pretransform --source sm.csv --target xen.csv \
    --output sm_pre.csv --search

smint-alignment centroid --source nuc.csv --target xen.csv \
    --output-dir ./run --method ransac --watch

smint-alignment st-sm --st Z2.csv --sm Ven5B.csv \
    --st-points z2_st_points.npy --sm-points z2_sm_points.npy \
    --output-dir /vast/scratch/you/run --submit --watch
```

`--submit` sends the work to SLURM; `--watch` blocks until it finishes.
`smint-register <job_spec.json>` runs a spec directly and is what the batch
script invokes.

---

## Removed in this release

`st_align_wrapper` has been deleted. Its `align_spatial_transcriptomics`
shelled out to a `stalign` command-line tool that does not exist — STalign is a
Python library — so every call returned None. `load_alignment` and
`save_alignment` read and wrote a `transformation.json` /
`aligned_coordinates.csv` format that nothing in the package produced.

| Removed | Use instead |
|---|---|
| `align_spatial_transcriptomics` | `register_sm_to_st` or `register_centroids` |
| `run_alignment` | `register_centroids` |
| `transform_coordinates` | `apply_affine` / `apply_pretransform` |
| `save_alignment` / `load_alignment` | `save_transform` / `load_transform` |
| `apply_transformation` | `apply_affine` |
| `prepare_visium_data` | — (the project uses Xenium/CosMx, not Visium) |

The `smint-alignment` console script now points at `smint.cli.align`. It
previously referenced `scripts.run_alignment`, which is not part of the
installed package, so it failed with `ModuleNotFoundError`. `smint-segmentation`
had the same defect and has been withdrawn until it has a real implementation.

`smint.alignment.xenium_metabolomics` remains, deprecated, superseded by
`st_sm_registration`.
