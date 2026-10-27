# Changelog

User-visible changes to SMINT, newest first, compiled from the
[commit history](https://github.com/JurgenKriel/SMINT/commits/main). Releases
are listed on [GitHub](https://github.com/JurgenKriel/SMINT/releases); changes
within a release are grouped by topic or month.

## 2.0.0 (2026-10-27)

SMINT is described in [Kriel *et al.*, *Cell Reports Methods*
(2026)](https://doi.org/10.1016/j.crmeth.2026.101306); see
[Citation](citation.md).

### Upgrading from 1.0.0

- `register_centroids`, `register_centroid_files`, `smint-alignment centroid`
  and the napari plugin now default to `method="multiround"` followed by
  snapping refinement. `x_transformed` / `y_transformed` hold the refined
  positions; the transform-only ones are in `x_transformed_smooth` /
  `y_transformed_smooth`. Pass `--no-refine` / `refine=False` for the old
  behaviour.
- `smint-alignment centroid` detects the centroid columns; the old source
  default `centroid_x centroid_y` no longer has to exist.
- `run_lddmm_alignment` initialises from the landmarks
  (`init_from_landmarks=True`); pass `False` to start from the identity.
- Distributed segmentation has a console script,
  `smint-segment-distributed`, in place of
  `python -m smint.segmentation.distributed_seg`.
- Python 3.10 or newer is required.
- `st_align_wrapper` and the `smint-segmentation` console script were
  removed; see [August 2026: registration overhaul](#august-2026-registration-overhaul)
  for their replacements.

### Registration

- **Snapping refinement.** After centroid registration, nuclei that are
  confidently matched to a Xenium cell are moved exactly onto it
  (`refine_by_assignment`, a port of the lab's `alignment_refinement.r`).
  `register_centroid_files` and `smint-alignment centroid` refine by default;
  `--no-refine` / `refine=False` turns it off, `--snap-radius` sets the
  pairing distance. The output gains `x_transformed_smooth`,
  `y_transformed_smooth`, `registration_status` and `target_index`.
- **Remaining nuclei are assigned to free cells**, marked `assigned` and
  given a `neighbour_deviation` so they can be filtered; a warning fires when
  assigned pairs deviate much more than snapped ones.
  `--no-assign-remaining` opts out.
- **New default method `multiround`.** Displacement voting within tissue
  blocks, an affine round and a smooth thin-plate-spline round, each kept only
  if it improves the alignment. On Sample 2 z5 the held-out median distance
  to the nearest cell fell from 12.4 to 5.4 µm. The CLI, napari widget and
  worker use it by default.
- RANSAC affine estimation reimplemented in NumPy, avoiding very high memory
  use on large point sets.
- `register_centroids` warns when the held-out TRE is no better than the
  initial one.
- Fixed: `smint-alignment` passed a TPS regularisation of 1.0 by default,
  overriding the intended value for `multiround`.
- `smint-alignment --orientation` help now matches the documentation.
- **LDDMM starts from the landmarks.** STalign otherwise starts from the
  identity, so sections with a large offset barely moved (synthetic test:
  695 µm median error before, 0.1 µm after). Outputs record their frame in
  `<output>.registration.json`.
- napari: the ST/SM registration now writes results in the ST frame
  (`orientation="stalign"`) and keeps SM coordinates as given; jobs default to
  the cluster's default partition.
- `smint-alignment centroid` detects the centroid columns and gains
  `--gpu-type` (with `--gpus`).

### Segmentation

- **Cellpose 3 and 4 both supported.** Model paths load on both; legacy names
  (`cyto`, `nuclei`) fall back to Cellpose-SAM on Cellpose 4 with a warning;
  `channels` means the same on both.
- **`process_large_image` works again.** It previously failed to load any
  model; the broken chunk-visualisation plot is also fixed.
- **`run_distributed_segmentation` rebuilt**: overlapping tiles, per-worker
  model cache, local, SLURM (`dask-jobqueue`) or existing-scheduler backends,
  OME-Zarr / Zarr / TIFF / OME-TIFF input, and a `smint[distributed]` extra.
- **Stitching ported from `stitching/stitching.Rmd`** to Python
  (`smint.segmentation.stitching`): merge, trim, remove conflicts, then
  reconcile cells with nuclei. Fixes the old stitcher deleting whole cells.
  Writes outline and per-object summary CSVs.
- `run_segmentation.py`: `--chunk-size` and `--use-gpu` override the config
  only when passed. With `--distributed` it now passes `--cell-model`,
  `--nuclei-model`, `--cell-diameter` and `--nuclei-diameter` on (they were
  ignored), and `--no-gpu` runs either mode on the CPU.
- `smint-segment-distributed` console script; `--slurm_gpu_type` adds to the
  configuration's SLURM options instead of replacing them.

### Packaging and documentation

- numpy requirement relaxed to `>=1.24`, so SMINT installs into STalign
  (numpy < 2) worker environments.
- The base install now includes `shapely` and `pyyaml`; the optional
  metabolomics import no longer warns when unused; Python >= 3.10.
- `integration/metabolomics_integration.R` runs on SMINT output
  (`x_transformed` / `y_transformed`) without edits to variable names.
- Installation instructions point to GitHub; `pip install smint` fetches an
  unrelated package.
- Sample and patient identifiers removed from code, scripts and
  documentation; samples are referred to as Sample 2, Sample 3 and Samples
  A–C.
- Documentation site rebuilt: tutorials with stored outputs, concept pages,
  task guides, command-line and API reference; readable diagrams in light and
  dark mode; citation of the *Cell Reports Methods* paper.

### August 2026: GPU type for registration jobs

- **Choose the GPU type for registration jobs.** `SlurmResources.gpu_type`
  (default `"A30"`) renders as `--gres=gpu:<type>:<n>`;
  `available_gpu_types(partition)` lists what the cluster offers, and an
  unavailable type is rejected before submission. The napari plugin gains a
  GPU-type dropdown.
- Fixed: GPU registrations launched from napari could fail with
  `Error in dlopen: libtorch_cuda_linalg.so`. Workers also no longer inherit
  `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP` or `LD_PRELOAD` from the
  submitting process.

### August 2026: registration overhaul

- **Registration overhaul.** `st_sm_registration` (STalign LDDMM around a
  manual landmark step), `centroid_registration` (affine / RANSAC / TPS with
  held-out TRE) and `pretransform` (scale, rotate, flip as one inspectable
  affine).
- **napari plugin** for loading datasets, pre-registration, landmarks and
  launching jobs, and an **HPC job runner** (`smint.alignment.jobs`) that runs
  registration in a separate worker process, locally or through sbatch.
- **`smint-alignment` command line** with `st-sm`, `centroid`,
  `pretransform` and `columns`, plus `--submit` and `--watch`.
- **Choose which coordinate columns to register**, everywhere, with
  auto-detection as a logged fallback. Invalid partitions are reported before
  submission, with the valid ones listed.
- **Breaking:** `st_align_wrapper` removed (its functions never worked). Use
  `register_sm_to_st` / `register_centroids`, `apply_affine` /
  `apply_pretransform` and `save_transform` / `load_transform`. The
  `smint-segmentation` console script was withdrawn.

### April 2026: 3D gallery

- Interactive 3D cell-type comparison of three samples added to the
  [Gallery](visualizations.md), with a sample selector, zoom buttons for touch
  screens and a toggleable legend.

## 1.0.0 (2025-12-18)

The first GitHub release, "v.1.0.0" (tag `smint`), prepared for Zenodo
archiving. The package metadata of that release still read 0.1.0.

### December 2025

- Repository prepared for Zenodo archiving; `CITATION.cff` added.

### May 2025

- SMINT restructured as a Python package (`smint.segmentation`,
  `smint.alignment`, `smint.preprocessing`, `smint.visualization`,
  `smint.r_integration`) with optional dependencies and a MkDocs site
  deployed to GitHub Pages.

### September – November 2024

- First public scripts: Cellpose training and segmentation notebook, CZI tile
  export, STalign alignment of Xenium and spatial metabolomics, R stitching
  and transcript assignment, metabolomics integration, and code to reproduce
  the paper figures.
