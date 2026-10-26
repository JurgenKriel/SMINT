# Changelog

User-visible changes to SMINT, newest first, compiled from the
[commit history](https://github.com/JurgenKriel/SMINT/commits/main). The
package version is still 0.1.0, so changes are grouped by date.

## 2026-10-26

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
  only when passed.

### Packaging and documentation

- numpy requirement relaxed to `>=1.24`, so SMINT installs into STalign
  (numpy < 2) worker environments.
- Installation instructions point to GitHub; `pip install smint` fetches an
  unrelated package.
- Sample and patient identifiers removed from code, scripts and
  documentation; samples are referred to as Sample 2, Sample 3 and Samples
  A–C.
- Documentation site rebuilt: tutorials with stored outputs, concept pages,
  task guides, command-line and API reference.

## 2026-08-19

- **Choose the GPU type for registration jobs.** `SlurmResources.gpu_type`
  (default `"A30"`) renders as `--gres=gpu:<type>:<n>`;
  `available_gpu_types(partition)` lists what the cluster offers, and an
  unavailable type is rejected before submission. The napari plugin gains a
  GPU-type dropdown.
- Fixed: GPU registrations launched from napari could fail with
  `Error in dlopen: libtorch_cuda_linalg.so`. Workers also no longer inherit
  `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP` or `LD_PRELOAD` from the
  submitting process.

## 2026-08-17

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

## 2026-04-20

- Interactive 3D cell-type comparison of three samples added to the
  [Gallery](visualizations.md), with a sample selector, zoom buttons for touch
  screens and a toggleable legend.

## 2025-12-18

- Repository prepared for Zenodo archiving; `CITATION.cff` added.

## 2025-05-13 – 2025-05-14

- SMINT restructured as a Python package (`smint.segmentation`,
  `smint.alignment`, `smint.preprocessing`, `smint.visualization`,
  `smint.r_integration`) with optional dependencies and a MkDocs site
  deployed to GitHub Pages.

## 2024-09 – 2024-11

- First public scripts: Cellpose training and segmentation notebook, CZI tile
  export, STalign alignment of Xenium and spatial metabolomics, R stitching
  and transcript assignment, metabolomics integration, and code to reproduce
  the paper figures.
