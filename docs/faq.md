# FAQ and troubleshooting

Short answers to the questions that come up most, with links to the page that
covers each in full. Problems specific to one task are also listed at the end
of the [Segmentation](segmentation.md#troubleshooting),
[Alignment](alignment.md#troubleshooting) and
[HPC](hpc_deployment.md#troubleshooting) guides.

## Getting started

??? question "`pip install smint` installed something else"
    SMINT is not on PyPI; the `smint` name there is an unrelated placeholder.
    Install from GitHub:
    `pip install "git+https://github.com/JurgenKriel/SMINT.git"`.
    See [Installation](installation.md).

??? question "Why do I need more than one Python environment?"
    STalign needs numpy < 2, while napari and recent Cellpose builds use
    numpy 2. SMINT runs LDDMM in a separate worker process so that the two
    never meet. See [Which environment do I need?](installation.md#which-environment-do-i-need)

??? question "Importing SMINT prints a warning about STalign or rpy2"
    Both are optional. The warning means LDDMM registration (STalign) or the
    in-process R helpers (rpy2) are unavailable in this environment;
    everything else works. See
    [Missing optional dependencies](installation.md#missing-optional-dependencies).

??? question "Do I need a GPU?"
    No. Segmentation and LDDMM run on CPU, only more slowly; centroid
    registration and pre-registration do not use a GPU at all.

??? question "Which data does SMINT accept?"
    Images as TIFF / OME-TIFF (and Zarr / OME-Zarr for distributed
    segmentation); everything else as CSV tables with one row per cell,
    nucleus or pixel and an x/y column pair. SMINT was developed on Xenium
    spatial transcriptomics, MALDI spatial metabolomics and fluorescence
    post-stains. See [Data formats](reference/data_formats.md).

## Choosing a method

??? question "Which registration method should I use?"
    If both datasets come from the **same physical section** (a post-stain of
    the Xenium slide), use centroid registration. If they come from
    **neighbouring sections**, use pre-registration, landmarks and STalign
    LDDMM. See [Choosing a registration method](concepts/registration_methods.md)
    and [Pick the path for your data](alignment.md#pick-the-path-for-your-data).

??? question "Should I use `process_large_image` or `run_distributed_segmentation`?"
    `run_distributed_segmentation` in most cases: it reads the image lazily,
    stitches across tile boundaries and assigns each nucleus to a cell, and
    it runs locally as well as on SLURM. See
    [Choose an entry point](segmentation.md#choose-an-entry-point).

??? question "Cellpose 3 or Cellpose 4?"
    Cellpose 4 if you use Cellpose-SAM or a fine-tuned `cpsam` model;
    Cellpose 3 if you want the classic `cyto3` / `nuclei` models. On Cellpose
    4 those names become `cpsam`, with only a warning in the log. See
    [Cellpose 3 or Cellpose 4](installation.md#cellpose-3-or-cellpose-4).

## Interpreting results

??? question "My registration reports a TRE of almost zero. Is that good?"
    Look at `tre_validation`, not `tre_fit`. A flexible transform can drive
    the error on the pairs it was fitted to towards zero whether or not those
    pairs are right; only held-out data tells you how good the registration
    is. See [Check the result](alignment.md#check-the-result).

??? question "The registered SM data looks mirrored or rotated over my ST data"
    Register with `orientation="stalign"` (`--orientation stalign`). The
    default, `"dataset"`, keeps the transposed layout of earlier pipeline
    runs. See [Coordinate frames](concepts/coordinate_frames.md).

??? question "Some nuclei are `assigned` rather than `snapped`. Can I trust them?"
    Not without checking. `assigned` nuclei did not pass the neighbourhood
    consistency check and many sit on a neighbouring cell. Filter on
    `registration_status` and `neighbour_deviation`, or turn assignment off
    with `--no-assign-remaining`. See
    [Snapping nuclei onto their cells](alignment.md#snapping-nuclei-onto-their-cells).

??? question "Why are my segmentation coordinates so much larger than the Xenium ones?"
    Segmentation outputs are in image **pixels**; Xenium uses **µm**. Multiply
    by your pixel size before registering. See
    [Use the results for registration](segmentation.md#use-the-results-for-registration).

??? question "Registration ran but the result is in the wrong place"
    Check which coordinate columns were used: tables often carry several
    pairs, and the wrong pair misregisters without an error. Run
    `smint-alignment columns your.csv` and pass the columns explicitly. See
    [Check your coordinate columns](alignment.md#before-you-start-check-your-coordinate-columns).

??? question "How many landmarks do I need for LDDMM?"
    Enough to pin the whole section, spread to the edges, on features that
    are sharp in both modalities. The Sample 2 example used 11. They must be
    placed in the same order in both datasets.

## Running jobs

??? question "My SLURM job failed and wrote no logs"
    Something it needs is on node-local storage (`/tmp`, `/dev/shm`, …),
    usually `output_dir`. Everything must be on storage the compute nodes
    share. See [The shared-storage rule](hpc_deployment.md#the-shared-storage-rule).

??? question "My GPU job never starts"
    You may have asked for a GPU on a CPU partition, or for a GPU type that
    is busy or absent. Check `available_gpu_types(partition)`, or use
    `gpu_type=None` to accept any card. See
    [Choose a partition and a GPU](hpc_deployment.md#choose-a-partition-and-a-gpu).

??? question "`RuntimeError: Nanny failed to start` in distributed segmentation"
    The local Dask backend starts worker processes, which needs your script's
    code under `if __name__ == "__main__":`. See
    [Segment a whole slide with Dask](segmentation.md#segment-a-whole-slide-with-dask).

??? question "How do I follow a registration job?"
    Use `--watch` on the command line, `poll(output_dir)` in Python, or read
    `status.json` in the output directory. See
    [Register on SLURM](hpc_deployment.md#register-on-slurm).

## Still stuck?

Open an issue on [GitHub](https://github.com/JurgenKriel/SMINT/issues) with
the command you ran, the full error, and the log from the output directory
(`segmentation.log`, `distributed_seg_*.log`, `worker.log` or the SLURM
`.err` file).
