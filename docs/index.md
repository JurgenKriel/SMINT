---
hide:
  - navigation
  - toc
---

<div class="grid" markdown>

<div markdown>

# SMINT

**Spatial Multi-Omics Integration Toolkit.** SMINT segments cells and nuclei
across whole-slide images, then brings spatial transcriptomics, spatial
metabolomics and post-stain imaging into one coordinate system — interactively
in napari, from the command line, or as SLURM jobs on a GPU cluster.

```bash
pip install "git+https://github.com/JurgenKriel/SMINT.git"
```

To run the quick start and tutorials, clone the repository instead: they use
the example data in the checkout (see [Quick start](quickstart.md)).

[Get started](quickstart.md){ .md-button .md-button--primary }
[Tutorials](tutorials/index.md){ .md-button }

</div>

<p align="center" markdown>
![SMINT combines H&E, spatial transcriptomics and spatial metabolomics](assets/smint_logo_small.png){ width="380" }
</p>

</div>

## The workflow

```mermaid
flowchart LR
    A[Whole-slide image<br/>OME-TIFF / OME-Zarr] --> B[Segment cells & nuclei<br/>Cellpose 3/4, one GPU or many]
    B --> C[Stitch tiles<br/>reconcile cells & nuclei]
    C --> D{Same section<br/>as the Xenium run?}
    D -- yes --> E[Centroid registration<br/>multiround + snapping]
    D -- no --> F[Pre-register, landmarks<br/>STalign LDDMM]
    E --> G[One coordinate system:<br/>cells, transcripts, metabolites]
    F --> G
    G --> H[Integrate per cell<br/>transcripts + metabolites]
```

See [The SMINT workflow](concepts/workflow.md) for each step, and
[Choosing a registration method](concepts/registration_methods.md) for the
branch in the middle. The last step, one table with a row per cell, is shown in
[Tutorial 5](tutorials/05_integration.ipynb) and, in R, in
[R integration](r_integration.md).

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting started**

    ---

    Install SMINT and run a registration and a segmentation on example data
    in ten minutes.

    [:octicons-arrow-right-24: Quick start](quickstart.md)

-   :material-notebook-outline:{ .lg .middle } **Tutorials**

    ---

    Executed notebooks with every output shown: segmentation, same-section
    and sequential registration, distributed runs on SLURM.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-book-open-variant:{ .lg .middle } **User guide**

    ---

    How SMINT works: the workflow, coordinate frames, tiling and stitching,
    and how to choose a registration method.

    [:octicons-arrow-right-24: User guide](concepts/index.md)

-   :material-cursor-default-click-outline:{ .lg .middle } **napari plugin**

    ---

    Load datasets, pre-register by eye, place landmarks and launch
    registration jobs without leaving the viewer.

    [:octicons-arrow-right-24: napari plugin](napari_plugin.md)

-   :material-console:{ .lg .middle } **Reference**

    ---

    Command-line options, input and output formats, configuration keys and
    the Python API.

    [:octicons-arrow-right-24: Reference](reference/index.md)

-   :material-server-network:{ .lg .middle } **Running on HPC**

    ---

    Submit segmentation and registration to SLURM, on CPU or GPU, from
    Python, the command line or napari.

    [:octicons-arrow-right-24: Running on HPC](hpc_deployment.md)

</div>

## What it looks like

<div class="grid cards" markdown>

-   ![Cell and nucleus outlines on nuclei, GFAP and phalloidin](assets/landing/segmentation.png)

    **Segment cells and nuclei**

    Cells (white) and nuclei (yellow) on nuclei, GFAP and phalloidin.

    [:octicons-arrow-right-24: Segmentation](segmentation.md)

-   ![Post-stain nuclei placed onto Xenium cells](assets/landing/same_section.png)

    **Register the same section**

    Post-stain nuclei (pink) placed onto Xenium cells (blue rings).

    [:octicons-arrow-right-24: Same-section registration](tutorials/02_same_section_registration.ipynb)

-   ![MALDI pixels registered onto Xenium cells](assets/landing/sequential.png)

    **Register a neighbouring section**

    MALDI metabolomics (pink) on Xenium cells (blue) after STalign LDDMM.

    [:octicons-arrow-right-24: Sequential-section registration](tutorials/03_sequential_registration.ipynb)

</div>

The full before-and-after figures are in the [Gallery](visualizations.md).

## Citation

SMINT is described in [Kriel *et al.*, *Cell Reports Methods* (2026)](https://doi.org/10.1016/j.crmeth.2026.101306). If you use it in your research, please
[cite the paper](citation.md).
