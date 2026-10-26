# SMINT: Spatial Multi-Omics Integration

SMINT brings spatial omics modalities into one coordinate system. It segments
cells and nuclei across whole-slide images, then registers spatial
transcriptomics, spatial metabolomics and post-stain imaging onto each other —
interactively in napari, from the command line, or as SLURM jobs.

![SMINT Workflow](https://github.com/JurgenKriel/SMINT/raw/main/SpatialSegPaper_v2.png)

## Install

SMINT is installed from GitHub (it is not on PyPI). Python 3.10+:

```bash
pip install "git+https://github.com/JurgenKriel/SMINT.git"
```

STalign, used for sequential-section registration, and napari, for the
interactive plugin, are installed separately — see [Installation](installation.md).

## Registration

Two registration regimes, chosen by how the modalities were acquired:

| Situation | Method |
|---|---|
| **Sequential sections** — modalities from different physical sections | STalign LDDMM, guided by manual landmarks |
| **Same section, post-staining** — a second modality on the exact same section | Correspondence fitting: affine, RANSAC or thin-plate spline |

- **napari plugin** — load datasets, pre-register by eye (scale, rotate, flip),
  place paired landmarks, and launch registration without leaving the viewer.
  Open **Plugins → SMINT Registration**. See [napari Plugin](napari_plugin.md).
- **`smint-alignment` CLI** — the same machinery from the shell:

    ```bash
    # Sequential sections: STalign LDDMM from saved landmarks
    smint-alignment st-sm --st st.csv --sm sm.csv \
        --st-points st_points.npy --sm-points sm_points.npy --output-dir run1

    # Same section: fit nuclei centroids onto Xenium cells
    smint-alignment centroid --source nuclei.csv --target xenium.csv --output-dir run2

    # Coarse scale / rotate / flip, searching orientations by overlap
    smint-alignment pretransform --source sm.csv --target xenium.csv --search --output sm_pre.csv
    ```

- **Honest metrics** — Target Registration Error is reported on held-out pairs,
  so a flexible transform cannot flatter itself by memorising correspondences.
- **HPC native** — add `--submit` to run on SLURM (CPU or GPU) instead of
  locally; napari submits the same jobs and polls them in the background.

See [Alignment](alignment.md) for the full workflow.

## Segmentation

- **Dual-model segmentation** — cells and nuclei with Cellpose, in chunks across
  whole-slide images
- **Adaptive nuclei** — retries nuclear segmentation with relaxed thresholds
  when too few nuclei are found
- **Distributed processing** — tiles spread across Dask workers on multiple GPUs
- **Live monitoring** — watch segmentation progress tile by tile with
  `smint-viewer`

See [Segmentation](segmentation.md).

## Also in SMINT

- **R integration** — run existing R analysis scripts from Python
  ([R Integration](r_integration.md))
- **Visualization** — segmentation overlays, outline plots and RGB composites
- **HPC** — SLURM submission scripts for preprocessing and segmentation
  ([HPC Deployment](hpc_deployment.md))
- **Interactive 3D plots** — multi-omics data from the BCRL patient cohort
  ([Visualizations](visualizations.md))

## Documentation

- [Installation](installation.md) · [Configuration](configuration.md) ·
  [Dependency Handling](dependency_handling.md)
- [Alignment](alignment.md) · [napari Plugin](napari_plugin.md) ·
  [Xenium–Metabolomics](xenium_metabolomics_alignment.md)
- [Segmentation](segmentation.md) · [Examples](examples.md)
- [R Integration](r_integration.md) · [HPC Deployment](hpc_deployment.md)

## Citation

If you use SMINT in your research, please cite:

> Kriel, J. *SMINT — Spatial Multi-Omics Integration Toolkit* (v0.1.0).
> [https://github.com/JurgenKriel/SMINT](https://github.com/JurgenKriel/SMINT)

```bibtex
@software{kriel_smint,
  author  = {Kriel, Jurgen},
  title   = {SMINT - Spatial Multi-Omics Integration Toolkit},
  version = {0.1.0},
  url     = {https://github.com/JurgenKriel/SMINT},
  license = {MIT}
}
```

GitHub's **Cite this repository** button provides the same citation from
`CITATION.cff`.
