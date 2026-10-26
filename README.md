
# SMINT - Spatial Multi-Omics Integration Toolkit
![Workflow Image]([https://github.com/JurgenKriel/SMINT/raw/main/smint_new_logo_2.png])

SMINT brings spatial omics modalities into one coordinate system. It segments cells and nuclei across whole-slide images, then registers spatial transcriptomics, spatial metabolomics and post-stain imaging onto each other — interactively in napari, from the command line, or as SLURM jobs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jurgenkriel.github.io/SMINT/)

## Features

- **Registration** of sequential sections with STalign LDDMM, and of post-stained same-section data by affine / RANSAC / thin-plate-spline correspondence fitting
- **napari plugin** for loading datasets, coarse pre-registration (scale, rotate, flip), landmark placement and launching registration jobs
- **`smint-alignment` CLI** for the same workflow from the shell, run locally or submitted to SLURM (CPU or GPU)
- **Honest metrics**: Target Registration Error reported on held-out pairs
- **Whole-slide segmentation** of cells and nuclei with Cellpose, with adaptive nuclei thresholds and distributed processing across GPUs with Dask
- **Live segmentation viewer** (`smint-viewer`)
- **R integration** for running existing R analysis scripts from Python
- **HPC deployment scripts** for SLURM-based clusters

## Documentation

Full documentation: [https://jurgenkriel.github.io/SMINT/](https://jurgenkriel.github.io/SMINT/)

## Installation

SMINT is not published on PyPI. The `smint` name there is an unrelated
placeholder, so `pip install smint` will **not** install this package. Install
from GitHub instead (Python 3.10+):

```bash
pip install "git+https://github.com/JurgenKriel/SMINT.git"
```

Or from a source checkout, for development:

```bash
git clone https://github.com/JurgenKriel/SMINT.git
cd SMINT
pip install -e .
```

Registration with STalign needs STalign itself, installed separately in an
environment with numpy < 2 (see the
[Alignment documentation](https://jurgenkriel.github.io/SMINT/alignment/#environments)):

```bash
pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
```

The napari plugin needs napari in the environment you launch it from:

```bash
pip install "napari[all]"
```

For more information, see the [Dependency Handling Documentation](https://jurgenkriel.github.io/SMINT/dependency_handling/).

## Quick Start

### Registration in napari

Launch `napari` and open **Plugins → SMINT Registration**. The four widgets —
*Load datasets*, *Pre-register*, *Landmarks* and *Register* — take you from raw
CSVs to a submitted registration job. See the
[napari plugin guide](https://jurgenkriel.github.io/SMINT/napari_plugin/).

### Registration from the command line

```bash
# Sequential sections: STalign LDDMM from saved landmarks
smint-alignment st-sm --st st.csv --sm sm.csv \
    --st-points st_points.npy --sm-points sm_points.npy --output-dir run1

# Same section: fit nuclei centroids onto Xenium cells
smint-alignment centroid --source nuclei.csv --target xenium.csv --output-dir run2

# Add --submit to run on SLURM instead, and --watch to wait for it
```

Or from Python:

```python
from smint.alignment import register_centroids

result = register_centroids(nucleus_xy, xenium_xy, method="ransac+tps")
print(result["tre_validation"])  # TRE on held-out pairs
```

See the [Alignment guide](https://jurgenkriel.github.io/SMINT/alignment/).

### Cell Segmentation

```python
from smint.segmentation import process_large_image

# Run cell segmentation on a whole-slide image
process_large_image(
    image_path="path/to/whole_slide_image.ome.tiff",
    csv_base_path="path/to/output/segmentation",  # writes <base>_cells.csv, <base>_nuclei.csv
    cell_model_path="cyto",  # Use pre-trained Cellpose model
    cells_diameter=120.0,
    cells_channels=[1, 2],  # Green and Blue channels for cells
    nuclei_model_path="nuclei",  # Use pre-trained nuclei model
    nuclei_diameter=60.0,
    nuclei_channels=[2, 1],  # Blue and Green channels for nuclei
    visualize=True  # Generate visualization images
)
```

See the [Segmentation guide](https://jurgenkriel.github.io/SMINT/segmentation/).

## Analysis Scripts

The repository also holds the analysis scripts used alongside the package.

### Transcriptomics workflow

1. **Stitch polygons** — `stitching/stitching.Rmd`
2. **Assign transcripts to polygons** — `assignment/transcript_assignment.R`

### Alignment workflow

1. **Set up STalign** — `pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"`
2. **Align ST to SM** — `alignment/ST_align_xenium.py`

### Integration

1. **Metabolomics cluster analysis** — `integration/metabolomics_integration.R`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

The documentation is built with MkDocs and deployed to GitHub Pages by
`.github/workflows/deploy-docs.yml` on every push to `main`. To preview it
locally:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```

## Citation

If you use SMINT in your research, please cite it using GitHub's **Cite this
repository** button (from `CITATION.cff`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
