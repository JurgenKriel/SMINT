
# SMINT - Spatial Multi-Omics Integration Toolkit
![Workflow Image](https://github.com/JurgenKriel/SMINT/raw/main/SpatialSegPaper_v2.png)

SMINT is a Python package for Spatial Multi-Omics Integration with enhanced segmentation capabilities and streamlined workflow. It provides a comprehensive toolkit for processing, analyzing, and visualizing spatial multi-omics data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jurgenkriel.github.io/SMINT/)

## Features

- Multi-GPU cell segmentation for whole-slide images
- Distributed segmentation using Dask for improved performance
- Live segmentation monitoring with intuitive visualization tools
- Streamlined alignment workflow using ST Align
- Integration with R analysis scripts via rpy2
- Graceful dependency handling for all optional components
- Comprehensive documentation with step-by-step guides
- HPC deployment scripts for SLURM-based clusters

## Documentation

Comprehensive documentation is available at [https://jurgenkriel.github.io/SMINT/](https://jurgenkriel.github.io/SMINT/)

## Installation

```bash
pip install smint
```

## Optional Dependencies

SMINT is designed with graceful dependency handling to ensure robust operation across different environments.

```bash
# Full installation with all dependencies
pip install "smint[all]"

# Install specific feature sets
pip install "smint[segmentation]"  # For segmentation functionality
pip install "smint[visualization]"  # For visualization functionality
pip install "smint[r_integration]"  # For R integration
```

For more information, see the [Dependency Handling Documentation](https://jurgenkriel.github.io/SMINT/dependency_handling/).

## Linking to GitHub Repository

This project is linked to the GitHub repository at [https://github.com/JurgenKriel/SMINT](https://github.com/JurgenKriel/SMINT).

### Pushing to GitHub

To contribute changes to the GitHub repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/JurgenKriel/SMINT.git
   cd SMINT
   ```

2. Make your changes and test them locally.

3. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

### Documentation Deployment

Documentation is automatically deployed to GitHub Pages using GitHub Actions. The workflow file is located at `.github/workflows/deploy-docs.yml`.

When you push changes to the `main` branch, the documentation will be automatically built and deployed to [https://jurgenkriel.github.io/SMINT/](https://jurgenkriel.github.io/SMINT/).

To build documentation locally for testing:

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings
mkdocs serve
```

## Usage Examples

### Cell Segmentation

```python
import smint
from smint.segmentation import process_large_image

# Run cell segmentation on a whole-slide image
process_large_image(
    image_path="path/to/whole_slide_image.ome.tiff",
    csv_base_path="path/to/output_directory",
    cell_model_path="cyto",  # Use pre-trained Cellpose model
    cells_diameter=120.0,
    cells_channels=[1, 2],  # Green and Blue channels for cells
    nuclei_model_path="nuclei",  # Use pre-trained nuclei model
    nuclei_diameter=60.0,
    nuclei_channels=[2, 1],  # Blue and Green channels for nuclei
    visualize=True  # Generate visualization images
)
```

### Visualization

```python
from smint.visualization import visualize_segmentation_results

# Create interactive visualization of segmentation results
visualize_segmentation_results(
    image_path="path/to/whole_slide_image.ome.tiff",
    segmentation_csv="path/to/segmentation_results.csv",
    output_html="path/to/output/visualization.html",
    thumbnail_size=(800, 800)
)
```

### R Integration

```python
from smint.r_integration import run_spatialomics_analysis

# Run spatial statistics analysis using R
run_spatialomics_analysis(
    segmentation_data="path/to/segmentation_results.csv",
    expression_data="path/to/gene_expression.csv",
    output_directory="path/to/output",
    spatial_methods=["Ripley", "Morans"]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Transcriptomics workflow  

### 1. Stitch Polygons
See stitching script under stitching/stitching.Rmd

### 2. Align and Assign Transcripts to polygons
See alignment of polygons and transcripts under
See transcript assignment script under assignment/transcript_assignment.R

## Alignment Workflow

### 1. Setup STalign

```
pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
```

### 2. Align ST to SM
See alignment script under alignment/ST_align_xenium.py


## Integration 

### 1. Metabolomics Cluster Analysis 
See metabolite assignment script under integration/metabolomics_integration.R
>>>>>>> e9fadf4d662d283dafd59ae756781bbe1e40863c
