# Installation

This guide covers the installation of SMINT and its dependencies.

## Prerequisites

SMINT requires the following:

- Python 3.8 or higher
- CUDA toolkit (for GPU support, optional)
- R 4.0 or higher (for R integration, optional)

### System Requirements

- **Memory**: At least 16GB RAM, 32GB+ recommended for large images
- **CPU**: Multi-core processor (8+ cores recommended)
- **GPU**: NVIDIA GPU with at least 8GB VRAM (for GPU-accelerated segmentation, optional)
- **Storage**: Depends on your image sizes, but typically 100GB+ free space

## Installation Methods

SMINT is not published on PyPI — the `smint` name there is an unrelated
placeholder, so `pip install smint` will not install this package. Install from
GitHub instead.

### Option 1: Install from GitHub

```bash
pip install "git+https://github.com/JurgenKriel/SMINT.git"
```

### Option 2: Install from Source

For development or to contribute to the project:

```bash
git clone https://github.com/JurgenKriel/SMINT.git
cd SMINT
pip install -e .
```

## Dependencies

SMINT is designed with a modular approach to dependencies. Core functionality will work with minimal dependencies, while optional features require additional packages.

### Core Dependencies (Automatically Installed)

These dependencies are automatically installed when you install SMINT:

- **numpy**: For numerical operations
- **pandas**: For data manipulation
- **matplotlib**: For visualization
- **dask**: For parallel computing (base package)
- **scikit-image**: For image processing
- **tifffile**: For reading/writing OME-TIFF files
- **cellpose** (3 or 4): For cell segmentation models
- **opencv-python**: For image processing and contour extraction

!!! note "Cellpose 3 and 4"
    SMINT works with Cellpose 3 and Cellpose 4. Models are given as a built-in
    name or a path to a trained model. Cellpose 3 has the `cyto`, `cyto2`,
    `cyto3` and `nuclei` models; Cellpose 4 has only Cellpose-SAM (`cpsam`), so
    on Cellpose 4 those names fall back to `cpsam` with a warning — pass a
    fine-tuned model path instead. `channels` (`[main, nuclear]`, 1-based, 0 =
    none) selects the input channels identically on both versions.

### Optional Dependencies

These dependencies enable additional features but are not required for core functionality:

#### Distributed Processing

- **smint[distributed]**: Dask distributed, dask-jobqueue (SLURM workers),
  zarr and shapely, for `run_distributed_segmentation`
  ```bash
  pip install "smint[distributed] @ git+https://github.com/JurgenKriel/SMINT.git"
  ```

- **dask-cuda**: One local worker per GPU (requires CUDA)
  ```bash
  pip install dask-cuda
  ```

#### R Integration

- **rpy2**: For direct R integration
  ```bash
  pip install rpy2
  ```

### Installation with All Optional Dependencies

To install SMINT with all optional dependencies:

```bash
pip install "smint[all] @ git+https://github.com/JurgenKriel/SMINT.git"
```

Or selectively, from a source checkout:

```bash
pip install -e ".[r_integration]"  # For R integration features
pip install -e ".[docs]"           # To build this documentation
pip install -e ".[dev]"            # pytest, flake8, black
```

Registration with STalign also needs STalign itself, which is installed
separately and requires numpy < 2 (see [Alignment](alignment.md#environments)):

```bash
pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
```

