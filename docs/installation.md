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
- **cellpose** (3.x): For cell segmentation models
- **opencv-python**: For image processing and contour extraction

!!! note "Cellpose 3 is required"
    SMINT's segmentation requires **Cellpose 3** (`cellpose>=3,<4`), which is
    installed automatically. Cellpose 4 ships only its SAM-based `cpsam` model and
    ignores the `cyto`/`nuclei` model types and `channels` this pipeline uses, so
    don't upgrade it in a SMINT environment.

### Optional Dependencies

These dependencies enable additional features but are not required for core functionality:

#### Distributed Processing

- **dask[distributed]**: For distributed computing
  ```bash
  pip install "dask[distributed]"
  ```

- **dask-cuda**: For GPU-accelerated distributed computing (requires CUDA)
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

## Graceful Dependency Handling

SMINT is designed to degrade gracefully when optional dependencies are missing:

1. **Missing a dependency?** SMINT will log a warning but continue to operate with limited functionality.
2. **Function that requires a missing dependency?** You'll get a clear error message explaining which package you need to install.
3. **Want to check available functionality?** Run the following code:

```python
import smint
print(f"SMINT package version: {smint.__version__}")
print("Available modules:")
[print(f"- {module}") for module in smint.__all__]
```

You can also check specific dependency availability:

```python
# Check cellpose availability
try:
    import cellpose
    print("Cellpose is available")
except ImportError:
    print("Cellpose is not available")
```
