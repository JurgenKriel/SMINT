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

### Option 1: Install from PyPI (Recommended)

The simplest way to install SMINT is using pip:

```bash
pip install smint
```

### Option 2: Install from Source

For the latest development version or to contribute to the project:

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

### Optional Dependencies

These dependencies enable additional features but are not required for core functionality:

#### Cell Segmentation

- **cellpose**: For cell segmentation models
  ```bash
  pip install cellpose
  ```

- **opencv-python**: For image processing and contour extraction
  ```bash
  pip install opencv-python
  ```

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
pip install "smint[all]"
```

Or selectively:

```bash
pip install "smint[segmentation]"  # For cell segmentation features
pip install "smint[distributed]"   # For distributed computing features
pip install "smint[r]"             # For R integration features
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
