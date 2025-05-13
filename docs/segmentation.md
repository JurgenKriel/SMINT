# Cell Segmentation

SMINT provides powerful tools for segmenting cells and nuclei in whole-slide images, with support for distributed processing and multiple GPUs.

## Quick Start

### Single-Process Segmentation

```bash
python -m scripts.run_segmentation \
    --image path/to/image.tif \
    --output-dir results/segmentation \
    --visualize
