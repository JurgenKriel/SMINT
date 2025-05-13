# Spatial Alignment

SMINT provides a streamlined workflow for aligning different types of spatial omics data using the ST Align tool.

## Quick Start

### Basic Alignment

```bash
python -m scripts.run_alignment \
    --reference reference_data.csv \
    --target target_data.csv \
    --output-dir results/alignment \
    --method affine
