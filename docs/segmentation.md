# Cell Segmentation

SMINT provides powerful tools for segmenting cells and nuclei in whole-slide images, with support for distributed processing and multiple GPUs. This comprehensive guide covers all aspects of the segmentation pipeline, from data preparation to post-processing.

## Overview of Segmentation Pipeline

The SMINT segmentation pipeline consists of several key stages:

1. **Image Preprocessing** - Preparing and normalizing input images
2. **Cell Segmentation** - Identifying whole cells using Cellpose
3. **Nuclei Segmentation** - Identifying cell nuclei (optional)
4. **Post-processing** - Refining segmentation results and extracting features
5. **Visualization** - Generating visual outputs for quality control

## Input Data Requirements

### Supported Image Formats

SMINT supports the following image formats:
- **OME-TIFF** (preferred) - Multi-channel, multi-resolution format with metadata
- **TIFF** - Standard format, single or multi-channel
- **CZI** - Carl Zeiss format; export tiles first (see `segmentation/export_czi_tile_coordinates.ipynb`)

### Required Image Properties

For optimal segmentation results, input images should have:
- **Resolution**: Ideally 0.5-1 μm/pixel for cell segmentation, 0.25-0.5 μm/pixel for nuclei
- **Channels**: 
  - At least one membrane/cytoplasm channel (e.g., WGA, phalloidin)
  - At least one nuclear channel (e.g., DAPI, Hoechst) for nuclei segmentation
- **Bit depth**: 8-bit or 16-bit grayscale per channel
- **Quality**: Minimal noise, good contrast between cells and background

!!! note "Cellpose 3 is required"
    SMINT's segmentation requires **Cellpose 3** (`cellpose>=3,<4`), which is
    installed automatically. Cellpose 4 ships only its SAM-based `cpsam` model and
    ignores the `cyto`/`nuclei` model types and `channels` this pipeline uses, so
    don't upgrade it in a SMINT environment.

## Quick Start

Segmentation is run with `scripts/run_segmentation.py` from a source checkout.
Model parameters (thresholds, channels, adaptive nuclei settings) come from a
YAML config — see [Configuration](configuration.md) and
`configs/segmentation_config.yaml`.

### Single-Process Segmentation

```bash
python scripts/run_segmentation.py \
    --image path/to/image.ome.tiff \
    --output-dir results/segmentation \
    --config configs/segmentation_config.yaml \
    --cell-diameter 60 \
    --nuclei-diameter 30 \
    --visualize
```

### Distributed Segmentation

Add `--distributed` to process chunks in parallel with Dask, and `--use-gpu`
to run on the available GPUs:

```bash
python scripts/run_segmentation.py \
    --image path/to/large_image.ome.tiff \
    --output-dir results/segmentation \
    --config configs/segmentation_config.yaml \
    --distributed \
    --use-gpu \
    --chunk-size 2048 2048
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to input image | Required |
| `--output-dir` | Directory to save results | Required |
| `--config` | YAML configuration file | Built-in defaults |
| `--distributed` | Run in distributed mode using Dask | Off |
| `--use-gpu` | Use GPU for computation (distributed mode; single-process uses the GPU when Cellpose finds one) | Off |
| `--cell-model` | Cell model: built-in name (e.g. `cyto`) or path to a custom model | From config |
| `--nuclei-model` | Nuclei model: built-in name (e.g. `nuclei`) or path to a custom model | From config |
| `--cell-diameter` | Expected cell diameter in pixels | From config |
| `--nuclei-diameter` | Expected nuclei diameter in pixels | From config |
| `--chunk-size` | Chunk height and width | `2048 2048` |
| `--visualize` | Generate visualizations | Off |
| `--live-viewer` | Write live-view images for the live viewer | Off |
| `--save-config` | Save the effective configuration to the output directory | Off |
| `--debug` | Enable debug logging | Off |

## Segmentation API

For programmatic usage within Python scripts:

```python
from smint.segmentation import process_large_image

# Basic usage
results = process_large_image(
    image_path="path/to/image.ome.tiff",
    csv_base_path="results/segmentation",
    chunk_size=(2048, 2048),
    # Cell parameters
    cell_model_path="cyto",
    cells_diameter=80.0,
    cells_flow_threshold=0.4,
    cells_cellprob_threshold=-1.5,
    cells_channels=[0, 0],  # [channel, 0] for grayscale, [channel1, channel2] for RGB
    # Nuclei parameters
    nuclei_model_path="nuclei",
    nuclei_diameter=40.0,
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=-1.5,
    nuclei_channels=[1, 0],  # [channel, 0] for grayscale
    # Visualization
    visualize=True,
    visualize_output_dir="results/visualization",
    num_visualize_chunks=5,
    visualize_roi_size=(512, 512)
)

# Access the results
print(results["total_cells"], results["total_nuclei"])
cells_csv = results["cells_csv_path"]     # <csv_base_path>_cells.csv
nuclei_csv = results["nuclei_csv_path"]   # <csv_base_path>_nuclei.csv
```

For distributed processing:

```python
from smint.segmentation import run_distributed_segmentation

results = run_distributed_segmentation(
    image_path="path/to/large_image.ome.tiff",
    output_dir="results/segmentation",
    config_path="configs/segmentation_config.yaml",
    n_workers=None,          # None = use all available workers
    use_gpu=True,
    chunk_size=(2048, 2048),
)
```

## Advanced Adaptive Segmentation

SMINT supports adaptive segmentation that dynamically adjusts parameters based on local image characteristics:

```python
from smint.segmentation import process_large_image

results = process_large_image(
    # Basic parameters as above, plus:
    enable_adaptive_nuclei=True,
    nuclei_adaptive_cellprob_lower_limit=-6.0,
    nuclei_adaptive_cellprob_step_decrement=0.2,
    nuclei_max_adaptive_attempts=3,
    adaptive_nuclei_trigger_ratio=0.05  # Retry if nuclei count < 5% of cells
)
```

## Output Files and Formats

`process_large_image` writes:

| File | Description | Format |
|------|-------------|--------|
| `<csv_base_path>_cells.csv` | Cell outline coordinates | CSV with `global_cell_id`, `x`, `y` |
| `<csv_base_path>_nuclei.csv` | Nuclei outline coordinates | CSV with `global_cell_id`, `x`, `y` |
| `chunk_<y>_<x>.png` | Per-chunk visualizations (with `visualize=True`) | PNG |
| `live_update.png` | Latest chunk, for the live viewer | PNG |

`run_distributed_segmentation` writes per-chunk `outlines_<y>_<x>_<model>.csv`
files and merges them into `combined_cell_outlines.csv`.

## Live Segmentation Viewer

SMINT includes a Tkinter viewer for monitoring segmentation in real time. From
the command line:

```bash
smint-viewer path/to/image.ome.tiff results/segmentation results/segmentation/current_tile_info.txt --interval 1000
```

Or from Python:

```python
from smint.visualization import run_viewer

run_viewer(
    full_scan_path="path/to/image.ome.tiff",
    segmentation_history_dir="results/segmentation",
    tile_info_path="results/segmentation/current_tile_info.txt",
    update_interval_ms=1000,  # Update every 1 second
)
```

## Common Issues and Troubleshooting

### Poor Segmentation Quality

- **Problem**: Cells or nuclei not properly detected
- **Solution**: Adjust diameter, flow_threshold, and cellprob_threshold. Try using larger diameter for bigger cells, lower flow_threshold for weakly stained cells.

### Memory Errors

- **Problem**: "CUDA out of memory" or other memory-related errors
- **Solution**: Reduce chunk_size, or use distributed processing with multiple GPUs.

### Processing Speed

- **Problem**: Segmentation taking too long
- **Solution**: Use multi-GPU processing, reduce visualization, skip nuclei segmentation if not needed.

### Boundary Artifacts

- **Problem**: Cell masks cut off at chunk boundaries
- **Solution**: Use larger chunks so fewer cells straddle a boundary.

## Performance Benchmarks

SMINT segmentation performance on different hardware configurations:

| Image Size | Hardware | Processing Time | Memory Usage |
|------------|----------|-----------------|--------------|
| 10k × 10k | Single GPU (RTX 3090) | ~5 minutes | ~8 GB VRAM |
| 50k × 50k | Single GPU (RTX 3090) | ~1 hour | ~10 GB VRAM |
| 50k × 50k | 4× GPUs (RTX 3090) | ~15 minutes | ~8 GB VRAM per GPU |
| 100k × 100k | 4× GPUs (RTX 3090) | ~1 hour | ~8 GB VRAM per GPU |

## Tips for Best Results

1. **Image Quality**: Start with high-quality, well-stained images for best results
2. **Parameter Tuning**: Optimize cell_diameter, flow_threshold, and cellprob_threshold for your specific images
3. **Chunk Size**: Balance between processing speed (larger chunks) and memory usage (smaller chunks)
4. **Custom Models**: Train custom Cellpose models for specialized cell types
5. **Channel Selection**: Choose channels with strongest cell/nuclei signal for segmentation
6. **Validation**: Always validate segmentation quality with visualizations
7. **Adaptive Approach**: Use adaptive parameters for images with varying intensity or cell density
