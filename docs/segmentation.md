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
- **CZI** - Carl Zeiss format, requires additional processing

### Required Image Properties

For optimal segmentation results, input images should have:
- **Resolution**: Ideally 0.5-1 μm/pixel for cell segmentation, 0.25-0.5 μm/pixel for nuclei
- **Channels**: 
  - At least one membrane/cytoplasm channel (e.g., WGA, phalloidin)
  - At least one nuclear channel (e.g., DAPI, Hoechst) for nuclei segmentation
- **Bit depth**: 8-bit or 16-bit grayscale per channel
- **Quality**: Minimal noise, good contrast between cells and background

## Quick Start

### Single-Process Segmentation

For standard whole-slide images that fit in memory:

```bash
python -m scripts.run_segmentation \
    --image path/to/image.ome.tiff \
    --output-dir results/segmentation \
    --cell-channel 0 \
    --nuclei-channel 1 \
    --cell-diameter 60 \
    --nuclei-diameter 30 \
    --visualize
```

### Multi-GPU Distributed Segmentation

For very large images requiring multiple GPUs:

```bash
python -m scripts.run_distributed_segmentation \
    --image path/to/large_image.ome.tiff \
    --output-dir results/segmentation \
    --cell-channel 0 \
    --nuclei-channel 1 \
    --cell-diameter 60 \
    --nuclei-diameter 30 \
    --chunk-size 2048 2048 \
    --gpus 0 1 2 3 \
    --visualize
```

## Detailed Parameter Guide

### Common Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `--image` | Path to input image | Required | - |
| `--output-dir` | Directory to save results | Required | - |
| `--cell-channel` | Channel index for cell segmentation | 0 | Depends on staining |
| `--nuclei-channel` | Channel index for nuclei segmentation | 1 | Depends on staining |
| `--cell-diameter` | Expected cell diameter in pixels | 80 | 40-120 |
| `--nuclei-diameter` | Expected nuclei diameter in pixels | 40 | 20-60 |
| `--flow-threshold` | Flow threshold for Cellpose | 0.4 | 0.2-0.8 |
| `--cellprob-threshold` | Cell probability threshold | -1.0 | -3.0-0.0 |
| `--visualize` | Generate visualizations | False | - |
| `--chunk-size` | Size of image chunks to process | 2048 2048 | Depends on GPU memory |

### Advanced Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--pretrained-model` | Path to custom Cellpose model | None | Use for specialized cell types |
| `--model-type` | Cellpose model type | "cyto" | Options: "cyto", "nuclei", "cyto2", or custom path |
| `--min-cell-size` | Minimum cell size in pixels | 15 | Filters out small objects |
| `--omit-overlap` | Remove overlapping cell masks | False | Useful for densely packed cells |
| `--adaptive-threshold` | Use adaptive thresholding | False | Helps with variable image intensity |
| `--normalize-channels` | Normalize channel intensities | True | Improves segmentation quality |
| `--save-zarr` | Save results in Zarr format | False | Useful for very large datasets |
| `--no-nuclei` | Skip nuclei segmentation | False | Speeds up processing |

### Distributed Processing Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--gpus` | GPU device IDs to use | None | Space-separated list of GPU IDs |
| `--workers-per-gpu` | Dask workers per GPU | 1 | Increase for CPU-bound tasks |
| `--memory-limit` | Memory limit per worker | "10GB" | Adjust based on available RAM |
| `--scheduler-address` | Dask scheduler address | None | For connecting to existing cluster |
| `--overlap` | Overlap between chunks | 64 | Prevents boundary artifacts |
| `--batch-size` | Number of chunks per batch | 4 | Adjust based on GPU memory |

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
cell_outlines = results["cell_outlines"]
nuclei_outlines = results["nuclei_outlines"]
```

For distributed processing:

```python
from smint.segmentation.distributed_seg import process_large_image_distributed

# Basic distributed usage
results = process_large_image_distributed(
    image_path="path/to/large_image.ome.tiff",
    output_zarr_path="results/segmentation.zarr",
    csv_path="results/segmentation.csv",
    blocksize=(2048, 2048),
    channel=0,  # Main channel for segmentation
    gpus=[0, 1, 2, 3],  # List of GPU IDs
    overlap=64,  # Overlap between chunks
    batch_size=4,  # Number of chunks per batch
    model_type="cyto",  # Cellpose model type
    diameter=80.0,  # Expected cell diameter
    flow_threshold=0.4,
    cellprob_threshold=-1.5
)
```

## Advanced Adaptive Segmentation

SMINT supports adaptive segmentation that dynamically adjusts parameters based on local image characteristics:

```python
from smint.segmentation import process_large_image

results = process_large_image(
    # Basic parameters as above, plus:
    enable_adaptive_nuclei=True,
    nuclei_adaptive_flow_min=0.1,
    nuclei_adaptive_flow_step_decrement=0.1,
    nuclei_max_adaptive_attempts=5,
    adaptive_nuclei_trigger_ratio=0.05  # Retry if nuclei count < 5% of cells
)
```

## Output Files and Formats

SMINT generates the following output files:

| File | Description | Format |
|------|-------------|--------|
| `cells_outlines.csv` | Cell outline coordinates | CSV with columns: `cell_id,x,y,chunk_id` |
| `nuclei_outlines.csv` | Nuclei outline coordinates | CSV with columns: `nuclei_id,x,y,chunk_id` |
| `cell_features.csv` | Extracted cell features | CSV with measurements for each cell |
| `segmentation_metadata.json` | Segmentation parameters and stats | JSON |
| `visualization/*.png` | Visualization images | PNG |
| `chunks/*.npy` | Raw segmentation masks (if saved) | NumPy arrays |
| `*.zarr` | Zarr store (for distributed processing) | Zarr directory structure |

## Live Segmentation Viewer

SMINT includes a live segmentation viewer for monitoring the segmentation process in real-time:

```python
from smint.visualization.live_scan_viewer import LiveScanViewer
import tkinter as tk

# Initialize the viewer
root = tk.Tk()
viewer = LiveScanViewer(
    master=root,
    full_scan_path="path/to/image.ome.tiff",
    segmentation_history_dir="results/segmentation",
    tile_info_path="results/tile_info.json",
    update_interval_ms=1000  # Update every 1 second
)

# Start the viewer
viewer.pack(fill=tk.BOTH, expand=True)
root.mainloop()
```

## Common Issues and Troubleshooting

### Poor Segmentation Quality

- **Problem**: Cells or nuclei not properly detected
- **Solution**: Adjust diameter, flow_threshold, and cellprob_threshold. Try using larger diameter for bigger cells, lower flow_threshold for weakly stained cells.

### Memory Errors

- **Problem**: "CUDA out of memory" or other memory-related errors
- **Solution**: Reduce chunk_size, increase overlap, or use distributed processing with multiple GPUs.

### Processing Speed

- **Problem**: Segmentation taking too long
- **Solution**: Use multi-GPU processing, reduce visualization, skip nuclei segmentation if not needed.

### Boundary Artifacts

- **Problem**: Cell masks cut off at chunk boundaries
- **Solution**: Increase overlap between chunks or post-process with stitch_masks=True.

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
