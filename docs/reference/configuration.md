# Configuration

SMINT uses YAML configuration files to control various aspects of the processing pipeline. This page explains how to configure SMINT for your specific needs.

## Configuration File Structure

SMINT configuration files are written in YAML format and have the following structure:

```yaml
# Base configuration
output_dir: "results/segmentation"
tile_info_path: "results/segmentation/current_tile_info.txt"
live_update_image_path: "results/segmentation/live_view.png"

# Distributed processing
use_gpu: true
n_workers: null  # null = use all available workers
chunk_size: [2048, 2048]
memory_limit: "16GB"

# Model paths
model_paths:
  - "cyto"  # Built-in Cellpose model for cytoplasm
  - "nuclei"  # Built-in Cellpose model for nuclei
  # - "/path/to/custom/model"  # Path to a custom model

# Cell segmentation parameters
cell_model_params:
  diameter: 120.0
  flow_threshold: 0.4
  cellprob_threshold: -1.5
  channels: [1, 2]  # [cytoplasm, nucleus]

# Nuclear segmentation parameters
nuclei_model_params:
  diameter: 40.0
  flow_threshold: 0.4
  cellprob_threshold: -1.2
  channels: [2, 0]  # [nucleus, no second channel]
  
# Adaptive nuclear segmentation
adaptive_nuclei:
  enable: false
  cellprob_lower_limit: -6.0
  step_decrement: 0.2
  max_attempts: 3
  trigger_ratio: 0.05

# Preprocessing options
preprocessing:
  sigma: 1.0  # Gaussian blur sigma
  normalize: true
  channel_names:
    - "DAPI"
    - "AF568"
    - "AF647"

# Visualization options
visualization:
  enable: true
  output_dir: "results/segmentation/visualizations"
  num_chunks_to_visualize: 5
  roi_size: [2024, 2024]
  background_channel_indices: [0, 1]  # Channels to use for visualization background
