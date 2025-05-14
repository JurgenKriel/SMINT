# Examples

This page provides complete examples of SMINT workflows, demonstrating how to use the package for different spatial omics applications.

## Basic Workflow

The following example demonstrates a complete workflow from preprocessing through segmentation to visualization:

```python
import os
from smint.preprocessing import preprocess_ome_tiff
from smint.segmentation import process_large_image
from smint.visualization import visualize_cell_outlines
import pandas as pd
import matplotlib.pyplot as plt

# Set up directories
image_path = "path/to/image.ome.tif"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Preprocess the image
preprocessed = preprocess_ome_tiff(
    image_path=image_path,
    sigma=1.5,
    normalize=True
)

# Combine channels if needed
if len(preprocessed) >= 2:
    from smint.preprocessing import combine_channels
    
    channel_names = list(preprocessed.keys())
    combined = combine_channels(
        preprocessed[channel_names[0]],
        preprocessed[channel_names[1]]
    )
    
    # Save combined image
    import tifffile
    combined_path = os.path.join(output_dir, "combined.tif")
    tifffile.imwrite(combined_path, combined)
    print(f"Saved combined image to {combined_path}")
    
    # Use combined image for segmentation
    segmentation_input = combined_path
else:
    # Use first channel for segmentation
    import tifffile
    channel_name = list(preprocessed.keys())[0]
    segmentation_input = os.path.join(output_dir, f"{channel_name}.tif")
    tifffile.imwrite(segmentation_input, preprocessed[channel_name])

# Step 2: Segment cells and nuclei
results = process_large_image(
    image_path=segmentation_input,
    csv_base_path=os.path.join(output_dir, "cell_outlines"),
    chunk_size=(2048, 2048),
    # Cell Model parameters
    cell_model_path="cyto",
    cells_diameter=120.0,
    cells_flow_threshold=0.4,
    cells_cellprob_threshold=-1.5,
    cells_channels=[0, 0],
    # Nuclei Model parameters
    nuclei_model_path="nuclei",
    nuclei_diameter=40.0,
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=-1.2,
    nuclei_channels=[0, 0],
    # Visualization parameters
    visualize=True,
    visualize_output_dir=os.path.join(output_dir, "visualizations")
)

print(f"Found {results['total_cells']} cells and {results['total_nuclei']} nuclei")

# Step 3: Load and visualize the results
if results.get('cells_csv_path'):
    cells_df = pd.read_csv(results['cells_csv_path'])
    print(f"Loaded {cells_df['global_cell_id'].nunique()} cells")
    
    # Create visualization
    fig = visualize_cell_outlines(cells_df, color_by='global_cell_id')
    plt.savefig(os.path.join(output_dir, "cell_outlines.png"), dpi=300)
    plt.close(fig)
