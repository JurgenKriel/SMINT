# Aligning Xenium Spatial Transcriptomics with Spatial Metabolomics

This guide provides a detailed walkthrough for aligning 10X Xenium spatial transcriptomics data with spatial metabolomics data using STalign within the SMINT framework.

## Overview

Integrating spatial transcriptomics data with spatial metabolomics presents unique challenges due to:
- Different resolution and sampling density
- Distinct coordinate systems
- Potential tissue deformation between acquisitions
- Different feature types (genes vs. metabolites)

The SMINT package leverages STalign (from the [JEFworks Lab](https://jef.works/STalign/STalign.html)) to perform robust alignment using diffeomorphic transformations that can account for non-linear deformations between modalities.

## Prerequisites

Before beginning the alignment process, ensure you have:

- 10X Xenium spatial transcriptomics data processed into a coordinate file
- Spatial metabolomics data in matrix format with x,y coordinates
- The STalign package installed (`pip install STalign`)
- Sufficient computational resources (GPU recommended for large datasets)

## Step 1: Data Preparation

### Spatial Metabolomics Data

The spatial metabolomics data should be in a CSV format with:
- Row indices for unique spot/pixel identifiers
- `x` and `y` columns for spatial coordinates
- Additional columns for m/z values representing metabolite intensities

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_sm_matrix(mtx_file, verbose=True):
    """
    Read a Spatial Metabolomics matrix file and extract coordinates and data
    """
    if verbose:
        print("Reading matrix file...")

    try:
        data = pd.read_csv(mtx_file, index_col=0)
        numeric_cols = []
        for col in data.columns:
            try:
                numeric_cols.append(float(col))
            except ValueError:
                continue
    except Exception as e:
        raise Exception(f"Error reading matrix file: {e}")

    coordinates = data[['x', 'y']].values

    if verbose:
        print(f"Found {len(coordinates)} coordinate pairs")
        print(f"Found {len(numeric_cols)} m/z values")

    return coordinates, data, numeric_cols
```

### Xenium Data

Xenium data should be processed to extract cell centroids:

```python
def read_xenium_data(xenium_file, verbose=True):
    """
    Read 10X Xenium data and extract cell coordinates
    """
    if verbose:
        print("Reading Xenium cell coordinates...")
    
    try:
        data = pd.read_csv(xenium_file)
        # Ensure the necessary columns exist
        required_cols = ['x_centroid', 'y_centroid']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in Xenium data")
    except Exception as e:
        raise Exception(f"Error reading Xenium file: {e}")
    
    coordinates = data[['x_centroid', 'y_centroid']].values
    
    if verbose:
        print(f"Found {len(coordinates)} cell coordinates")
    
    return coordinates, data
```

## Step 2: Visualizing Data Before Alignment

Visualizing both datasets before alignment helps assess the initial misalignment and identify potential challenges:

```python
def plot_non_zero_coordinates(coordinates, data, title="Coordinates Plot"):
    """
    Plot the coordinates of all non-zero m/z ratios for spatial metabolomics
    or all cell coordinates for Xenium data
    """
    plt.figure(figsize=(12, 10))
    
    # For spatial metabolomics data
    if 'mz_values' in data.columns:
        mz_values = [float(col) for col in data.columns if col.replace('.', '').isdigit()]
        for mz in mz_values:
            column_name = str(mz)
            intensities = data[column_name].values
            
            # Filter coordinates where intensity is greater than zero
            non_zero_indices = np.where(intensities > 0)[0]
            filtered_coordinates = coordinates[non_zero_indices]
            
            # Plot non-zero coordinates
            plt.scatter(filtered_coordinates[:, 0], filtered_coordinates[:, 1], 
                        s=10, alpha=0.5, color='blue')
    else:
        # For Xenium data, plot all coordinates
        plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                    s=10, alpha=0.5, color='red')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(title)
    plt.grid(True)
    plt.xlim(coordinates[:, 0].min() - 1, coordinates[:, 0].max() + 1)
    plt.ylim(coordinates[:, 1].min() - 1, coordinates[:, 1].max() + 1)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()
```

## Step 3: Rasterization

STalign works with rasterized (gridded) data. The rasterization step transforms point clouds into pixel-based representations:

```python
import STalign

def rasterize_coordinates(x_coords, y_coords, pixel_size, verbose=True):
    """
    Convert point coordinates to a rasterized grid
    
    Parameters:
    -----------
    x_coords : numpy.ndarray
        X coordinates of the points
    y_coords : numpy.ndarray
        Y coordinates of the points
    pixel_size : float
        Size of the pixels in the rasterized grid
    verbose : bool
        Whether to display the rasterization result
        
    Returns:
    --------
    tuple
        (X grid, Y grid, intensity values, figure if verbose=True)
    """
    if verbose:
        print(f"Rasterizing {len(x_coords)} points with pixel size {pixel_size}")
    
    # Perform rasterization
    X_grid, Y_grid, intensity, fig = STalign.rasterize(x_coords, y_coords, dx=pixel_size)
    
    if verbose:
        print(f"Rasterized grid shape: {intensity.shape}")
        plt.savefig(f"rasterized_grid_size_{pixel_size}.png", dpi=300)
        plt.close(fig)
    
    return X_grid, Y_grid, intensity, fig if verbose else None
```

The pixel size parameter (`dx`) is critical:
- Too small: Sparse representation with many empty pixels
- Too large: Loss of spatial resolution and detail
- Optimal: Usually determined empirically (start with 20-50µm for Xenium data)

## Step 4: Running LDDMM Alignment

The Large Deformation Diffeomorphic Metric Mapping (LDDMM) algorithm performs the alignment:

```python
def run_lddmm_alignment(source_grid, target_grid, source_intensity, target_intensity, params=None):
    """
    Perform LDDMM alignment between source and target datasets
    
    Parameters:
    -----------
    source_grid : tuple
        (Y grid, X grid) for source data
    target_grid : tuple
        (Y grid, X grid) for target data
    source_intensity : numpy.ndarray
        Intensity values for source grid
    target_intensity : numpy.ndarray
        Intensity values for target grid
    params : dict, optional
        Parameters for LDDMM algorithm
        
    Returns:
    --------
    dict
        LDDMM output containing transformation parameters
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'niter': 1000,  # Maximum iterations
            'diffeo_start': 50,  # Start diffeomorphic transformation earlier
            'sigmaM': 0.5,  # Kernel width for momentum field
            'sigmaB': 1.2,  # Kernel width for backward map
            'sigmaA': 1.2,  # Kernel width for forward map
            'epV': 500,     # Regularization parameter
        }
    
    # Check for GPU availability and set device
    if torch.cuda.is_available():
        params['device'] = 'cuda:0'
        print("Using GPU for LDDMM calculation")
    else:
        params['device'] = 'cpu'
        print("Using CPU for LDDMM calculation (this may be slow)")
    
    # Run LDDMM
    print("Running LDDMM alignment...")
    output = STalign.LDDMM(
        source_grid,  # [Y_source, X_source]
        source_intensity,
        target_grid,  # [Y_target, X_target]
        target_intensity,
        **params
    )
    
    print("LDDMM alignment complete")
    return output
```

### LDDMM Parameters Explained

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|------------------|
| `niter` | Maximum iterations for optimization | 1000 | 500-2000 |
| `diffeo_start` | When to start diffeomorphic transformation | 50 | 20-100 |
| `sigmaM` | Kernel width for momentum field | 0.5 | 0.1-1.0 |
| `sigmaB` | Kernel width for backward map | 1.2 | 0.5-2.0 |
| `sigmaA` | Kernel width for forward map | 1.2 | 0.5-2.0 |
| `epV` | Regularization parameter | 500 | 100-1000 |
| `device` | Computation device | 'cpu' | 'cpu' or 'cuda:0' |

## Step 5: Transforming Point Coordinates

After obtaining the LDDMM transformation, apply it to transform the original spatial metabolomics coordinates:

```python
def transform_points(lddmm_output, source_coordinates):
    """
    Transform source coordinates using LDDMM output
    
    Parameters:
    -----------
    lddmm_output : dict
        Output from the LDDMM alignment
    source_coordinates : numpy.ndarray
        Original source coordinates to transform (shape: n x 2)
        
    Returns:
    --------
    numpy.ndarray
        Transformed coordinates (shape: n x 2)
    """
    # Extract transformation parameters
    A = lddmm_output['A']
    v = lddmm_output['v']
    xv = lddmm_output['xv']
    
    # Ensure device is set correctly for tensor operations
    if torch.cuda.is_available():
        torch.set_default_device('cuda:0')
    else:
        torch.set_default_device('cpu')
    
    # Stack y and x coordinates (note the order for STalign)
    stacked_coords = np.stack([source_coordinates[:, 1], source_coordinates[:, 0]], axis=1)
    
    # Apply transformation
    transformed_points = STalign.transform_points_source_to_target(xv, v, A, stacked_coords)
    transformed_points = transformed_points.cpu().numpy()
    
    # Return as [x, y] format
    return np.column_stack([transformed_points[:, 1], transformed_points[:, 0]])
```

## Step 6: Visualizing Alignment Results

Visualize the alignment to assess quality:

```python
def visualize_alignment(xenium_coords, transformed_met_coords, output_path=None):
    """
    Visualize the alignment between Xenium and transformed metabolomics data
    
    Parameters:
    -----------
    xenium_coords : numpy.ndarray
        Xenium cell coordinates (shape: n x 2)
    transformed_met_coords : numpy.ndarray
        Transformed metabolomics coordinates (shape: m x 2)
    output_path : str, optional
        Path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot Xenium coordinates
    ax.scatter(
        xenium_coords[:, 0], xenium_coords[:, 1],
        s=0.3, alpha=0.5, label='Xenium', color='blue'
    )
    
    # Plot transformed metabolomics coordinates
    ax.scatter(
        transformed_met_coords[:, 0], transformed_met_coords[:, 1],
        s=0.2, alpha=0.4, label='Transformed Metabolomics', color='orange'
    )
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Alignment of Xenium and Spatial Metabolomics Data')
    ax.legend()
    ax.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
```

## Step 7: Saving Transformed Coordinates

Save the transformed coordinates along with the original data:

```python
def save_transformed_data(original_data, transformed_coords, output_path):
    """
    Save the original data with transformed coordinates
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original data with features
    transformed_coords : numpy.ndarray
        Transformed coordinates (shape: n x 2)
    output_path : str
        Path to save the combined data
    """
    # Ensure the number of rows match
    assert len(original_data) == len(transformed_coords), "Number of coordinates doesn't match data rows"
    
    # Create a copy of the original data
    result_df = original_data.copy()
    
    # Add transformed coordinates
    result_df['x_transformed'] = transformed_coords[:, 0]
    result_df['y_transformed'] = transformed_coords[:, 1]
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Saved transformed data to {output_path}")
```

## Complete Workflow Example

Here's a complete example that performs all the steps:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import STalign
import os

# Step 1: Read data
met_file = "path/to/metabolomics_data.csv"
xenium_file = "path/to/xenium_data.csv"

# Create output directory
output_dir = "alignment_results"
os.makedirs(output_dir, exist_ok=True)

# Read metabolomics data
met_coords, met_data, mz_values = read_sm_matrix(met_file)

# Read Xenium data
xenium_coords, xenium_data = read_xenium_data(xenium_file)

# Step 2: Visualize before alignment
plot_non_zero_coordinates(met_coords, met_data, "Spatial Metabolomics Coordinates")
plot_non_zero_coordinates(xenium_coords, xenium_data, "Xenium Cell Coordinates")

# Step 3: Rasterize both datasets
# Experiment with different pixel sizes
pixel_size = 30  # in the same units as your coordinates (usually µm)
X_met, Y_met, intensity_met, _ = rasterize_coordinates(
    met_coords[:, 0], met_coords[:, 1], pixel_size
)
X_xenium, Y_xenium, intensity_xenium, _ = rasterize_coordinates(
    xenium_coords[:, 0], xenium_coords[:, 1], pixel_size
)

# Step 4: Run LDDMM alignment
# Customize parameters based on your data
lddmm_params = {
    'niter': 1000,
    'diffeo_start': 50,
    'sigmaM': 0.5,
    'sigmaB': 1.2,
    'sigmaA': 1.2,
    'epV': 500,
}

lddmm_output = run_lddmm_alignment(
    [Y_met, X_met], [Y_xenium, X_xenium],
    intensity_met, intensity_xenium,
    params=lddmm_params
)

# Step 5: Transform metabolomics coordinates
transformed_met_coords = transform_points(lddmm_output, met_coords)

# Step 6: Visualize alignment results
output_plot_path = os.path.join(output_dir, "xenium_metabolomics_alignment.png")
visualize_alignment(xenium_coords, transformed_met_coords, output_plot_path)

# Step 7: Save transformed data
output_data_path = os.path.join(output_dir, "metabolomics_transformed.csv")
save_transformed_data(met_data, transformed_met_coords, output_data_path)
```

## Optimization Tips

### Tuning the LDDMM Parameters

1. **Pixel Size (dx)**: 
   - Start with a moderate value (20-50µm) 
   - Too small: Sparse representation
   - Too large: Loss of detail
   - Optimal: Balances detail and computational efficiency

2. **sigmaM, sigmaB, sigmaA (Kernel Width Parameters)**:
   - Control the smoothness of the transformation
   - Smaller values: Allow more local deformations
   - Larger values: More global transformation, less flexibility
   - Recommended approach: Start with defaults, then adjust if alignment quality is poor

3. **epV (Regularization Parameter)**:
   - Controls the trade-off between data fitting and transformation smoothness
   - Higher values: Smoother, more conservative transformations
   - Lower values: More aggressive fitting, potentially more distortion
   - Start with 500, increase if the transformation appears too distorted

### GPU vs. CPU Performance

- For large datasets (>10,000 points), GPU acceleration is strongly recommended
- With GPU: Minutes to tens of minutes
- Without GPU: Hours to days for large datasets

## Troubleshooting

### Common Issues

1. **Poor Alignment Quality**
   - Check initial data quality and preprocessing
   - Try different pixel sizes for rasterization
   - Adjust LDDMM parameters, particularly sigmaM and epV
   - Consider a rough manual pre-alignment if datasets are severely misaligned

2. **Memory Errors**
   - Reduce rasterization resolution (increase pixel size)
   - Process subsets of the data
   - Use a machine with more GPU memory

3. **Inverted or Flipped Alignment**
   - Pre-process your data to ensure consistent orientation
   - Try flipping one dataset before alignment

4. **Alignment Fails to Converge**
   - Increase the number of iterations (niter)
   - Try a simpler transformation first, then use that as initialization
   - Check for outliers in your data

## Integration with SMINT

The SMINT package simplifies this entire process with a single function:

```python
from smint.alignment import align_xenium_to_metabolomics

# Simple usage
aligned_data = align_xenium_to_metabolomics(
    xenium_file="path/to/xenium_data.csv",
    metabolomics_file="path/to/metabolomics_data.csv",
    output_dir="alignment_results",
    pixel_size=30,
    visualize=True
)

# Advanced usage with custom parameters
aligned_data = align_xenium_to_metabolomics(
    xenium_file="path/to/xenium_data.csv",
    metabolomics_file="path/to/metabolomics_data.csv",
    output_dir="alignment_results",
    pixel_size=30,
    xenium_x_col="x_centroid",
    xenium_y_col="y_centroid",
    met_x_col="x",
    met_y_col="y",
    lddmm_params={
        'niter': 1500,
        'sigmaM': 0.3,
        'epV': 600
    },
    visualize=True,
    save_intermediate=True
)
```

## References

1. Levy-Jurgenson, A., Fei, D., Xia, S.Y. et al. STORM: Super Tissue Object Reconstruction in Medicine—a toolkit for multiplex fluorescence spatial proteomics. Cell 186, 2547–2566.e17 (2023). [DOI: 10.1016/j.cell.2023.04.015](https://doi.org/10.1016/j.cell.2023.04.015)

2. Fu, Y., Fan, J. STalign: Align spatial transcriptomics data across multiple platforms. bioRxiv. [DOI: 10.1101/2023.09.11.557230](https://doi.org/10.1101/2023.09.11.557230)

3. 10x Genomics Xenium documentation: [https://www.10xgenomics.com/products/xenium-in-situ](https://www.10xgenomics.com/products/xenium-in-situ)
