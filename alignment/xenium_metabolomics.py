"""
Alignment module for Xenium spatial transcriptomics and spatial metabolomics data.

This module provides functions to align 10X Xenium spatial transcriptomics data 
with spatial metabolomics data using the STalign package.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Optional imports - STalign and torch
try:
    import torch
    import STalign
    STALIGN_AVAILABLE = True
except ImportError:
    logger.warning("STalign or torch not available. Xenium-Metabolomics alignment functionality will be limited.")
    STALIGN_AVAILABLE = False


def read_sm_matrix(mtx_file, x_col='x', y_col='y', verbose=True):
    """
    Read a Spatial Metabolomics matrix file and extract coordinates and data.
    
    Parameters
    ----------
    mtx_file : str
        Path to the metabolomics matrix file
    x_col : str, optional
        Column name for x coordinates
    y_col : str, optional
        Column name for y coordinates
    verbose : bool, optional
        Whether to print progress information
        
    Returns
    -------
    tuple
        (coordinates, data, numeric_columns)
        coordinates: numpy.ndarray with shape (n, 2)
        data: pandas.DataFrame with all data
        numeric_columns: list of numeric column names (m/z values)
    """
    if not os.path.exists(mtx_file):
        raise FileNotFoundError(f"Metabolomics file not found: {mtx_file}")
        
    if verbose:
        logger.info("Reading metabolomics matrix file...")

    try:
        data = pd.read_csv(mtx_file, index_col=0)
        
        # Check if coordinate columns exist
        if x_col not in data.columns or y_col not in data.columns:
            raise ValueError(f"Required coordinate columns '{x_col}' and '{y_col}' not found in data")
            
        numeric_cols = []
        for col in data.columns:
            if col in [x_col, y_col]:
                continue
                
            try:
                numeric_cols.append(float(col))
            except ValueError:
                continue
    except Exception as e:
        raise Exception(f"Error reading matrix file: {e}")

    coordinates = data[[x_col, y_col]].values

    if verbose:
        logger.info(f"Found {len(coordinates)} coordinate pairs")
        logger.info(f"Found {len(numeric_cols)} m/z values")

    return coordinates, data, numeric_cols


def read_xenium_data(xenium_file, x_col='x_centroid', y_col='y_centroid', verbose=True):
    """
    Read 10X Xenium data and extract cell coordinates.
    
    Parameters
    ----------
    xenium_file : str
        Path to the Xenium coordinate file
    x_col : str, optional
        Column name for x coordinates
    y_col : str, optional
        Column name for y coordinates
    verbose : bool, optional
        Whether to print progress information
        
    Returns
    -------
    tuple
        (coordinates, data)
        coordinates: numpy.ndarray with shape (n, 2)
        data: pandas.DataFrame with all data
    """
    if not os.path.exists(xenium_file):
        raise FileNotFoundError(f"Xenium file not found: {xenium_file}")
        
    if verbose:
        logger.info("Reading Xenium cell coordinates...")
    
    try:
        data = pd.read_csv(xenium_file)
        
        # Ensure the necessary columns exist
        required_cols = [x_col, y_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in Xenium data")
    except Exception as e:
        raise Exception(f"Error reading Xenium file: {e}")
    
    coordinates = data[[x_col, y_col]].values
    
    if verbose:
        logger.info(f"Found {len(coordinates)} cell coordinates")
    
    return coordinates, data


def plot_coordinates(coordinates, title="Coordinates Plot", color='blue', 
                     alpha=0.5, size=10, output_path=None):
    """
    Plot coordinates for visualization.
    
    Parameters
    ----------
    coordinates : numpy.ndarray
        Coordinates to plot, shape (n, 2)
    title : str, optional
        Plot title
    color : str, optional
        Point color
    alpha : float, optional
        Point transparency
    size : float, optional
        Point size
    output_path : str, optional
        Path to save the plot image
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.scatter(coordinates[:, 0], coordinates[:, 1], 
              s=size, alpha=alpha, color=color)

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title)
    ax.grid(True)
    
    # Add some padding to the limits
    ax.set_xlim(coordinates[:, 0].min() - 1, coordinates[:, 0].max() + 1)
    ax.set_ylim(coordinates[:, 1].min() - 1, coordinates[:, 1].max() + 1)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    return fig


def rasterize_coordinates(x_coords, y_coords, pixel_size, verbose=True):
    """
    Convert point coordinates to a rasterized grid.
    
    Parameters
    ----------
    x_coords : numpy.ndarray
        X coordinates of the points
    y_coords : numpy.ndarray
        Y coordinates of the points
    pixel_size : float
        Size of the pixels in the rasterized grid
    verbose : bool, optional
        Whether to display information about the rasterization
        
    Returns
    -------
    tuple
        (X grid, Y grid, intensity values, figure if verbose=True)
    """
    if not STALIGN_AVAILABLE:
        raise ImportError("STalign package is required for rasterization")
        
    if verbose:
        logger.info(f"Rasterizing {len(x_coords)} points with pixel size {pixel_size}")
    
    # Perform rasterization
    X_grid, Y_grid, intensity, fig = STalign.rasterize(x_coords, y_coords, dx=pixel_size)
    
    if verbose:
        logger.info(f"Rasterized grid shape: {intensity.shape}")
    
    return X_grid, Y_grid, intensity, fig if verbose else None


def run_lddmm_alignment(source_grid, target_grid, source_intensity, target_intensity, params=None):
    """
    Perform LDDMM alignment between source and target datasets.
    
    Parameters
    ----------
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
        
    Returns
    -------
    dict
        LDDMM output containing transformation parameters
    """
    if not STALIGN_AVAILABLE:
        raise ImportError("STalign package is required for LDDMM alignment")
        
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
        logger.info("Using GPU for LDDMM calculation")
    else:
        params['device'] = 'cpu'
        logger.info("Using CPU for LDDMM calculation (this may be slow)")
    
    # Run LDDMM
    logger.info("Running LDDMM alignment...")
    output = STalign.LDDMM(
        source_grid,  # [Y_source, X_source]
        source_intensity,
        target_grid,  # [Y_target, X_target]
        target_intensity,
        **params
    )
    
    logger.info("LDDMM alignment complete")
    return output


def transform_points(lddmm_output, source_coordinates):
    """
    Transform source coordinates using LDDMM output.
    
    Parameters
    ----------
    lddmm_output : dict
        Output from the LDDMM alignment
    source_coordinates : numpy.ndarray
        Original source coordinates to transform (shape: n x 2)
        
    Returns
    -------
    numpy.ndarray
        Transformed coordinates (shape: n x 2)
    """
    if not STALIGN_AVAILABLE:
        raise ImportError("STalign package is required for coordinate transformation")
        
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


def visualize_alignment(xenium_coords, transformed_met_coords, 
                         xenium_label="Xenium", met_label="Metabolomics",
                         output_path=None):
    """
    Visualize the alignment between Xenium and transformed metabolomics data.
    
    Parameters
    ----------
    xenium_coords : numpy.ndarray
        Xenium cell coordinates (shape: n x 2)
    transformed_met_coords : numpy.ndarray
        Transformed metabolomics coordinates (shape: m x 2)
    xenium_label : str, optional
        Label for Xenium data in legend
    met_label : str, optional
        Label for metabolomics data in legend
    output_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot Xenium coordinates
    ax.scatter(
        xenium_coords[:, 0], xenium_coords[:, 1],
        s=0.3, alpha=0.5, label=xenium_label, color='blue'
    )
    
    # Plot transformed metabolomics coordinates
    ax.scatter(
        transformed_met_coords[:, 0], transformed_met_coords[:, 1],
        s=0.2, alpha=0.4, label=met_label, color='orange'
    )
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Alignment of Xenium and Spatial Metabolomics Data')
    ax.legend()
    ax.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_transformed_data(original_data, transformed_coords, output_path,
                         x_col='x_transformed', y_col='y_transformed'):
    """
    Save the original data with transformed coordinates.
    
    Parameters
    ----------
    original_data : pandas.DataFrame
        Original data with features
    transformed_coords : numpy.ndarray
        Transformed coordinates (shape: n x 2)
    output_path : str
        Path to save the combined data
    x_col : str, optional
        Column name for transformed x coordinates
    y_col : str, optional
        Column name for transformed y coordinates
        
    Returns
    -------
    pandas.DataFrame
        Data with transformed coordinates added
    """
    # Ensure the number of rows match
    assert len(original_data) == len(transformed_coords), "Number of coordinates doesn't match data rows"
    
    # Create a copy of the original data
    result_df = original_data.copy()
    
    # Add transformed coordinates
    result_df[x_col] = transformed_coords[:, 0]
    result_df[y_col] = transformed_coords[:, 1]
    
    # Save to CSV
    if output_path:
        result_df.to_csv(output_path, index=False)
        logger.info(f"Saved transformed data to {output_path}")
    
    return result_df


def align_xenium_to_metabolomics(
    xenium_file,
    metabolomics_file,
    output_dir="alignment_results",
    pixel_size=30,
    xenium_x_col="x_centroid",
    xenium_y_col="y_centroid",
    met_x_col="x",
    met_y_col="y",
    lddmm_params=None,
    visualize=True,
    save_intermediate=False
):
    """
    Align Xenium spatial transcriptomics data to spatial metabolomics data.
    
    Parameters
    ----------
    xenium_file : str
        Path to the Xenium coordinate file
    metabolomics_file : str
        Path to the metabolomics matrix file
    output_dir : str, optional
        Directory to save alignment results
    pixel_size : float, optional
        Size of pixels for rasterization
    xenium_x_col : str, optional
        Column name for Xenium x coordinates
    xenium_y_col : str, optional
        Column name for Xenium y coordinates
    met_x_col : str, optional
        Column name for metabolomics x coordinates
    met_y_col : str, optional
        Column name for metabolomics y coordinates
    lddmm_params : dict, optional
        Parameters for LDDMM algorithm
    visualize : bool, optional
        Whether to generate visualization plots
    save_intermediate : bool, optional
        Whether to save intermediate results
        
    Returns
    -------
    pandas.DataFrame
        Metabolomics data with transformed coordinates added
    """
    if not STALIGN_AVAILABLE:
        raise ImportError("STalign package is required for alignment")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Read data
    logger.info("Reading input data...")
    met_coords, met_data, _ = read_sm_matrix(
        metabolomics_file, x_col=met_x_col, y_col=met_y_col
    )
    
    xenium_coords, xenium_data = read_xenium_data(
        xenium_file, x_col=xenium_x_col, y_col=xenium_y_col
    )
    
    # Step 2: Visualize before alignment (if requested)
    if visualize:
        logger.info("Generating pre-alignment visualizations...")
        met_plot = plot_coordinates(
            met_coords, 
            title="Spatial Metabolomics - Before Alignment",
            color='orange',
            output_path=os.path.join(output_dir, "metabolomics_before_alignment.png") if save_intermediate else None
        )
        
        xenium_plot = plot_coordinates(
            xenium_coords, 
            title="Xenium Spatial Transcriptomics",
            color='blue',
            output_path=os.path.join(output_dir, "xenium_coordinates.png") if save_intermediate else None
        )
        
        if not save_intermediate:
            plt.close(met_plot)
            plt.close(xenium_plot)
    
    # Step 3: Rasterize both datasets
    logger.info(f"Rasterizing with pixel size {pixel_size}...")
    X_met, Y_met, intensity_met, _ = rasterize_coordinates(
        met_coords[:, 0], met_coords[:, 1], pixel_size
    )
    X_xenium, Y_xenium, intensity_xenium, _ = rasterize_coordinates(
        xenium_coords[:, 0], xenium_coords[:, 1], pixel_size
    )
    
    # Step 4: Run LDDMM alignment
    lddmm_output = run_lddmm_alignment(
        [Y_met, X_met], [Y_xenium, X_xenium],
        intensity_met, intensity_xenium,
        params=lddmm_params
    )
    
    # Step 5: Transform metabolomics coordinates
    logger.info("Transforming coordinates...")
    transformed_met_coords = transform_points(lddmm_output, met_coords)
    
    # Step 6: Visualize alignment results
    if visualize:
        logger.info("Generating post-alignment visualization...")
        alignment_plot = visualize_alignment(
            xenium_coords, 
            transformed_met_coords,
            xenium_label="Xenium", 
            met_label="Transformed Metabolomics",
            output_path=os.path.join(output_dir, "xenium_metabolomics_alignment.png")
        )
        plt.close(alignment_plot)
    
    # Step 7: Save transformed data
    logger.info("Saving transformed data...")
    output_data_path = os.path.join(output_dir, "metabolomics_transformed.csv")
    result_df = save_transformed_data(
        met_data,
        transformed_met_coords,
        output_path=output_data_path
    )
    
    logger.info("Alignment complete!")
    return result_df
