# Alignment API

The alignment module provides functions for aligning spatial omics data using ST Align.

## Overview

The alignment module includes the following key functionalities:

- **run_alignment**: Run the ST Align algorithm to align spatial omics data
- **transform_coordinates**: Apply transformation to coordinates based on alignment results
- **create_config**: Create a configuration file for the ST Align algorithm
- **validate_alignment**: Validate alignment results using quality metrics
- **align_xenium_to_metabolomics**: Align 10X Xenium data to spatial metabolomics data using LDDMM

## Function Reference

```python
def run_alignment(source_data, target_data, config=None, method="similarity"):
    """
    Run ST Align to align source data to target data.
    
    Parameters
    ----------
    source_data : str or DataFrame
        Path to source data CSV or DataFrame
    target_data : str or DataFrame
        Path to target data CSV or DataFrame
    config : dict, optional
        Configuration parameters for ST Align
    method : str, optional
        Transformation method, one of "rigid", "similarity", "affine", "projective"
        
    Returns
    -------
    dict
        Dictionary containing transformation matrix and alignment metrics
    """
    pass

def transform_coordinates(coordinates, transformation_matrix):
    """
    Transform coordinates using a transformation matrix.
    
    Parameters
    ----------
    coordinates : DataFrame or array-like
        Coordinates to transform
    transformation_matrix : array-like
        Transformation matrix from alignment
        
    Returns
    -------
    DataFrame or array-like
        Transformed coordinates
    """
    pass

def create_config(parameters=None):
    """
    Create a configuration dictionary for ST Align.
    
    Parameters
    ----------
    parameters : dict, optional
        Custom parameters to override defaults
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    pass

def validate_alignment(source_transformed, target, metrics=None):
    """
    Validate alignment results using quality metrics.
    
    Parameters
    ----------
    source_transformed : DataFrame
        Transformed source coordinates
    target : DataFrame
        Target coordinates
    metrics : list, optional
        Metrics to calculate
        
    Returns
    -------
    dict
        Dictionary of quality metrics
    """
    pass

# Xenium-Metabolomics Alignment

def align_xenium_to_metabolomics(xenium_file, metabolomics_file, output_dir="alignment_results", 
                                pixel_size=30, xenium_x_col="x_centroid", xenium_y_col="y_centroid", 
                                met_x_col="x", met_y_col="y", lddmm_params=None, 
                                visualize=True, save_intermediate=False):
    """
    Align Xenium spatial transcriptomics data to spatial metabolomics data using LDDMM.
    
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
    pass

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
    pass

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
    pass

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
    pass
```

## Dependency Handling

The alignment module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | R bridge functionality | Stub implementation with helpful error messages |
| pandas | Data manipulation | Required dependency |
| numpy | Matrix operations | Required dependency |
| matplotlib | Visualization | Limited visualization capabilities |
| torch | GPU acceleration for LDDMM | CPU-only operations (slower) |
| STalign | Xenium-Metabolomics alignment | Functionality unavailable with informative error messages |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality. In the case of the Xenium-Metabolomics alignment, the `torch` and `STalign` packages are required, but their absence is gracefully handled so that other parts of the SMINT package can still function.
