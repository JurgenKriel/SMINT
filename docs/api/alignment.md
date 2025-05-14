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
```

## Dependency Handling

The alignment module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | R bridge functionality | Stub implementation with helpful error messages |
| pandas | Data manipulation | Required dependency |
| numpy | Matrix operations | Required dependency |
| matplotlib | Visualization | Limited visualization capabilities |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.
