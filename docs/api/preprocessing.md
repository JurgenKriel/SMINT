# Preprocessing API

The preprocessing module provides functions for preparing and transforming spatial omics data for analysis.

## Overview

The preprocessing module includes the following key functionalities:

- **split_large_ometiff**: Split a large OME-TIFF file into smaller chunks
- **extract_channels**: Extract specific channels from an OME-TIFF file
- **normalize_expression**: Normalize expression data for analysis
- **scale_data**: Scale data to a specific range

## Function Reference

```python
def split_large_ometiff(file_path, output_dir, chunk_size=(2048, 2048)):
    """
    Split a large OME-TIFF file into smaller chunks.
    
    Parameters
    ----------
    file_path : str
        Path to the input OME-TIFF file
    output_dir : str
        Directory to save the output chunks
    chunk_size : tuple, optional
        Size of chunks (height, width)
        
    Returns
    -------
    dict
        Information about the chunking process
    """
    pass

def extract_channels(file_path, output_path, channels=None):
    """
    Extract specific channels from an OME-TIFF file.
    
    Parameters
    ----------
    file_path : str
        Path to the input OME-TIFF file
    output_path : str
        Path to save the extracted channels
    channels : list, optional
        List of channel indices to extract
        
    Returns
    -------
    dict
        Information about the extraction process
    """
    pass

def normalize_expression(data, method="log1p", scale_factor=10000):
    """
    Normalize expression data for analysis.
    
    Parameters
    ----------
    data : DataFrame or array-like
        Expression data to normalize
    method : str, optional
        Normalization method
    scale_factor : float, optional
        Scale factor for normalization
        
    Returns
    -------
    DataFrame or array-like
        Normalized data
    """
    pass

def scale_data(data, feature_range=(0, 1), axis=0):
    """
    Scale data to a specific range.
    
    Parameters
    ----------
    data : DataFrame or array-like
        Data to scale
    feature_range : tuple, optional
        Range to scale to
    axis : int, optional
        Axis along which to scale
        
    Returns
    -------
    DataFrame or array-like
        Scaled data
    """
    pass
```

## Dependency Handling

The preprocessing module relies on several libraries for optimal performance, but can operate with limited functionality when some dependencies are missing:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| tifffile | OME-TIFF reading/writing | Stub implementation with helpful error messages |
| scikit-image | Image transformations | Limited image processing capabilities |
| OpenCV (cv2) | Fast image I/O | Slower fallback implementation |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.
