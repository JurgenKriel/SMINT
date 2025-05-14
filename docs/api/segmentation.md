# Segmentation API

The segmentation module provides functions for segmenting cells and nuclei in whole-slide images.

## Overview

The segmentation module includes the following key functionalities:

- **process_large_image**: Process a large whole-slide image for cell segmentation
- **run_distributed_segmentation**: Run segmentation using distributed computing
- **get_cell_outlines**: Extract cell outlines from segmentation masks
- **segment_chunk**: Segment a single chunk of a whole-slide image
- **extract_contours**: Extract contours from segmentation masks
- **save_masks**: Save segmentation masks to disk

## Function Reference

```python
def process_large_image(image_path, csv_base_path, **kwargs):
    """
    Process a large whole-slide image for cell segmentation.
    
    Parameters
    ----------
    image_path : str
        Path to the input image file
    csv_base_path : str
        Base path for output CSV files
    **kwargs : dict
        Additional parameters for segmentation
        
    Returns
    -------
    dict
        Information about the segmentation results
    """
    pass

def run_distributed_segmentation(image_path, output_path, **kwargs):
    """
    Run segmentation using distributed computing.
    
    Parameters
    ----------
    image_path : str
        Path to the input image file
    output_path : str
        Path for output files
    **kwargs : dict
        Additional parameters for distributed segmentation
        
    Returns
    -------
    dict
        Information about the segmentation results
    """
    pass

def get_cell_outlines(masks):
    """
    Extract cell outlines from segmentation masks.
    
    Parameters
    ----------
    masks : ndarray
        Segmentation masks
        
    Returns
    -------
    list
        List of cell outline coordinates
    """
    pass

def segment_chunk(chunk_data, model_instance, **kwargs):
    """
    Segment a single chunk of a whole-slide image.
    
    Parameters
    ----------
    chunk_data : ndarray
        Data for the current chunk
    model_instance : object
        Cellpose model instance
    **kwargs : dict
        Additional parameters for chunk segmentation
        
    Returns
    -------
    dict
        Segmentation results for the chunk
    """
    pass

def extract_contours(mask):
    """
    Extract contours from segmentation masks.
    
    Parameters
    ----------
    mask : ndarray
        Segmentation mask
        
    Returns
    -------
    list
        List of contours
    """
    pass

def save_masks(masks, output_path):
    """
    Save segmentation masks to disk.
    
    Parameters
    ----------
    masks : ndarray
        Segmentation masks
    output_path : str
        Path to save masks
        
    Returns
    -------
    bool
        Success status
    """
    pass
```

## Optional Dependencies

The segmentation module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| Cellpose | Cell segmentation models | Stub implementation with helpful error messages |
| OpenCV (cv2) | Image I/O and contour extraction | Limited visualization, no mask saving/loading |
| Dask | Distributed processing | Single-process implementation only |
| Distributed | Multi-node computation | Single-node implementation only |
| CUDA | GPU acceleration | CPU-only implementation |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.