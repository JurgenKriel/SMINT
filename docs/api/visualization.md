# Visualization API

The visualization module provides functions for visualizing segmentation results and spatial data.

## Overview

The visualization module includes the following key functionalities:

- **create_rgb_composite**: Create an RGB composite image from multiple channels
- **visualize_segmentation_overlay**: Overlay segmentation results on an image
- **visualize_cell_outlines**: Visualize cell outlines on an image
- **plot_cell_features**: Plot cell features as a heatmap
- **create_segmentation_animation**: Create an animation of segmentation results
- **LiveScanViewer**: Interactive viewer for live segmentation results

## Function Reference

```python
def create_rgb_composite(image_data, channel_indices=(0, 1, 2), normalize=True):
    """
    Create an RGB composite image from multiple channels.
    
    Parameters
    ----------
    image_data : ndarray
        Multi-channel image data with shape (C, H, W)
    channel_indices : tuple, optional
        Indices of channels to use for RGB channels
    normalize : bool, optional
        Whether to normalize each channel
        
    Returns
    -------
    ndarray
        RGB composite image with shape (H, W, 3)
    """
    pass

def visualize_segmentation_overlay(image, segmentation_mask, alpha=0.5, colors=None):
    """
    Overlay segmentation results on an image.
    
    Parameters
    ----------
    image : ndarray
        Background image
    segmentation_mask : ndarray
        Segmentation mask with unique IDs for each object
    alpha : float, optional
        Transparency of the overlay
    colors : ndarray, optional
        Color map for segmentation mask
        
    Returns
    -------
    ndarray
        Image with segmentation overlay
    """
    pass

def visualize_cell_outlines(image, cell_outlines, color=(1, 0, 0), thickness=2):
    """
    Visualize cell outlines on an image.
    
    Parameters
    ----------
    image : ndarray
        Background image
    cell_outlines : list
        List of cell outline coordinates
    color : tuple, optional
        RGB color for outlines
    thickness : int, optional
        Thickness of outline lines
        
    Returns
    -------
    ndarray
        Image with cell outlines
    """
    pass

def plot_cell_features(cell_data, feature_name, image=None, colormap='viridis'):
    """
    Plot cell features as a heatmap.
    
    Parameters
    ----------
    cell_data : DataFrame
        Cell data with coordinates and features
    feature_name : str
        Name of feature to plot
    image : ndarray, optional
        Background image
    colormap : str, optional
        Color map for heatmap
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with cell feature plot
    """
    pass

def create_segmentation_animation(image_sequence, mask_sequence, output_path, fps=5):
    """
    Create an animation of segmentation results.
    
    Parameters
    ----------
    image_sequence : list
        List of background images
    mask_sequence : list
        List of segmentation masks
    output_path : str
        Path to save animation
    fps : int, optional
        Frames per second
        
    Returns
    -------
    str
        Path to saved animation
    """
    pass

class LiveScanViewer:
    """
    Interactive viewer for live segmentation results.
    
    Parameters
    ----------
    master : tkinter.Tk
        Tkinter master window
    full_scan_path : str
        Path to full scan image
    segmentation_history_dir : str
        Directory containing segmentation history
    tile_info_path : str
        Path to tile information file
    update_interval_ms : int, optional
        Update interval in milliseconds
    
    Methods
    -------
    start()
        Start the viewer
    update_views()
        Update all views
    show_next_segmentation()
        Show next segmentation result
    show_previous_segmentation()
        Show previous segmentation result
    """
    pass
```

## Dependency Handling

The visualization module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| matplotlib | All plotting functionality | Stub implementation with helpful error messages |
| OpenCV (cv2) | Image I/O and processing | Limited visualization capabilities |
| tkinter | Live Scan Viewer GUI | Command-line only interface |
| numpy | Data manipulation | Required dependency |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.
