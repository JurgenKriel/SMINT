"""
Visualization utilities for segmentation and alignment results.

This module provides functions for visualizing segmentation results,
creating overlays, and generating RGB composites.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from skimage.exposure import rescale_intensity
import cv2
import logging
from pathlib import Path
import os

def create_rgb_composite(channels, red_channel=None, green_channel=None, blue_channel=None, 
                        percentile_min=1, percentile_max=99):
    """
    Create an RGB composite image from multiple channels.
    
    Args:
        channels (dict): Dictionary mapping channel names to image arrays
        red_channel (str, optional): Channel name to use for red
        green_channel (str, optional): Channel name to use for green
        blue_channel (str, optional): Channel name to use for blue
        percentile_min (int): Minimum percentile for normalization
        percentile_max (int): Maximum percentile for normalization
    
    Returns:
        numpy.ndarray: RGB composite image
    """
    logger = logging.getLogger(__name__)
    
    if not channels:
        logger.error("No channels provided")
        return None
    
    # Get shape from first channel
    first_channel = next(iter(channels.values()))
    height, width = first_channel.shape
    
    # Initialize empty RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Function to normalize a channel
    def normalize_channel(channel):
        if channel is None or channel.size == 0:
            return np.zeros((height, width), dtype=np.float32)
        
        # Convert to float for processing
        channel_float = channel.astype(np.float32)
        
        # Calculate percentiles
        p_min, p_max = np.percentile(channel_float, (percentile_min, percentile_max))
        
        # Handle edge case
        if p_min == p_max:
            return np.zeros_like(channel_float)
        
        # Normalize to 0-1 range
        return rescale_intensity(channel_float, in_range=(p_min, p_max), out_range=(0, 1))
    
    # Add channels to RGB image
    channel_indices = {'red': 0, 'green': 1, 'blue': 2}
    
    for channel_name, color_idx in [
        (red_channel, channel_indices['red']), 
        (green_channel, channel_indices['green']), 
        (blue_channel, channel_indices['blue'])
    ]:
        if channel_name and channel_name in channels:
            rgb_image[..., color_idx] = normalize_channel(channels[channel_name])
            logger.info(f"Added {channel_name} to {list(channel_indices.keys())[color_idx]} channel")
    
    # Clip to ensure valid RGB values
    rgb_image = np.clip(rgb_image, 0, 1)
    
    return rgb_image

def visualize_segmentation_overlay(image, masks, outlines=None, figsize=(10, 10), 
                                  alpha=0.5, outline_color='red', cmap='viridis'):
    """
    Visualize segmentation results overlaid on an image.
    
    Args:
        image (numpy.ndarray): Background image 
        masks (numpy.ndarray): Segmentation masks
        outlines (pandas.DataFrame, optional): Dataframe with contour points
        figsize (tuple): Figure size
        alpha (float): Transparency of the mask overlay
        outline_color (str): Color for outlines
        cmap (str): Colormap for masks
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display background image
    if image.ndim == 3 and image.shape[2] == 3:  # RGB image
        ax.imshow(image)
    else:  # Grayscale image
        ax.imshow(image, cmap='gray')
    
    # Add mask overlay with transparency
    if masks is not None and masks.size > 0:
        masked = np.ma.masked_where(masks == 0, masks)
        ax.imshow(masked, cmap=cmap, alpha=alpha)
    
    # Add outlines
    if outlines is not None and not outlines.empty:
        for cell_id, group in outlines.groupby('global_cell_id' if 'global_cell_id' in outlines.columns else 'cell_id'):
            ax.plot(group['x'], group['y'], color=outline_color, linewidth=1.0, alpha=0.8)
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def visualize_cell_outlines(outlines_df, image=None, figsize=(12, 12), 
                           cell_id_col='global_cell_id', color_by=None, 
                           colormap='tab20', alpha=0.8, linewidth=1.0):
    """
    Visualize cell outlines from a dataframe.
    
    Args:
        outlines_df (pandas.DataFrame): DataFrame with cell outline coordinates
        image (numpy.ndarray, optional): Background image
        figsize (tuple): Figure size
        cell_id_col (str): Column name for cell IDs
        color_by (str, optional): Column to use for coloring outlines
        colormap (str): Matplotlib colormap
        alpha (float): Transparency of outlines
        linewidth (float): Width of outline lines
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    if outlines_df.empty:
        return None
    
    # Check required columns
    required_cols = [cell_id_col, 'x', 'y']
    if not all(col in outlines_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display background image if provided
    if image is not None:
        if image.ndim == 3 and image.shape[2] == 3:  # RGB image
            ax.imshow(image)
        else:  # Grayscale image
            ax.imshow(image, cmap='gray')
    
    # Determine coloring scheme
    if color_by is not None and color_by in outlines_df.columns:
        # Get unique values for coloring
        unique_values = outlines_df[color_by].unique()
        cmap = plt.get_cmap(colormap, len(unique_values))
        color_map = {val: cmap(i) for i, val in enumerate(unique_values)}
        
        # Plot each group with its color
        for val, group in outlines_df.groupby(color_by):
            for cell_id, cell_group in group.groupby(cell_id_col):
                ax.plot(cell_group['x'], cell_group['y'], 
                        color=color_map[val], 
                        linewidth=linewidth, 
                        alpha=alpha)
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[val], edgecolor='k',
                                label=str(val)) for val in unique_values]
        ax.legend(handles=legend_elements, title=color_by, 
                 loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        # Plot each cell with a unique color
        unique_cells = outlines_df[cell_id_col].unique()
        cmap = plt.get_cmap(colormap, len(unique_cells))
        
        for i, cell_id in enumerate(unique_cells):
            cell_data = outlines_df[outlines_df[cell_id_col] == cell_id]
            ax.plot(cell_data['x'], cell_data['y'], 
                    color=cmap(i), 
                    linewidth=linewidth, 
                    alpha=alpha)
    
    ax.set_title(f"Cell Outlines (n={outlines_df[cell_id_col].nunique()})")
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def create_segmentation_animation(image_sequence, masks_sequence, 
                                 outlines_sequence=None, fps=5,
                                 output_path=None, dpi=150):
    """
    Create an animation of segmentation progress.
    
    Args:
        image_sequence (list): List of background images
        masks_sequence (list): List of segmentation masks
        outlines_sequence (list, optional): List of outline dataframes
        fps (int): Frames per second
        output_path (str, optional): Path to save animation
        dpi (int): DPI for output
    
    Returns:
        matplotlib.animation.Animation: Animation object
    """
    import matplotlib.animation as animation
    
    # Validate input
    if len(image_sequence) != len(masks_sequence):
        raise ValueError("Image and mask sequences must have the same length")
    
    if outlines_sequence is not None and len(outlines_sequence) != len(image_sequence):
        raise ValueError("Outlines sequence must have the same length as images")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Function to update each frame
    def update_frame(i):
        ax.clear()
        
        # Display background image
        if image_sequence[i].ndim == 3 and image_sequence[i].shape[2] == 3:  # RGB image
            ax.imshow(image_sequence[i])
        else:  # Grayscale image
            ax.imshow(image_sequence[i], cmap='gray')
        
        # Add mask overlay with transparency
        if masks_sequence[i] is not None and masks_sequence[i].size > 0:
            masked = np.ma.masked_where(masks_sequence[i] == 0, masks_sequence[i])
            ax.imshow(masked, cmap='viridis', alpha=0.5)
        
        # Add outlines if available
        if outlines_sequence is not None and outlines_sequence[i] is not None:
            outlines = outlines_sequence[i]
            if not outlines.empty:
                for cell_id, group in outlines.groupby('global_cell_id' if 'global_cell_id' in outlines.columns else 'cell_id'):
                    ax.plot(group['x'], group['y'], color='red', linewidth=1.0, alpha=0.8)
        
        ax.set_title(f"Frame {i+1}/{len(image_sequence)}")
        ax.axis('off')
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(image_sequence),
        interval=1000/fps, blit=False
    )
    
    # Save if output path is provided
    if output_path:
        ani.save(output_path, fps=fps, dpi=dpi)
    
    plt.close()
    return ani

def plot_cell_features(features_df, x='x', y='y', color_by=None, 
                      size_by=None, figsize=(10, 10), cmap='viridis',
                      size_scale=20, title=None):
    """
    Plot cell features in spatial coordinates.
    
    Args:
        features_df (pandas.DataFrame): DataFrame with cell features
        x (str): Column for x-coordinates
        y (str): Column for y-coordinates
        color_by (str, optional): Column to use for point colors
        size_by (str, optional): Column to use for point sizes
        figsize (tuple): Figure size
        cmap (str): Colormap for colors
        size_scale (float): Multiplier for point sizes
        title (str, optional): Plot title
    
    Returns:
        matplotlib.figure.Figure: Figure with the plot
    """
    if features_df.empty:
        return None
    
    # Check required columns
    required_cols = [x, y]
    if color_by is not None:
        required_cols.append(color_by)
    if size_by is not None:
        required_cols.append(size_by)
        
    if not all(col in features_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set point sizes
    sizes = None
    if size_by is not None:
        # Normalize sizes to 0-1 range, then scale
        min_val = features_df[size_by].min()
        max_val = features_df[size_by].max()
        if min_val != max_val:
            sizes = ((features_df[size_by] - min_val) / (max_val - min_val) * size_scale) + 5
        else:
            sizes = np.ones(len(features_df)) * size_scale
    
    # Create scatter plot
    scatter = ax.scatter(
        features_df[x], 
        features_df[y],
        c=features_df[color_by] if color_by else None,
        s=sizes if sizes is not None else size_scale,
        cmap=cmap,
        alpha=0.7
    )
    
    # Add colorbar if coloring by a variable
    if color_by:
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label(color_by)
    
    # Add legend for sizes if applicable
    if size_by:
        # Create legend for sizes
        import matplotlib.lines as mlines
        size_levels = np.linspace(features_df[size_by].min(), features_df[size_by].max(), 4)
        handles = []
        
        for level in size_levels:
            size = ((level - min_val) / (max_val - min_val) * size_scale) + 5
            handles.append(mlines.Line2D([], [], 
                                       color='black', 
                                       marker='o', 
                                       linestyle='None',
                                       markersize=np.sqrt(size/np.pi), 
                                       label=f'{level:.2f}'))
        
        ax.legend(handles=handles, title=size_by, loc='upper right')
    
    # Set title and labels
    ax.set_title(title or "Cell Features")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    # Invert y-axis since image coordinates typically have origin at top-left
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig
