"""
Utility functions for cell segmentation.

This module provides helper functions for segmenting cells and nuclei
using Cellpose and for extracting contours from segmentation masks.
"""

import numpy as np
from skimage import measure
import logging
import sys

# Simple check for cellpose availability without importing it
try:
    import importlib.util
    CELLPOSE_AVAILABLE = importlib.util.find_spec("cellpose") is not None
except ImportError:
    CELLPOSE_AVAILABLE = False

def get_cell_outlines(masks):
    """
    Extract cell outlines from a segmentation mask.
    
    Args:
        masks (numpy.ndarray): Segmentation mask with cell IDs
    
    Returns:
        list: List of dictionaries containing cell outlines
    """
    cells = []
    cell_ids = np.unique(masks)[1:]  # Skip background (0)
    
    for cell_id in cell_ids:
        cell_mask = masks == cell_id
        contours = measure.find_contours(cell_mask, 0.5)
        
        if len(contours) > 0:
            # Use the largest contour if there are multiple
            contour = max(contours, key=len)
            
            cells.append({
                'cell_id': int(cell_id),
                'x_coords': contour[:, 1],
                'y_coords': contour[:, 0],
            })
    
    return cells

def segment_chunk(chunk_data, model_instance, chunk_position=None, diameter=80,
                  flow_threshold=0.8, cellprob_threshold=-3.5, channels=[0, 0], object_type="object"):
    """
    Segment a chunk of an image using Cellpose.
    
    Args:
        chunk_data (numpy.ndarray): Chunk of image data to segment
        model_instance: Cellpose model
        chunk_position (tuple, optional): Position of chunk in the original image (y_start, x_start)
        diameter (float): Expected object diameter
        flow_threshold (float): Flow threshold for segmentation
        cellprob_threshold (float): Cell probability threshold
        channels (list): Channels to use for segmentation
        object_type (str): Type of object being segmented (for logging)
    
    Returns:
        tuple: (masks, outlines_info)
            - masks (numpy.ndarray): Segmentation mask
            - outlines_info (list): List of dictionaries with contour information
    """
    logger = logging.getLogger(__name__)
    
    # Check if cellpose is available
    if not CELLPOSE_AVAILABLE:
        logger.error("Cellpose is not available - segmentation cannot be performed")
        return np.zeros((chunk_data.shape[-2], chunk_data.shape[-1]) if chunk_data.ndim >= 2 else (0, 0), dtype=np.uint16), []
    
    if model_instance is None:
        logger.error(f"Model not provided to segment_chunk for {object_type}.")
        return np.array([], dtype=np.uint16), []

    # Handle chunk dimension issues
    if chunk_data.ndim > 2 and channels == [0, 0]:
        if chunk_data.ndim == 3:
            chunk_data = chunk_data[0]
        elif chunk_data.ndim == 4:
            chunk_data = chunk_data[0, 0]
        else:
            logger.error(f"Cannot handle {object_type} chunk dimension {chunk_data.ndim} for grayscale model.")
            return np.zeros(chunk_data.shape[-2:], dtype=np.uint16), []
    elif chunk_data.ndim < 2:
        logger.error(f"{object_type} chunk has insufficient dimensions at {chunk_position}: shape {chunk_data.shape}")
        return np.array([], dtype=np.uint16), []
        
    if chunk_data.size == 0:
        logger.warning(f"Empty {object_type} chunk at position {chunk_position}")
        return np.array([], dtype=np.uint16), []
        
    if any(s < 10 for s in chunk_data.shape[-2:]):
        logger.warning(f"{object_type} chunk spatial dimensions too small at {chunk_position}: shape {chunk_data.shape[-2:]}")
        return np.zeros(chunk_data.shape[-2:], dtype=np.uint16), []

    try:
        # Run segmentation
        masks, flows, styles = model_instance.eval(
            chunk_data,
            diameter=int(diameter),  # Cast to int to avoid type errors
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels
        )
        
        # Get outlines
        outlines_info = get_cell_outlines(masks)
        
        # Add position offset to outlines
        if chunk_position is not None:
            y_offset, x_offset = chunk_position
            for item_info in outlines_info:
                item_info['x_coords'] = np.array(item_info['x_coords']) + x_offset
                item_info['y_coords'] = np.array(item_info['y_coords']) + y_offset
                item_info['chunk_id'] = f"{y_offset}_{x_offset}"
                
        return masks.astype(np.uint16), outlines_info
        
    except Exception as e:
        logger.error(f"Error processing {object_type} chunk at {chunk_position} with shape {chunk_data.shape}: {e}", exc_info=True)
        return np.zeros(chunk_data.shape[-2:] if chunk_data.ndim >= 2 else (0, 0), dtype=np.uint16), []
