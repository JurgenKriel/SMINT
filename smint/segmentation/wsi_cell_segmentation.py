"""
Whole slide image cell segmentation module.

This module provides functions for segmenting cells in whole slide images
using Cellpose models, with support for both cell and nuclear segmentation.
"""

import dask.array as da
import tifffile
import numpy as np
import os
import pandas as pd
from skimage import measure
from skimage.exposure import rescale_intensity
import importlib
import logging
import sys

# Check if cellpose is available
CELLPOSE_AVAILABLE = False
try:
    import importlib.util
    CELLPOSE_AVAILABLE = importlib.util.find_spec("cellpose") is not None
except ImportError:
    CELLPOSE_AVAILABLE = False

if not CELLPOSE_AVAILABLE:
    logging.warning("Cellpose package not available. Some functionality will be limited.")

from datetime import datetime
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path

from .cell_utils import get_cell_outlines, segment_chunk

def setup_logger(csv_base_path):
    """
    Set up a logger for the segmentation process.
    
    Args:
        csv_base_path (str): Base path for output CSV files, used for log filename
    
    Returns:
        logging.Logger: Configured logger
    """
    log_file = f"{csv_base_path}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices=[0, 1]):
    """
    Create RGB visualization of a region of interest from a multi-channel image stack.
    
    Args:
        image_data_stack: Multi-channel image data, shape (C, Y, X)
        y_start, y_end, x_start, x_end: ROI coordinates
        vis_ch_indices: Indices of channels to use for visualization [R, G, B]
    
    Returns:
        numpy.ndarray: RGB image for visualization
    """
    if image_data_stack.ndim != 3 or image_data_stack.shape[0] < max(vis_ch_indices) + 1:
        raise ValueError(f"Image stack has shape {image_data_stack.shape}, need channels at indices {vis_ch_indices}")

    channels_data = []
    for ch_idx in vis_ch_indices:
        channels_data.append(image_data_stack[ch_idx, y_start:y_end, x_start:x_end])

    roi_ch_g_data = channels_data[0]
    roi_ch_b_data = channels_data[1] if len(channels_data) > 1 else np.zeros_like(channels_data[0])
    roi_ch_r_data = channels_data[2] if len(channels_data) > 2 else np.zeros_like(channels_data[0])

    p_low, p_high = 1, 99
    def normalize_channel(channel_data):
        if channel_data.size == 0: return channel_data
        c_min, c_max = np.percentile(channel_data, (p_low, p_high))
        return rescale_intensity(channel_data, in_range=(c_min, c_max if c_max > c_min else c_max + 1e-6), out_range=(0.0, 1.0))

    roi_ch_g_norm = normalize_channel(roi_ch_g_data)
    roi_ch_b_norm = normalize_channel(roi_ch_b_data)
    roi_ch_r_norm = normalize_channel(roi_ch_r_data)

    rgb_image = np.zeros((roi_ch_g_norm.shape[0], roi_ch_g_norm.shape[1], 3), dtype=float)

    if len(vis_ch_indices) > 2:  # R, G, B
        rgb_image[..., 0] = roi_ch_r_norm
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_b_norm
    elif len(vis_ch_indices) > 1:  # G, B
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_b_norm
    elif len(vis_ch_indices) > 0:  # G (grayscale)
        rgb_image[..., 0] = roi_ch_g_norm
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_g_norm
    return np.clip(rgb_image, 0, 1)

def visualize_roi_combined(image_data_stack, roi_position, roi_size,
                          df_cells_outlines=None, df_nuclei_outlines=None,
                          vis_ch_indices=[0, 1], figsize=(10, 10),
                          original_bg_channels_for_title=[0, 1]):
    """
    Visualize a region of interest with cell and nuclear outlines.
    
    Args:
        image_data_stack: Multi-channel image data
        roi_position: (y_start, x_start) position of ROI
        roi_size: (height, width) size of ROI
        df_cells_outlines: DataFrame of cell outlines
        df_nuclei_outlines: DataFrame of nuclear outlines
        vis_ch_indices: Indices of channels to use for visualization
        figsize: Figure size for the plot
        original_bg_channels_for_title: Original channel indices for title
    
    Returns:
        matplotlib.figure.Figure: Figure with visualization
    """
    logger = logging.getLogger(__name__)
    y_start_param, x_start_param = roi_position
    height_param, width_param = roi_size

    if image_data_stack.ndim != 3:
        logger.error(f"visualize_roi_combined expects 3D (C,Y,X) stack, got {image_data_stack.shape}")
        return None

    img_c, img_h, img_w = image_data_stack.shape

    y_start_roi = max(0, y_start_param)
    x_start_roi = max(0, x_start_param)
    y_end_roi = min(y_start_param + height_param, img_h)
    x_end_roi = min(x_start_param + width_param, img_w)

    actual_roi_height = y_end_roi - y_start_roi
    actual_roi_width = x_end_roi - x_start_roi

    if actual_roi_height <= 0 or actual_roi_width <= 0:
        logger.warning(f"ROI at {roi_position} size {roi_size} results in zero area. Stack shape: {image_data_stack.shape}")
        return None
    try:
        rgb_roi = create_rgb_roi(image_data_stack, y_start_roi, y_end_roi, x_start_roi, x_end_roi, vis_ch_indices)
    except ValueError as e:
        logger.error(f"Error creating RGB ROI: {e}")
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb_roi)

    ch_map_str_list = []
    if len(original_bg_channels_for_title) == 1:
        ch_map_str_list.append(f"Display: G=OrigCh{original_bg_channels_for_title[0]}")
    elif len(original_bg_channels_for_title) == 2:
        ch_map_str_list.append(f"Display: G=OrigCh{original_bg_channels_for_title[0]}, B=OrigCh{original_bg_channels_for_title[1]}")
    elif len(original_bg_channels_for_title) >= 3:
        ch_map_str_list.append(f"Display: R=OrigCh{original_bg_channels_for_title[0]}, G=OrigCh{original_bg_channels_for_title[1]}, B=OrigCh{original_bg_channels_for_title[2]}")

    title_bg_str = ", ".join(ch_map_str_list) if ch_map_str_list else "N/A"
    title = (f'ROI ({y_start_roi},{x_start_roi}) Size ({actual_roi_width}x{actual_roi_height}) | '
             f'{title_bg_str}')

    outline_info = []
    x_offset_for_plot = x_start_roi
    y_offset_for_plot = y_start_roi

    if df_cells_outlines is not None and not df_cells_outlines.empty:
        roi_df_cells = df_cells_outlines[
            (df_cells_outlines['x'] >= x_start_roi) & (df_cells_outlines['x'] < x_end_roi) &
            (df_cells_outlines['y'] >= y_start_roi) & (df_cells_outlines['y'] < y_end_roi)
        ].copy()
        if not roi_df_cells.empty:
            roi_df_cells['x_rel'] = roi_df_cells['x'] - x_offset_for_plot
            roi_df_cells['y_rel'] = roi_df_cells['y'] - y_offset_for_plot
            for obj_id, group in roi_df_cells.groupby('global_cell_id'):
                ax.plot(group['x_rel'], group['y_rel'], 'r-', linewidth=1.0, alpha=0.7)
            outline_info.append(f"Cells({roi_df_cells['global_cell_id'].nunique()}):Red")

    if df_nuclei_outlines is not None and not df_nuclei_outlines.empty:
        roi_df_nuclei = df_nuclei_outlines[
            (df_nuclei_outlines['x'] >= x_start_roi) & (df_nuclei_outlines['x'] < x_end_roi) &
            (df_nuclei_outlines['y'] >= y_start_roi) & (df_nuclei_outlines['y'] < y_end_roi)
        ].copy()
        if not roi_df_nuclei.empty:
            roi_df_nuclei['x_rel'] = roi_df_nuclei['x'] - x_offset_for_plot
            roi_df_nuclei['y_rel'] = roi_df_nuclei['y'] - y_offset_for_plot
            for obj_id, group in roi_df_nuclei.groupby('global_cell_id'):
                ax.plot(group['x_rel'], group['y_rel'], 'cyan-', linewidth=1.0, alpha=0.7)
            outline_info.append(f"Nuclei({roi_df_nuclei['global_cell_id'].nunique()}):Cyan")

    if outline_info:
        title += " | Outlines: " + ", ".join(outline_info)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    return fig

def process_large_image(
    image_path,
    csv_base_path,
    chunk_size=(2048, 2048),
    # Cell Model parameters
    cell_model_path="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700",
    cells_diameter=120.0,
    cells_flow_threshold=0.4,
    cells_cellprob_threshold=-1.5,
    cells_channels=[1, 2],
    # Nuclei Model parameters
    nuclei_model_path="nuclei",
    nuclei_diameter=40.0,
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=-1.2,
    nuclei_channels=[2, 0],
    # Adaptive Nuclei Segmentation parameters
    enable_adaptive_nuclei=False,
    nuclei_adaptive_cellprob_lower_limit=-6.0,
    nuclei_adaptive_cellprob_step_decrement=0.2,
    nuclei_max_adaptive_attempts=3,
    adaptive_nuclei_trigger_ratio=0.05,
    # Visualization parameters
    visualize=True,
    visualize_output_dir=None,
    num_visualize_chunks=5,
    visualize_roi_size=(2024, 2024),
    vis_bg_channel_indices=[0, 1],
    # Live update parameters
    live_update_image_path=None,
    tile_info_file_for_viewer=None
):
    """
    Process a large image for cell and nuclear segmentation.
    
    Args:
        image_path (str): Path to the input image file
        csv_base_path (str): Base path for output CSV files
        chunk_size (tuple): Size of chunks for processing
        cell_model_path (str): Path to the cell segmentation model
        cells_diameter (float): Diameter of cells for segmentation
        cells_flow_threshold (float): Flow threshold for cell segmentation
        cells_cellprob_threshold (float): Cell probability threshold
        cells_channels (list): Channels to use for cell segmentation
        nuclei_model_path (str): Path to the nuclei segmentation model
        nuclei_diameter (float): Diameter of nuclei for segmentation
        nuclei_flow_threshold (float): Flow threshold for nuclei segmentation
        nuclei_cellprob_threshold (float): Nuclei probability threshold
        nuclei_channels (list): Channels to use for nuclei segmentation
        enable_adaptive_nuclei (bool): Enable adaptive nuclei segmentation
        nuclei_adaptive_cellprob_lower_limit (float): Lower limit for adaptive cellprob
        nuclei_adaptive_cellprob_step_decrement (float): Decrement step for adaptive cellprob
        nuclei_max_adaptive_attempts (int): Maximum attempts for adaptive segmentation
        adaptive_nuclei_trigger_ratio (float): Trigger ratio for adaptive segmentation
        visualize (bool): Enable visualization
        visualize_output_dir (str): Directory for visualization output
        num_visualize_chunks (int): Number of chunks to visualize
        visualize_roi_size (tuple): Size of ROI for visualization
        vis_bg_channel_indices (list): Channel indices for visualization
        live_update_image_path (str): Path for live update image
        tile_info_file_for_viewer (str): Path for tile info file
    
    Returns:
        dict: Results containing paths to output files and segmentation statistics
    """
    # Setup logging
    logger = setup_logger(csv_base_path)
    
    start_time = datetime.now()
    logger.info("="*50)
    logger.info(f"Starting DUAL MODEL image processing at {start_time}")
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output CSV base: {csv_base_path}")
    logger.info(f"Chunk size: {chunk_size}")

    logger.info(f"Cell Model Path: {cell_model_path}")
    logger.info(f"Cells Diameter: {cells_diameter}, Flow: {cells_flow_threshold}, Prob: {cells_cellprob_threshold}, Channels: {cells_channels}")

    logger.info(f"Nuclei Model Path: {nuclei_model_path}")
    logger.info(f"Nuclei Diameter: {nuclei_diameter}, Flow: {nuclei_flow_threshold}, Initial Prob: {nuclei_cellprob_threshold}, Channels: {nuclei_channels}")

    if enable_adaptive_nuclei:
        logger.info("Adaptive Nuclei Segmentation (Cellprob DECREASING): ENABLED")
        logger.info(f"  Cellprob Lower Limit: {nuclei_adaptive_cellprob_lower_limit}, Step Decrement: {nuclei_adaptive_cellprob_step_decrement}, Max Attempts: {nuclei_max_adaptive_attempts}")
        logger.info(f"  Trigger Ratio (nuclei/cells): < {adaptive_nuclei_trigger_ratio}")
    else:
        logger.info("Adaptive Nuclei Segmentation: DISABLED")

    logger.info(f"Visualization Enabled: {visualize}")
    if visualize:
        if visualize_output_dir is None:
            visualize_output_dir = os.path.join(os.path.dirname(csv_base_path) or ".", "visualizations")
        logger.info(f"Visualization Output Dir: {visualize_output_dir}")
        logger.info(f"Num Chunks to Visualize: {num_visualize_chunks}, ROI Size: {visualize_roi_size}")
        logger.info(f"Visualization BG Channels (Original TIFF, 0-indexed): {vis_bg_channel_indices}")

    if live_update_image_path:
        logger.info(f"Live update image will be saved to: {live_update_image_path}")
        live_update_dir = os.path.dirname(live_update_image_path)
        if live_update_dir and not os.path.exists(live_update_dir):
            try:
                os.makedirs(live_update_dir)
                logger.info(f"Created directory for live update image: {live_update_dir}")
            except Exception as e:
                logger.error(f"Could not create directory {live_update_dir} for live update image: {e}. Live updates disabled.")
                live_update_image_path = None

    if tile_info_file_for_viewer:
        logger.info(f"Tile info for live viewer will be written to: {tile_info_file_for_viewer}")
        tile_info_dir = os.path.dirname(tile_info_file_for_viewer)
        if tile_info_dir and not os.path.exists(tile_info_dir):
            try:
                os.makedirs(tile_info_dir)
                logger.info(f"Created directory for tile info: {tile_info_dir}")
            except Exception as e:
                logger.error(f"Could not create directory {tile_info_dir} for tile info: {e}. Tile info disabled.")
                tile_info_file_for_viewer = None

    # Load the whole slide image
    try:
        logger.info(f"Loading image with tifffile: {image_path}")
        with tifffile.TiffFile(image_path) as tif:
            metadata = {}
            if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                metadata['ome'] = tif.ome_metadata
                logger.info(f"OME Metadata found, size: {len(str(tif.ome_metadata))} characters")
            else:
                logger.info("No OME metadata found")
            
            # Get image data - handle different dimensions
            if len(tif.pages) > 1:
                logger.info(f"Multiple pages found ({len(tif.pages)})")
                
                # Check if there are separate series for channels
                if tif.series and len(tif.series) > 1:
                    logger.info(f"Multiple series found ({len(tif.series)})")
                    
                    # Assume first series is the one we want or find the largest
                    series_to_use = max(tif.series, key=lambda s: s.shape[0] * s.shape[1] if len(s.shape) >= 2 else 0)
                    
                    if len(series_to_use.shape) not in (3, 4, 5):
                        logger.error(f"Expected 3D (CYX), 4D (ZCYX or TCYX), or 5D (TZCYX) data, got shape {series_to_use.shape}")
                        return {"error": "Unsupported image dimensions"}
                    
                    data = series_to_use.asarray()
                    logger.info(f"Loaded series data with shape {data.shape}")
                else:
                    # No series, stack pages as T dimension
                    data = np.stack([page.asarray() for page in tif.pages])
                    logger.info(f"Stacked pages into array with shape {data.shape}")
            else:
                # Single page
                data = tif.pages[0].asarray()
                logger.info(f"Loaded single page data with shape {data.shape}")
                
            # Normalize dimensions to ensure CYX format
            if data.ndim == 2:  # YX
                data = data[np.newaxis, :, :]  # Convert to CYX with C=1
                logger.info(f"Converted 2D YX to 3D CYX with shape {data.shape}")
            elif data.ndim == 3:
                # Could be CYX, ZYX, or TYX - assume CYX if first dim is small
                if data.shape[0] <= 10:  # Reasonable number of channels
                    logger.info(f"Assuming 3D data with shape {data.shape} is CYX")
                else:
                    # Probably ZYX, use single Z-slice as grayscale
                    data = data[data.shape[0]//2][np.newaxis, :, :]  # Middle Z-slice as single channel
                    logger.info(f"Converted possible ZYX to CYX with shape {data.shape} (middle Z-slice)")
            elif data.ndim == 4:  # Could be ZCYX, TZYX, or TCYX
                if data.shape[0] <= 10 and data.shape[1] <= 10:  # Small first dims
                    # Probably TCYX - take first timepoint
                    data = data[0]
                    logger.info(f"Extracted first timepoint from probable TCYX, new shape {data.shape}")
                elif data.shape[0] <= 10:  # T/Z is small, C could be large
                    # Probably ZCYX
                    data = data[:, data.shape[1]//2, :, :]  # Take middle C as channels
                    logger.info(f"Extracted channels from middle C-position of probable ZCYX, new shape {data.shape}")
                else:
                    # Complex case, take middle Z/T and all channels
                    data = data[data.shape[0]//2]
                    logger.info(f"Extracted middle T/Z from 4D data, new shape {data.shape}")
            elif data.ndim == 5:  # TZCYX
                # Take first T, middle Z
                data = data[0, data.shape[1]//2]
                logger.info(f"Extracted first T, middle Z from 5D TZCYX data, new shape {data.shape}")
            
            # Final verification that we have CYX
            if data.ndim == 3:
                if data.shape[0] > 10:
                    logger.warning(f"First dimension (C) is unusually large: {data.shape[0]}. This might not be CYX format.")
                image_data_stack = data
            else:
                logger.error(f"Failed to convert data to CYX format, current shape: {data.shape}")
                return {"error": "Could not convert data to required CYX format"}
                
            logger.info(f"Final image data stack shape: {image_data_stack.shape}")
            
    except Exception as e:
        logger.error(f"Error loading image: {e}", exc_info=True)
        return {"error": f"Failed to load image: {str(e)}"}

    # Load models
    cell_model = None
    nuclei_model = None
    
    # Check if cellpose is available
    if not CELLPOSE_AVAILABLE:
        logger.warning("Cellpose package not available. Skipping model loading.")
        return {"error": "Cellpose package is required but not installed"}
        
    try:
        # Import locally to avoid circular imports
        from cellpose import models
        
        # Check for GPU
        use_gpu = models.use_gpu()
        logger.info(f"GPU available and will be used: {use_gpu}")
    
        # Load cell model
        if os.path.exists(cell_model_path) or cell_model_path.lower() in ["cyto", "nuclei", "cyto2"]:
            logger.info(f"Loading cell model from: {cell_model_path}")
            cell_model = models.CellposeModel(gpu=use_gpu, model_type=cell_model_path)
            logger.info("Cell model loaded successfully")
        else:
            cell_model = None
            logger.warning(f"Cell model path not found, skipping cell segmentation: {cell_model_path}")
        
        # Load nuclei model
        if os.path.exists(nuclei_model_path) or nuclei_model_path.lower() in ["cyto", "nuclei", "cyto2"]:
            logger.info(f"Loading nuclei model from: {nuclei_model_path}")
            nuclei_model = models.CellposeModel(gpu=use_gpu, model_type=nuclei_model_path)
            logger.info("Nuclei model loaded successfully")
        else:
            nuclei_model = None
            logger.warning(f"Nuclei model path not found, skipping nuclei segmentation: {nuclei_model_path}")
        
        if cell_model is None and nuclei_model is None:
            logger.error("Both cell and nuclei models failed to load. Cannot proceed.")
            return {"error": "Failed to load any segmentation models"}
            
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return {"error": f"Failed to load models: {str(e)}"}

    # Create output directories
    csv_base_dir = os.path.dirname(csv_base_path)
    if csv_base_dir and not os.path.exists(csv_base_dir):
        os.makedirs(csv_base_dir, exist_ok=True)
    
    if visualize and visualize_output_dir:
        os.makedirs(visualize_output_dir, exist_ok=True)

    # Process image in chunks
    c, img_h, img_w = image_data_stack.shape
    logger.info(f"Image dimensions: {img_h}x{img_w} with {c} channels")
    
    chunk_h, chunk_w = chunk_size
    
    # Calculate number of chunks
    n_chunks_h = (img_h + chunk_h - 1) // chunk_h
    n_chunks_w = (img_w + chunk_w - 1) // chunk_w
    total_chunks = n_chunks_h * n_chunks_w
    
    logger.info(f"Processing in {total_chunks} chunks ({n_chunks_h}x{n_chunks_w}) of size {chunk_h}x{chunk_w}")
    
    # Initialize DataFrames for cell and nuclei outlines
    all_cells_outlines = []
    all_nuclei_outlines = []
    
    # Track statistics
    stats = {
        'processed_chunks': 0,
        'total_cells': 0,
        'total_nuclei': 0,
        'start_time': start_time,
        'elapsed_time': 0
    }
    
    # Process image in chunks
    cell_id_counter = 0
    nuclei_id_counter = 0
    
    chunks_to_visualize = []
    if visualize and num_visualize_chunks > 0:
        # Select random chunks for visualization
        all_chunk_indices = [(y, x) for y in range(n_chunks_h) for x in range(n_chunks_w)]
        if num_visualize_chunks < total_chunks:
            import random
            chunks_to_visualize = random.sample(all_chunk_indices, num_visualize_chunks)
        else:
            chunks_to_visualize = all_chunk_indices
    
    for chunk_y in range(n_chunks_h):
        for chunk_x in range(n_chunks_w):
            chunk_start_time = time.time()
            
            # Calculate chunk boundaries
            y_start = chunk_y * chunk_h
            y_end = min(y_start + chunk_h, img_h)
            x_start = chunk_x * chunk_w
            x_end = min(x_start + chunk_w, img_w)
            
            # Skip if chunk is empty
            if y_start >= img_h or x_start >= img_w:
                continue
                
            logger.info(f"Processing chunk [{chunk_y},{chunk_x}] at position ({y_start},{x_start}) size {y_end-y_start}x{x_end-x_start}")
            
            # Extract chunk
            chunk_data = image_data_stack[:, y_start:y_end, x_start:x_end]
            
            # Process chunk for cells
            cells_masks = None
            cells_outlines = []
            if cell_model:
                try:
                    logger.info(f"Running cell segmentation on chunk with shape {chunk_data.shape}")
                    cells_masks, cells_outlines = segment_chunk(
                        chunk_data, 
                        cell_model, 
                        chunk_position=(y_start, x_start),
                        diameter=cells_diameter,
                        flow_threshold=cells_flow_threshold,
                        cellprob_threshold=cells_cellprob_threshold,
                        channels=cells_channels,
                        object_type="cell"
                    )
                    
                    # Add global cell ID
                    for outline in cells_outlines:
                        cell_id_counter += 1
                        outline['global_cell_id'] = cell_id_counter
                    
                    logger.info(f"Found {len(cells_outlines)} cells in chunk")
                    stats['total_cells'] += len(cells_outlines)
                    all_cells_outlines.extend(cells_outlines)
                except Exception as e:
                    logger.error(f"Error in cell segmentation for chunk [{chunk_y},{chunk_x}]: {e}", exc_info=True)
            
            # Process chunk for nuclei
            nuclei_masks = None
            nuclei_outlines = []
            
            if nuclei_model:
                try:
                    curr_nuclei_cellprob_threshold = nuclei_cellprob_threshold
                    
                    logger.info(f"Running nuclei segmentation on chunk with shape {chunk_data.shape}")
                    nuclei_masks, nuclei_outlines = segment_chunk(
                        chunk_data, 
                        nuclei_model, 
                        chunk_position=(y_start, x_start),
                        diameter=nuclei_diameter,
                        flow_threshold=nuclei_flow_threshold,
                        cellprob_threshold=curr_nuclei_cellprob_threshold,
                        channels=nuclei_channels,
                        object_type="nucleus"
                    )
                    
                    # Adaptive nuclei segmentation
                    if enable_adaptive_nuclei and cells_outlines:
                        nuclei_to_cells_ratio = len(nuclei_outlines) / len(cells_outlines)
                        
                        attempts = 0
                        while (nuclei_to_cells_ratio < adaptive_nuclei_trigger_ratio and 
                               attempts < nuclei_max_adaptive_attempts and 
                               curr_nuclei_cellprob_threshold > nuclei_adaptive_cellprob_lower_limit):
                            
                            # Decrease the threshold (less restrictive)
                            curr_nuclei_cellprob_threshold -= nuclei_adaptive_cellprob_step_decrement
                            curr_nuclei_cellprob_threshold = max(curr_nuclei_cellprob_threshold, nuclei_adaptive_cellprob_lower_limit)
                            
                            logger.info(f"Adaptive nuclei segmentation: attempt {attempts+1}, new threshold {curr_nuclei_cellprob_threshold}")
                            
                            # Try again with new threshold
                            nuclei_masks, nuclei_outlines = segment_chunk(
                                chunk_data, 
                                nuclei_model, 
                                chunk_position=(y_start, x_start),
                                diameter=nuclei_diameter,
                                flow_threshold=nuclei_flow_threshold,
                                cellprob_threshold=curr_nuclei_cellprob_threshold,
                                channels=nuclei_channels,
                                object_type="nucleus"
                            )
                            
                            # Check if we improved
                            if nuclei_outlines:
                                nuclei_to_cells_ratio = len(nuclei_outlines) / len(cells_outlines)
                                logger.info(f"New nuclei/cells ratio: {nuclei_to_cells_ratio:.3f} ({len(nuclei_outlines)}/{len(cells_outlines)})")
                            else:
                                nuclei_to_cells_ratio = 0
                                
                            attempts += 1
                            
                        if attempts > 0:
                            logger.info(f"Adaptive nuclei segmentation completed after {attempts} attempts. "
                                       f"Final threshold: {curr_nuclei_cellprob_threshold}, "
                                       f"Final ratio: {nuclei_to_cells_ratio:.3f}")
                    
                    # Add global cell ID
                    for outline in nuclei_outlines:
                        nuclei_id_counter += 1
                        outline['global_cell_id'] = nuclei_id_counter
                    
                    logger.info(f"Found {len(nuclei_outlines)} nuclei in chunk")
                    stats['total_nuclei'] += len(nuclei_outlines)
                    all_nuclei_outlines.extend(nuclei_outlines)
                except Exception as e:
                    logger.error(f"Error in nuclei segmentation for chunk [{chunk_y},{chunk_x}]: {e}", exc_info=True)
            
            # Create visualization if requested
            if visualize and (chunk_y, chunk_x) in chunks_to_visualize:
                try:
                    # Create visualization of the chunk with outlines
                    if not os.path.exists(visualize_output_dir):
                        os.makedirs(visualize_output_dir, exist_ok=True)
                        
                    # Convert outlines to DataFrame for visualization
                    df_cells = None
                    if cells_outlines:
                        cells_data = []
                        for cell in cells_outlines:
                            for x, y in zip(cell['x_coords'], cell['y_coords']):
                                cells_data.append({
                                    'global_cell_id': cell['global_cell_id'],
                                    'x': x,
                                    'y': y
                                })
                        df_cells = pd.DataFrame(cells_data)
                    
                    df_nuclei = None
                    if nuclei_outlines:
                        nuclei_data = []
                        for nucleus in nuclei_outlines:
                            for x, y in zip(nucleus['x_coords'], nucleus['y_coords']):
                                nuclei_data.append({
                                    'global_cell_id': nucleus['global_cell_id'],
                                    'x': x,
                                    'y': y
                                })
                        df_nuclei = pd.DataFrame(nuclei_data)
                    
                    # Create visualization
                    fig = visualize_roi_combined(
                        image_data_stack,
                        roi_position=(y_start, x_start),
                        roi_size=(y_end - y_start, x_end - x_start),
                        df_cells_outlines=df_cells,
                        df_nuclei_outlines=df_nuclei,
                        vis_ch_indices=vis_bg_channel_indices,
                        figsize=(10, 10)
                    )
                    
                    if fig:
                        out_path = os.path.join(visualize_output_dir, f"chunk_{chunk_y}_{chunk_x}.png")
                        fig.savefig(out_path, dpi=150)
                        plt.close(fig)
                        logger.info(f"Saved visualization to {out_path}")
                    else:
                        logger.warning(f"Failed to create visualization for chunk {chunk_y}_{chunk_x}")
                        
                except Exception as e:
                    logger.error(f"Error creating visualization for chunk [{chunk_y},{chunk_x}]: {e}", exc_info=True)
            
            # Save live update if requested
            if live_update_image_path and (cells_masks is not None or nuclei_masks is not None):
                try:
                    # Create visualization for live update
                    display_ch = min(vis_bg_channel_indices[0], chunk_data.shape[0]-1) if vis_bg_channel_indices else 0
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(chunk_data[display_ch], cmap='gray')
                    
                    if cells_masks is not None and cells_masks.size > 0:
                        plt.contour(cells_masks, colors='r', linewidths=0.5)
                        
                    if nuclei_masks is not None and nuclei_masks.size > 0:
                        plt.contour(nuclei_masks, colors='cyan', linewidths=0.5)
                        
                    cell_count_str = f"{len(cells_outlines)} cells" if cells_outlines else "No cells"
                    nuclei_count_str = f"{len(nuclei_outlines)} nuclei" if nuclei_outlines else "No nuclei"
                    plt.title(f"Chunk ({y_start},{x_start}) - {cell_count_str}, {nuclei_count_str}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(live_update_image_path, dpi=150)
                    plt.close()
                    
                    logger.info(f"Saved live update image to {live_update_image_path}")
                except Exception as e:
                    logger.error(f"Error creating live update image: {e}")
            
            # Update tile info for viewer if requested
            if tile_info_file_for_viewer:
                try:
                    # Save current tile info for live viewer
                    # Format: y,x,tile_h,tile_w,scan_h,scan_w,seg_filename
                    seg_filename = os.path.basename(live_update_image_path) if live_update_image_path else "live_update.png"
                    with open(tile_info_file_for_viewer, 'w') as f:
                        f.write(f"{y_start},{x_start},{y_end-y_start},{x_end-x_start},{img_h},{img_w},{seg_filename}")
                    logger.info(f"Updated tile info in {tile_info_file_for_viewer}")
                except Exception as e:
                    logger.error(f"Error updating tile info: {e}")
            
            # Update stats
            stats['processed_chunks'] += 1
            chunk_time = time.time() - chunk_start_time
            logger.info(f"Chunk processed in {chunk_time:.1f} seconds")
            
            # Provide progress update
            progress = stats['processed_chunks'] / total_chunks * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            remaining = (elapsed / stats['processed_chunks']) * (total_chunks - stats['processed_chunks']) if stats['processed_chunks'] > 0 else 0
            
            logger.info(f"Progress: {progress:.1f}% ({stats['processed_chunks']}/{total_chunks} chunks), "
                      f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
    
    # Create DataFrames and save to CSV
    try:
        # Save cells outlines
        if all_cells_outlines:
            cells_data = []
            for cell in all_cells_outlines:
                for x, y in zip(cell['x_coords'], cell['y_coords']):
                    cells_data.append({
                        'global_cell_id': cell['global_cell_id'],
                        'chunk_id': cell.get('chunk_id', ""),
                        'x': int(x),
                        'y': int(y)
                    })
            df_cells = pd.DataFrame(cells_data)
            cells_csv_path = f"{csv_base_path}_cells.csv"
            df_cells.to_csv(cells_csv_path, index=False)
            logger.info(f"Saved {len(df_cells)} cell outline points for {stats['total_cells']} cells to {cells_csv_path}")
        else:
            cells_csv_path = None
            logger.warning("No cell outlines found - CSV not created")
        
        # Save nuclei outlines
        if all_nuclei_outlines:
            nuclei_data = []
            for nucleus in all_nuclei_outlines:
                for x, y in zip(nucleus['x_coords'], nucleus['y_coords']):
                    nuclei_data.append({
                        'global_cell_id': nucleus['global_cell_id'],
                        'chunk_id': nucleus.get('chunk_id', ""),
                        'x': int(x),
                        'y': int(y)
                    })
            df_nuclei = pd.DataFrame(nuclei_data)
            nuclei_csv_path = f"{csv_base_path}_nuclei.csv"
            df_nuclei.to_csv(nuclei_csv_path, index=False)
            logger.info(f"Saved {len(df_nuclei)} nuclei outline points for {stats['total_nuclei']} nuclei to {nuclei_csv_path}")
        else:
            nuclei_csv_path = None
            logger.warning("No nuclei outlines found - CSV not created")
            
    except Exception as e:
        logger.error(f"Error saving CSV files: {e}", exc_info=True)
        cells_csv_path = nuclei_csv_path = None
    
    # Calculate final stats
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    stats['end_time'] = end_time
    stats['total_time'] = total_time
    
    logger.info("="*50)
    logger.info(f"Processing completed at {end_time}")
    logger.info(f"Total time: {total_time:.1f} seconds")
    logger.info(f"Processed {stats['processed_chunks']}/{total_chunks} chunks")
    logger.info(f"Found {stats['total_cells']} cells and {stats['total_nuclei']} nuclei")
    logger.info("="*50)
    
    # Return results
    return {
        'cells_csv_path': cells_csv_path,
        'nuclei_csv_path': nuclei_csv_path,
        'total_cells': stats['total_cells'],
        'total_nuclei': stats['total_nuclei'],
        'processing_time': total_time,
        'start_time': start_time,
        'end_time': end_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a large image for cell and nuclear segmentation")
    parser.add_argument("--image_path", required=True, help="Path to the input image file")
    parser.add_argument("--csv_base_path", required=True, help="Base path for output CSV files")
    parser.add_argument("--chunk_size", type=int, nargs=2, default=[2048, 2048], help="Size of chunks for processing (height width)")
    parser.add_argument("--cell_model_path", default="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700", help="Path to the cell segmentation model")
    parser.add_argument("--cells_diameter", type=float, default=120.0, help="Diameter of cells for segmentation")
    parser.add_argument("--nuclei_model_path", default="nuclei", help="Path to the nuclei segmentation model")
    parser.add_argument("--nuclei_diameter", type=float, default=40.0, help="Diameter of nuclei for segmentation")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--visualize_output_dir", help="Directory for visualization output")
    parser.add_argument("--live_update_image_path", help="Path for live update image")
    parser.add_argument("--tile_info_file_for_viewer", help="Path for tile info file")
    
    args = parser.parse_args()
    
    process_large_image(
        image_path=args.image_path,
        csv_base_path=args.csv_base_path,
        chunk_size=tuple(args.chunk_size),
        cell_model_path=args.cell_model_path,
        cells_diameter=args.cells_diameter,
        nuclei_model_path=args.nuclei_model_path,
        nuclei_diameter=args.nuclei_diameter,
        visualize=args.visualize,
        visualize_output_dir=args.visualize_output_dir,
        live_update_image_path=args.live_update_image_path,
        tile_info_file_for_viewer=args.tile_info_file_for_viewer
    )
