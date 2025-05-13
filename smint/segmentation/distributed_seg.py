"""
Distributed segmentation using Dask.

This module provides functions for distributed segmentation of large images
using Dask for parallel processing across multiple GPUs.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import dask
import dask.array as da
import tifffile
import matplotlib.pyplot as plt
import importlib.util

# Check for required dependencies
DISTRIBUTED_AVAILABLE = importlib.util.find_spec("distributed") is not None
DASK_CUDA_AVAILABLE = importlib.util.find_spec("dask_cuda") is not None
CELLPOSE_AVAILABLE = importlib.util.find_spec("cellpose") is not None

# Log availability of optional dependencies
logger = logging.getLogger(__name__)
if not DISTRIBUTED_AVAILABLE:
    logger.warning("'distributed' package not available. Distributed processing will be limited.")
if not DASK_CUDA_AVAILABLE:
    logger.warning("'dask_cuda' package not available. GPU acceleration will not be available.")
if not CELLPOSE_AVAILABLE:
    logger.warning("'cellpose' package not available. Cell segmentation functionality will be limited.")

# Import optional dependencies only if available
if DISTRIBUTED_AVAILABLE:
    from distributed import Client, LocalCluster
if DASK_CUDA_AVAILABLE:
    from dask_cuda import LocalCUDACluster
if CELLPOSE_AVAILABLE:
    from cellpose import models
from pathlib import Path

from ..utils.config import load_config
from .cell_utils import segment_chunk, get_cell_outlines
from .postprocess import extract_contours, save_masks

def setup_logging(output_dir=None):
    """
    Set up logging configuration.
    
    Args:
        output_dir (str, optional): Directory to save log file. If None, log to console only.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('smint.distributed_seg')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f'distributed_seg_{time.strftime("%Y%m%d-%H%M%S")}.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_dask_client(n_workers=None, use_gpu=True, memory_limit='16GB'):
    """
    Set up a Dask distributed client.
    
    Args:
        n_workers (int, optional): Number of workers. If None, use all available resources.
        use_gpu (bool): Whether to use GPU for computation.
        memory_limit (str): Memory limit per worker.
    
    Returns:
        distributed.Client: Dask client
    """
    logger = logging.getLogger('smint.distributed_seg')
    
    if use_gpu:
        try:
            cluster = LocalCUDACluster(
                n_workers=n_workers,
                threads_per_worker=1,
                memory_limit=memory_limit
            )
            logger.info(f"Created LocalCUDACluster with {cluster.n_workers} workers")
        except Exception as e:
            logger.warning(f"Failed to create LocalCUDACluster: {e}. Falling back to LocalCluster.")
            cluster = LocalCluster(
                n_workers=n_workers or os.cpu_count(),
                threads_per_worker=1,
                memory_limit=memory_limit
            )
    else:
        cluster = LocalCluster(
            n_workers=n_workers or os.cpu_count(),
            threads_per_worker=1,
            memory_limit=memory_limit
        )
        logger.info(f"Created LocalCluster with {cluster.n_workers} workers")
    
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    
    return client

def load_image_as_dask_array(image_path, chunk_size=(2048, 2048)):
    """
    Load a large image as a Dask array.
    
    Args:
        image_path (str): Path to the image file.
        chunk_size (tuple): Size of chunks to split the image into.
    
    Returns:
        dask.array.Array: Dask array of the image.
    """
    logger = logging.getLogger('smint.distributed_seg')
    logger.info(f"Loading image: {image_path}")
    
    try:
        with tifffile.TiffFile(image_path) as tif:
            # Check if OME-TIFF
            is_ome = tif.is_ome
            logger.info(f"Image is {'OME-TIFF' if is_ome else 'regular TIFF'}")
            
            # Get image shape and dtype
            if len(tif.pages) > 1:
                # Handle multi-page TIFF
                logger.info(f"Multi-page TIFF with {len(tif.pages)} pages")
                
                # Get first page shape
                page_shape = tif.pages[0].shape
                dtype = tif.pages[0].dtype
                
                if tif.series:
                    # Get the largest series
                    series = max(tif.series, key=lambda s: s.shape[0] * s.shape[1])
                    dask_array = da.from_array(series.asarray(), chunks=chunk_size)
                else:
                    # Create array from all pages
                    dask_array = da.stack([da.from_array(page.asarray(), chunks=chunk_size) 
                                          for page in tif.pages])
            else:
                # Handle single-page TIFF
                page = tif.pages[0]
                dask_array = da.from_array(page.asarray(), chunks=chunk_size)
            
            logger.info(f"Created Dask array with shape {dask_array.shape} and chunks {dask_array.chunks}")
            return dask_array
    
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise

def preprocess_image(image_array, preprocessing_func=None):
    """
    Preprocess the image array.
    
    Args:
        image_array (dask.array.Array): The image array to preprocess.
        preprocessing_func (callable, optional): Function to preprocess the image.
    
    Returns:
        dask.array.Array: Preprocessed image array.
    """
    logger = logging.getLogger('smint.distributed_seg')
    
    # If no preprocessing function is provided, return the original array
    if preprocessing_func is None:
        return image_array
    
    logger.info("Applying preprocessing function")
    
    try:
        preprocessed_array = preprocessing_func(image_array)
        logger.info(f"Preprocessed array shape: {preprocessed_array.shape}")
        return preprocessed_array
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return image_array

def load_segmentation_models(model_paths):
    """
    Load cellpose models.
    
    Args:
        model_paths (list): List of paths to model files or built-in model names.
    
    Returns:
        dict: Dictionary of loaded models.
    """
    logger = logging.getLogger('smint.distributed_seg')
    models_dict = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path) if os.path.exists(model_path) else model_path
        logger.info(f"Loading model: {model_name} from {model_path}")
        
        try:
            # Check if GPU is available
            use_gpu = models.use_gpu()
            logger.info(f"Using GPU for model {model_name}: {use_gpu}")
            
            # Load the model
            model = models.CellposeModel(gpu=use_gpu, model_type=model_path)
            models_dict[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
    
    return models_dict

def segment_image_distributed(image_array, models_dict, output_dir, 
                             model_params=None, tile_info_path=None, 
                             live_update_image_path=None):
    """
    Segment an image using distributed computing.
    
    Args:
        image_array (dask.array.Array): The image array to segment.
        models_dict (dict): Dictionary of loaded Cellpose models.
        output_dir (str): Directory to save segmentation results.
        model_params (dict, optional): Dictionary of model parameters.
        tile_info_path (str, optional): Path to save tile information for live viewer.
        live_update_image_path (str, optional): Path to save live update images.
    
    Returns:
        dict: Dictionary of segmentation results.
    """
    logger = logging.getLogger('smint.distributed_seg')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default model parameters
    if model_params is None:
        model_params = {
            'diameter': 30.0,
            'flow_threshold': 0.4,
            'cellprob_threshold': -1.0,
            'channels': [0, 0]
        }
    
    # Initialize results container
    results = {
        'masks': [],
        'outlines': [],
        'tile_positions': []
    }
    
    # Get image shape
    image_shape = image_array.shape
    logger.info(f"Processing image with shape {image_shape}")
    
    # Process each chunk in the dask array
    chunk_coords = []
    for chunk_idx, chunk_slices in enumerate(da.core.slices_from_chunks(image_array.chunks)):
        if len(chunk_slices) > 2:  # For multi-dimensional arrays, focus on spatial dimensions
            y_slice, x_slice = chunk_slices[-2:]
        else:
            y_slice, x_slice = chunk_slices
        
        y_start, y_end = y_slice.start, y_slice.stop
        x_start, x_end = x_slice.start, x_slice.stop
        
        # Skip chunks outside image bounds
        if y_start >= image_shape[-2] or x_start >= image_shape[-1]:
            continue
        
        chunk_coords.append((chunk_idx, y_start, y_end, x_start, x_end))
    
    logger.info(f"Found {len(chunk_coords)} chunks to process")
    
    # Process chunks in parallel using dask
    futures = []
    client = dask.distributed.get_client()
    
    for chunk_idx, y_start, y_end, x_start, x_end in chunk_coords:
        # Extract chunk with proper handling of dimensionality
        if len(image_array.shape) > 2:
            chunk_data = image_array[..., y_start:y_end, x_start:x_end]
        else:
            chunk_data = image_array[y_start:y_end, x_start:x_end]
        
        # Create a future for processing this chunk
        for model_name, model in models_dict.items():
            future = client.submit(
                process_chunk,
                chunk_data=chunk_data,
                y_start=y_start,
                x_start=x_start,
                model=model,
                model_name=model_name,
                model_params=model_params,
                output_dir=output_dir,
                tile_info_path=tile_info_path,
                live_update_image_path=live_update_image_path
            )
            futures.append(future)
    
    # Collect results
    all_results = []
    for future in dask.distributed.as_completed(futures):
        try:
            result = future.result()
            all_results.append(result)
            logger.info(f"Completed chunk at {result['position']}")
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
    
    # Combine results
    logger.info(f"Successfully processed {len(all_results)} chunks")
    
    # Save combined results
    combined_df = pd.concat([pd.DataFrame(r['outlines']) for r in all_results if r['outlines']], ignore_index=True)
    combined_output_path = os.path.join(output_dir, "combined_cell_outlines.csv")
    combined_df.to_csv(combined_output_path, index=False)
    logger.info(f"Saved combined outlines to {combined_output_path}")
    
    return {
        'output_dir': output_dir,
        'combined_outlines_path': combined_output_path,
        'all_chunk_results': all_results
    }

def process_chunk(chunk_data, y_start, x_start, model, model_name, 
                 model_params, output_dir, tile_info_path=None, 
                 live_update_image_path=None):
    """
    Process a single chunk of the image.
    
    Args:
        chunk_data: The chunk of image data to process.
        y_start (int): Y-start position of the chunk in the original image.
        x_start (int): X-start position of the chunk in the original image.
        model: The Cellpose model to use.
        model_name (str): Name of the model.
        model_params (dict): Model parameters.
        output_dir (str): Directory to save results.
        tile_info_path (str, optional): Path to save tile information.
        live_update_image_path (str, optional): Path to save live visualization.
    
    Returns:
        dict: Results for this chunk.
    """
    start_time = time.time()
    
    # Compute the chunk data if it's a dask array
    if isinstance(chunk_data, da.Array):
        chunk_data = chunk_data.compute()
    
    # Convert to numpy array if not already
    chunk_data = np.asarray(chunk_data)
    
    # Handle dimensions
    if chunk_data.ndim > 2:
        # For multi-channel data, keep the expected dimensions for Cellpose
        # Cellpose expects (Z, Y, X) or (Y, X, C) or (Z, Y, X, C)
        if chunk_data.ndim == 3 and chunk_data.shape[0] <= 3:  # Likely (C, Y, X)
            chunk_data = np.moveaxis(chunk_data, 0, -1)  # Convert to (Y, X, C)
    
    # Extract parameters
    diameter = model_params.get('diameter', 30.0)
    flow_threshold = model_params.get('flow_threshold', 0.4)
    cellprob_threshold = model_params.get('cellprob_threshold', -1.0)
    channels = model_params.get('channels', [0, 0])
    
    # Run segmentation
    masks, flows, styles = model.eval(
        chunk_data,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=channels
    )
    
    # Get cell outlines and add position offset
    outlines = get_cell_outlines(masks)
    
    # Add position offset to outlines
    for item in outlines:
        item['y_coords'] = item['y_coords'] + y_start
        item['x_coords'] = item['x_coords'] + x_start
        item['chunk_position'] = f"{y_start}_{x_start}"
        item['model'] = model_name
    
    # Create a DataFrame from the outlines
    outlines_df = pd.DataFrame(outlines)
    
    # Write to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"outlines_{y_start}_{x_start}_{model_name}.csv")
    outlines_df.to_csv(csv_path, index=False)
    
    # Save mask image
    mask_path = os.path.join(output_dir, f"mask_{y_start}_{x_start}_{model_name}.tif")
    tifffile.imwrite(mask_path, masks.astype(np.uint16))
    
    # Update tile info for live viewer
    if tile_info_path:
        # Save current tile information for the live viewer
        # Format: y,x,tile_h,tile_w,scan_h,scan_w,seg_filename
        h, w = masks.shape
        seg_filename = f"mask_{y_start}_{x_start}_{model_name}.tif"
        with open(tile_info_path, 'w') as f:
            f.write(f"{y_start},{x_start},{h},{w},{chunk_data.shape[-2]},{chunk_data.shape[-1]},{seg_filename}")
    
    # Generate visualization for live update
    if live_update_image_path:
        try:
            # Create a visualization of the segmentation
            plt.figure(figsize=(10, 10))
            plt.imshow(chunk_data if chunk_data.ndim == 2 else chunk_data[..., 0], cmap='gray')
            plt.contour(masks, colors='r', linewidths=0.8)
            plt.title(f"Tile at ({y_start}, {x_start}) - {len(outlines)} cells")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(live_update_image_path, dpi=150)
            plt.close()
        except Exception as e:
            pass  # Don't fail if visualization fails
    
    end_time = time.time()
    
    return {
        'position': (y_start, x_start),
        'model': model_name,
        'outlines': outlines,
        'mask_path': mask_path,
        'csv_path': csv_path,
        'cell_count': len(outlines),
        'processing_time': end_time - start_time
    }

def run_distributed_segmentation(image_path, output_dir, config_path=None, 
                                n_workers=None, use_gpu=True, chunk_size=(2048, 2048),
                                tile_info_path=None, live_update_image_path=None):
    """
    Run distributed segmentation on a large image.
    
    Args:
        image_path (str): Path to the image file.
        output_dir (str): Directory to save segmentation results.
        config_path (str, optional): Path to configuration file.
        n_workers (int, optional): Number of workers. If None, use all available.
        use_gpu (bool): Whether to use GPU for computation.
        chunk_size (tuple): Size of chunks to split the image into.
        tile_info_path (str, optional): Path to save tile information for live viewer.
        live_update_image_path (str, optional): Path to save live update images.
    
    Returns:
        dict: Dictionary of segmentation results.
    """
    # Check for required dependencies
    if not DISTRIBUTED_AVAILABLE:
        logger.error("The 'distributed' package is required for distributed segmentation but is not installed.")
        return {
            "error": "Missing required dependency: distributed package is not installed. Try installing it with 'pip install dask[distributed]'."
        }
    
    if use_gpu and not DASK_CUDA_AVAILABLE:
        logger.warning("The 'dask_cuda' package is not available. Falling back to CPU-only mode.")
        use_gpu = False
    
    if not CELLPOSE_AVAILABLE:
        logger.error("The 'cellpose' package is required for segmentation but is not installed.")
        return {
            "error": "Missing required dependency: cellpose package is not installed."
        }
    # Set up logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting distributed segmentation with SMINT v{__import__('smint').__version__}")
    
    # Load configuration
    config = {}
    if config_path:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    
    # Extract model parameters from config
    model_paths = config.get('model_paths', ['cyto'])
    model_params = config.get('model_params', {
        'diameter': 30.0,
        'flow_threshold': 0.4,
        'cellprob_threshold': -1.0,
        'channels': [0, 0]
    })
    
    # Override chunk size from config if provided
    if 'chunk_size' in config:
        chunk_size = tuple(config.get('chunk_size'))
    
    # Set up Dask client
    client = setup_dask_client(n_workers, use_gpu)
    logger.info(f"Dask client set up with dashboard at {client.dashboard_link}")
    
    try:
        # Load image as dask array
        image_array = load_image_as_dask_array(image_path, chunk_size)
        
        # Preprocess image if needed
        preprocessing_func = None  # No preprocessing by default
        if 'preprocessing' in config:
            # Implement preprocessing based on config
            pass
        
        image_array = preprocess_image(image_array, preprocessing_func)
        
        # Load segmentation models
        models_dict = load_segmentation_models(model_paths)
        
        # Run segmentation
        results = segment_image_distributed(
            image_array, 
            models_dict, 
            output_dir,
            model_params=model_params,
            tile_info_path=tile_info_path,
            live_update_image_path=live_update_image_path
        )
        
        logger.info(f"Segmentation completed. Results saved to {output_dir}")
        return results
    
    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        raise
    
    finally:
        # Close the client
        client.close()
        logger.info("Dask client closed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distributed segmentation on a large image")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--config_path", help="Path to configuration file")
    parser.add_argument("--n_workers", type=int, help="Number of workers")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computation")
    parser.add_argument("--chunk_size", type=int, nargs=2, default=(2048, 2048), help="Chunk size (height, width)")
    parser.add_argument("--tile_info_path", help="Path to save tile information for live viewer")
    parser.add_argument("--live_update_image_path", help="Path to save live update images")
    
    args = parser.parse_args()
    
    run_distributed_segmentation(
        args.image_path,
        args.output_dir,
        args.config_path,
        args.n_workers,
        args.use_gpu,
        tuple(args.chunk_size),
        args.tile_info_path,
        args.live_update_image_path
    )
