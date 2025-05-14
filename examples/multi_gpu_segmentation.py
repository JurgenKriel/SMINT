#!/usr/bin/env python
"""
Multi-GPU segmentation example for SMINT.

This script demonstrates how to use multiple GPUs for distributed segmentation
of large whole-slide images using Dask.
"""

import os
import argparse
import logging
import time
from pathlib import Path

from smint.segmentation import run_distributed_segmentation
from smint.utils.config import load_config, save_config, merge_configs
from smint.visualization import run_viewer
import threading

def setup_logger(log_file=None):
    """Set up the logger."""
    logger = logging.getLogger('smint')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-GPU segmentation with SMINT')
    
    parser.add_argument('--image', required=True, 
                        help='Path to the input image file')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save segmentation results')
    
    parser.add_argument('--config', 
                        help='Path to configuration file')
    
    parser.add_argument('--n-workers', type=int,
                        help='Number of Dask workers')
    
    parser.add_argument('--chunk-size', type=int, nargs=2, default=[2048, 2048],
                        help='Size of chunks for processing (height width)')
    
    parser.add_argument('--memory-limit', default='16GB',
                        help='Memory limit per worker')
    
    parser.add_argument('--live-viewer', action='store_true',
                        help='Enable live viewer')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    parser.add_argument('--model', default='cyto',
                        help='Cellpose model to use')
    
    return parser.parse_args()

def start_live_viewer(image_path, output_dir):
    """Start the live viewer in a separate thread."""
    # Configure paths for live viewer
    segmentation_history_dir = output_dir
    tile_info_path = os.path.join(output_dir, "current_tile_info.txt")
    
    # Start the viewer
    run_viewer(image_path, segmentation_history_dir, tile_info_path)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, 'multi_gpu_segmentation.log')
    logger = setup_logger(log_file)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting SMINT distributed segmentation")
    logger.info(f"Input image: {args.image}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Configure paths for live updates
    tile_info_path = os.path.join(args.output_dir, "current_tile_info.txt")
    live_update_image_path = os.path.join(args.output_dir, "live_view.png")
    
    # Start live viewer in a separate thread if requested
    if args.live_viewer:
        logger.info("Starting live viewer")
        viewer_thread = threading.Thread(
            target=start_live_viewer,
            args=(args.image, args.output_dir),
            daemon=True
        )
        viewer_thread.start()
        
        # Give the viewer a moment to start
        time.sleep(2)
    
    # Load configuration
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Override with command-line arguments
    config_overrides = {
        'use_gpu': True,
        'n_workers': args.n_workers,
        'chunk_size': args.chunk_size,
        'memory_limit': args.memory_limit,
        'model_paths': [args.model],
        'tile_info_path': tile_info_path,
        'live_update_image_path': live_update_image_path
    }
    
    # Remove None values
    config_overrides = {k: v for k, v in config_overrides.items() if v is not None}
    
    # Merge configurations
    config = merge_configs(config, config_overrides)
    
    # Save effective configuration
    config_file = os.path.join(args.output_dir, 'segmentation_config.yaml')
    save_config(config, config_file)
    logger.info(f"Saved effective configuration to {config_file}")
    
    # Extract parameters for distributed segmentation
    n_workers = config.get('n_workers')
    use_gpu = config.get('use_gpu', True)
    chunk_size = tuple(config.get('chunk_size', (2048, 2048)))
    memory_limit = config.get('memory_limit', '16GB')
    
    # Log chunk size and memory limit
    logger.info(f"Using chunk size: {chunk_size}")
    logger.info(f"Using memory limit per worker: {memory_limit}")
    
    # Run distributed segmentation
    start_time = time.time()
    
    try:
        results = run_distributed_segmentation(
            image_path=args.image,
            output_dir=args.output_dir,
            config_path=args.config,
            n_workers=n_workers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            tile_info_path=tile_info_path,
            live_update_image_path=live_update_image_path
        )
        
        if results is None:
            logger.error("Segmentation failed")
            return 1
            
        # Log results
        logger.info(f"Segmentation completed successfully")
        logger.info(f"Results saved to {args.output_dir}")
        
        # Log cell counts if available
        if isinstance(results, dict) and 'all_chunk_results' in results:
            total_cells = sum(r.get('cell_count', 0) for r in results['all_chunk_results'] if isinstance(r, dict))
            logger.info(f"Total cells detected: {total_cells}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}", exc_info=True)
        return 1
    
    # Keep the main thread alive if the viewer is running
    if args.live_viewer and viewer_thread.is_alive():
        logger.info("Segmentation completed. Live viewer is still running.")
        logger.info("Press Ctrl+C to exit.")
        try:
            while viewer_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting.")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
