#!/usr/bin/env python
"""
Run segmentation on an image using SMINT.

This script provides a command-line interface for running the segmentation
pipeline on an image using the SMINT package.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from smint.segmentation import process_large_image, run_distributed_segmentation
from smint.utils.config import load_config, save_config, merge_configs, validate_config

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
    parser = argparse.ArgumentParser(description='Run segmentation on an image using SMINT')
    
    parser.add_argument('--image', required=True, 
                        help='Path to the input image file')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save segmentation results')
    
    parser.add_argument('--config', 
                        help='Path to configuration file')
    
    parser.add_argument('--distributed', action='store_true',
                        help='Run segmentation in distributed mode using Dask')
    
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for computation')
    
    parser.add_argument('--cell-model',
                        help='Path to cell segmentation model')
    
    parser.add_argument('--nuclei-model',
                        help='Path to nuclei segmentation model')
    
    parser.add_argument('--cell-diameter', type=float,
                        help='Diameter of cells for segmentation')
    
    parser.add_argument('--nuclei-diameter', type=float,
                        help='Diameter of nuclei for segmentation')
    
    parser.add_argument('--chunk-size', type=int, nargs=2, default=[2048, 2048],
                        help='Size of chunks for processing (height width)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    parser.add_argument('--live-viewer', action='store_true',
                        help='Enable live viewer')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    parser.add_argument('--save-config', action='store_true',
                        help='Save effective configuration to a file')
    
    return parser.parse_args()

def build_config_from_args(args):
    """Build a configuration dictionary from command-line arguments."""
    config = {}
    
    # Basic settings
    config['output_dir'] = args.output_dir
    
    if args.distributed:
        config['use_gpu'] = args.use_gpu
        config['chunk_size'] = args.chunk_size
    
    # Model paths
    model_paths = []
    if args.cell_model:
        model_paths.append(args.cell_model)
    if args.nuclei_model:
        model_paths.append(args.nuclei_model)
    
    if model_paths:
        config['model_paths'] = model_paths
    
    # Cell parameters
    cell_params = {}
    if args.cell_diameter:
        cell_params['diameter'] = args.cell_diameter
    
    if cell_params:
        config['cell_model_params'] = cell_params
    
    # Nuclei parameters
    nuclei_params = {}
    if args.nuclei_diameter:
        nuclei_params['diameter'] = args.nuclei_diameter
    
    if nuclei_params:
        config['nuclei_model_params'] = nuclei_params
    
    # Visualization
    if args.visualize:
        config['visualization'] = {'enable': True}
    
    # Live viewer
    if args.live_viewer:
        config['tile_info_path'] = os.path.join(args.output_dir, 'current_tile_info.txt')
        config['live_update_image_path'] = os.path.join(args.output_dir, 'live_view.png')
    
    return config

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = os.path.join(args.output_dir, 'segmentation.log')
    logger = setup_logger(log_file)
    logger.setLevel(log_level)
    
    logger.info("Starting SMINT segmentation")
    logger.info(f"Input image: {args.image}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        config = load_config()  # Try to load from default locations
    
    # Build configuration from command-line arguments
    args_config = build_config_from_args(args)
    
    # Merge configurations
    config = merge_configs(config, args_config)
    
    # Validate configuration
    valid, errors = validate_config(config)
    if not valid:
        logger.error("Invalid configuration:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Save effective configuration if requested
    if args.save_config:
        config_file = os.path.join(args.output_dir, 'segmentation_config.yaml')
        save_config(config, config_file)
        logger.info(f"Saved effective configuration to {config_file}")
    
    # Prepare parameters for segmentation
    if args.distributed:
        # Run distributed segmentation
        logger.info("Running distributed segmentation")
        
        # Extract parameters from config
        n_workers = config.get('n_workers')
        use_gpu = config.get('use_gpu', True)
        chunk_size = tuple(config.get('chunk_size', (2048, 2048)))
        tile_info_path = config.get('tile_info_path')
        live_update_image_path = config.get('live_update_image_path')
        
        # Run segmentation
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
                sys.exit(1)
                
            logger.info(f"Segmentation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            sys.exit(1)
            
    else:
        # Run single-process segmentation
        logger.info("Running single-process segmentation")
        
        # Extract parameters from config
        chunk_size = tuple(config.get('chunk_size', (2048, 2048)))
        
        # Cell model parameters
        cell_model_path = args.cell_model or config.get('model_paths', ['cyto'])[0]
        cell_params = config.get('cell_model_params', {})
        cells_diameter = args.cell_diameter or cell_params.get('diameter', 120.0)
        cells_flow_threshold = cell_params.get('flow_threshold', 0.4)
        cells_cellprob_threshold = cell_params.get('cellprob_threshold', -1.5)
        cells_channels = cell_params.get('channels', [1, 2])
        
        # Nuclei model parameters
        nuclei_model_path = args.nuclei_model or config.get('model_paths', ['nuclei'])[1] if len(config.get('model_paths', [])) > 1 else 'nuclei'
        nuclei_params = config.get('nuclei_model_params', {})
        nuclei_diameter = args.nuclei_diameter or nuclei_params.get('diameter', 40.0)
        nuclei_flow_threshold = nuclei_params.get('flow_threshold', 0.4)
        nuclei_cellprob_threshold = nuclei_params.get('cellprob_threshold', -1.2)
        nuclei_channels = nuclei_params.get('channels', [2, 0])
        
        # Adaptive nuclei segmentation
        adaptive_nuclei = config.get('adaptive_nuclei', {})
        enable_adaptive_nuclei = adaptive_nuclei.get('enable', False)
        nuclei_adaptive_cellprob_lower_limit = adaptive_nuclei.get('cellprob_lower_limit', -6.0)
        nuclei_adaptive_cellprob_step_decrement = adaptive_nuclei.get('step_decrement', 0.2)
        nuclei_max_adaptive_attempts = adaptive_nuclei.get('max_attempts', 3)
        adaptive_nuclei_trigger_ratio = adaptive_nuclei.get('trigger_ratio', 0.05)
        
        # Visualization parameters
        visualization = config.get('visualization', {})
        visualize = args.visualize or visualization.get('enable', False)
        visualize_output_dir = visualization.get('output_dir', os.path.join(args.output_dir, 'visualizations'))
        num_visualize_chunks = visualization.get('num_chunks_to_visualize', 5)
        visualize_roi_size = tuple(visualization.get('roi_size', (2024, 2024)))
        vis_bg_channel_indices = visualization.get('background_channel_indices', [0, 1])
        
        # Live update parameters
        tile_info_file_for_viewer = config.get('tile_info_path')
        live_update_image_path = config.get('live_update_image_path')
        
        # Run segmentation
        start_time = time.time()
        csv_base_path = os.path.join(args.output_dir, 'cell_outlines')
        
        try:
            results = process_large_image(
                image_path=args.image,
                csv_base_path=csv_base_path,
                chunk_size=chunk_size,
                # Cell Model parameters
                cell_model_path=cell_model_path,
                cells_diameter=cells_diameter,
                cells_flow_threshold=cells_flow_threshold,
                cells_cellprob_threshold=cells_cellprob_threshold,
                cells_channels=cells_channels,
                # Nuclei Model parameters
                nuclei_model_path=nuclei_model_path,
                nuclei_diameter=nuclei_diameter,
                nuclei_flow_threshold=nuclei_flow_threshold,
                nuclei_cellprob_threshold=nuclei_cellprob_threshold,
                nuclei_channels=nuclei_channels,
                # Adaptive Nuclei Segmentation parameters
                enable_adaptive_nuclei=enable_adaptive_nuclei,
                nuclei_adaptive_cellprob_lower_limit=nuclei_adaptive_cellprob_lower_limit,
                nuclei_adaptive_cellprob_step_decrement=nuclei_adaptive_cellprob_step_decrement,
                nuclei_max_adaptive_attempts=nuclei_max_adaptive_attempts,
                adaptive_nuclei_trigger_ratio=adaptive_nuclei_trigger_ratio,
                # Visualization parameters
                visualize=visualize,
                visualize_output_dir=visualize_output_dir,
                num_visualize_chunks=num_visualize_chunks,
                visualize_roi_size=visualize_roi_size,
                vis_bg_channel_indices=vis_bg_channel_indices,
                # Live update parameters
                live_update_image_path=live_update_image_path,
                tile_info_file_for_viewer=tile_info_file_for_viewer
            )
            
            if 'error' in results:
                logger.error(f"Segmentation failed: {results['error']}")
                sys.exit(1)
                
            logger.info(f"Found {results['total_cells']} cells and {results['total_nuclei']} nuclei")
            logger.info(f"Results saved to:")
            if results.get('cells_csv_path'):
                logger.info(f"  - Cells: {results['cells_csv_path']}")
            if results.get('nuclei_csv_path'):
                logger.info(f"  - Nuclei: {results['nuclei_csv_path']}")
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            sys.exit(1)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Segmentation completed in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
