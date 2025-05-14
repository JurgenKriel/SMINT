#!/usr/bin/env python
"""
Basic SMINT workflow example.

This script demonstrates a complete workflow from preprocessing through 
segmentation to visualization.
"""

import os
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
import numpy as np
from pathlib import Path

from smint.preprocessing import preprocess_ome_tiff, combine_channels
from smint.segmentation import process_large_image
from smint.visualization import visualize_cell_outlines, visualize_segmentation_overlay
from smint.utils.config import load_config

def setup_logger():
    """Set up the logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('smint_basic_workflow')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run a basic SMINT workflow')
    
    parser.add_argument('--image', required=True, 
                        help='Path to the input image file (OME-TIFF)')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save results')
    
    parser.add_argument('--config', 
                        help='Path to configuration file')
    
    parser.add_argument('--sigma', type=float, default=1.5,
                        help='Sigma value for Gaussian blur during preprocessing')
    
    parser.add_argument('--cell-diameter', type=float, default=120.0,
                        help='Diameter of cells for segmentation')
    
    parser.add_argument('--nuclei-diameter', type=float, default=40.0,
                        help='Diameter of nuclei for segmentation')
    
    parser.add_argument('--chunk-size', type=int, nargs=2, default=[2048, 2048],
                        help='Size of chunks for processing (height width)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    
    return parser.parse_args()

def main():
    """Main function to run the basic workflow."""
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = setup_logger()
    
    # Set up directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration if provided
    config = load_config(args.config) if args.config else {}
    
    logger.info(f"Starting basic SMINT workflow on {args.image}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Step 1: Preprocess the image
    logger.info("Step 1: Preprocessing image")
    preprocess_dir = os.path.join(output_dir, "preprocessing")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    preprocessed = preprocess_ome_tiff(
        image_path=args.image,
        sigma=args.sigma,
        normalize=True
    )
    
    logger.info(f"Preprocessed {len(preprocessed)} channels")
    
    # Save individual channels
    for channel_name, channel_data in preprocessed.items():
        channel_path = os.path.join(preprocess_dir, f"{channel_name}.tif")
        tifffile.imwrite(channel_path, channel_data)
        logger.info(f"Saved {channel_name} to {channel_path}")
    
    # Combine specific channels if there are multiple
    segmentation_input = None
    if len(preprocessed) >= 2:
        channel_names = list(preprocessed.keys())
        channel1 = preprocessed[channel_names[0]]
        channel2 = preprocessed[channel_names[1]]
        
        combined = combine_channels(channel1, channel2, normalize_result=True)
        combined_path = os.path.join(preprocess_dir, f"{channel_names[0]}_{channel_names[1]}_combined.tif")
        tifffile.imwrite(combined_path, combined)
        logger.info(f"Saved combined {channel_names[0]}+{channel_names[1]} to {combined_path}")
        
        # Use combined image for segmentation
        segmentation_input = combined_path
        
        # Create RGB composite if there are at least 3 channels
        if len(preprocessed) >= 3:
            rgb = np.zeros((channel1.shape[0], channel1.shape[1], 3), dtype=np.uint8)
            
            for i, channel_name in enumerate(list(preprocessed.keys())[:3]):
                rgb[:, :, i] = preprocessed[channel_name]
            
            rgb_path = os.path.join(preprocess_dir, "rgb_composite.tif")
            tifffile.imwrite(rgb_path, rgb)
            logger.info(f"Saved RGB composite to {rgb_path}")
    else:
        # Use first channel for segmentation
        channel_name = list(preprocessed.keys())[0]
        segmentation_input = os.path.join(preprocess_dir, f"{channel_name}.tif")
    
    # Step 2: Segment cells and nuclei
    logger.info(f"Step 2: Segmenting {segmentation_input}")
    segment_dir = os.path.join(output_dir, "segmentation")
    os.makedirs(segment_dir, exist_ok=True)
    
    # Get segmentation parameters from config or use defaults/args
    cell_params = config.get('cell_model_params', {})
    cell_diameter = args.cell_diameter or cell_params.get('diameter', 120.0)
    cell_flow_threshold = cell_params.get('flow_threshold', 0.4)
    cell_cellprob_threshold = cell_params.get('cellprob_threshold', -1.5)
    cell_channels = cell_params.get('channels', [0, 0])  # Default to first channel
    
    nuclei_params = config.get('nuclei_model_params', {})
    nuclei_diameter = args.nuclei_diameter or nuclei_params.get('diameter', 40.0)
    nuclei_flow_threshold = nuclei_params.get('flow_threshold', 0.4)
    nuclei_cellprob_threshold = nuclei_params.get('cellprob_threshold', -1.2)
    nuclei_channels = nuclei_params.get('channels', [0, 0])  # Default to first channel
    
    # Run segmentation
    results = process_large_image(
        image_path=segmentation_input,
        csv_base_path=os.path.join(segment_dir, "cell_outlines"),
        chunk_size=tuple(args.chunk_size),
        # Cell Model parameters
        cell_model_path=config.get('model_paths', ['cyto'])[0] if 'model_paths' in config else "cyto",
        cells_diameter=cell_diameter,
        cells_flow_threshold=cell_flow_threshold,
        cells_cellprob_threshold=cell_cellprob_threshold,
        cells_channels=cell_channels,
        # Nuclei Model parameters
        nuclei_model_path=config.get('model_paths', ['cyto', 'nuclei'])[1] if len(config.get('model_paths', [])) > 1 else "nuclei",
        nuclei_diameter=nuclei_diameter,
        nuclei_flow_threshold=nuclei_flow_threshold,
        nuclei_cellprob_threshold=nuclei_cellprob_threshold,
        nuclei_channels=nuclei_channels,
        # Visualization parameters
        visualize=args.visualize,
        visualize_output_dir=os.path.join(segment_dir, "visualizations"),
        num_visualize_chunks=5,
        visualize_roi_size=(2024, 2024),
        vis_bg_channel_indices=[0, 1] if len(preprocessed) >= 2 else [0]
    )
    
    if 'error' in results:
        logger.error(f"Segmentation failed: {results['error']}")
        return
    
    logger.info(f"Found {results['total_cells']} cells and {results['total_nuclei']} nuclei")
    
    # Step 3: Create summary visualizations
    logger.info("Step 3: Creating summary visualizations")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create cell outlines visualization
    cells_csv_path = results.get('cells_csv_path')
    if cells_csv_path and os.path.exists(cells_csv_path):
        cells_df = pd.read_csv(cells_csv_path)
        logger.info(f"Loaded {cells_df['global_cell_id'].nunique()} cells from {cells_csv_path}")
        
        # Create cell outlines visualization
        fig = visualize_cell_outlines(
            cells_df, 
            color_by='global_cell_id',
            figsize=(12, 12),
            title=f"Cell Outlines (n={cells_df['global_cell_id'].nunique()})"
        )
        
        if fig:
            cell_outlines_path = os.path.join(vis_dir, "cell_outlines.png")
            fig.savefig(cell_outlines_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved cell outlines visualization to {cell_outlines_path}")
    
    # Create nuclei outlines visualization
    nuclei_csv_path = results.get('nuclei_csv_path')
    if nuclei_csv_path and os.path.exists(nuclei_csv_path):
        nuclei_df = pd.read_csv(nuclei_csv_path)
        logger.info(f"Loaded {nuclei_df['global_cell_id'].nunique()} nuclei from {nuclei_csv_path}")
        
        # Create nuclei outlines visualization
        fig = visualize_cell_outlines(
            nuclei_df, 
            color_by='global_cell_id',
            figsize=(12, 12),
            title=f"Nuclei Outlines (n={nuclei_df['global_cell_id'].nunique()})"
        )
        
        if fig:
            nuclei_outlines_path = os.path.join(vis_dir, "nuclei_outlines.png")
            fig.savefig(nuclei_outlines_path, dpi=300)
            plt.close(fig)
            logger.info(f"Saved nuclei outlines visualization to {nuclei_outlines_path}")
    
    # Create overlay visualization if both cells and nuclei were segmented
    if cells_csv_path and nuclei_csv_path and os.path.exists(cells_csv_path) and os.path.exists(nuclei_csv_path):
        # Load background image (RGB composite or first channel)
        if os.path.exists(os.path.join(preprocess_dir, "rgb_composite.tif")):
            bg_image = tifffile.imread(os.path.join(preprocess_dir, "rgb_composite.tif"))
        else:
            channel_name = list(preprocessed.keys())[0]
            bg_image = tifffile.imread(os.path.join(preprocess_dir, f"{channel_name}.tif"))
            # Convert to RGB if grayscale
            if bg_image.ndim == 2:
                bg_image = np.stack([bg_image, bg_image, bg_image], axis=2)
        
        # Create a subset of cells and nuclei for a region of interest
        roi_size = 1024
        if bg_image.shape[0] > roi_size and bg_image.shape[1] > roi_size:
            # Calculate center ROI
            y_center = bg_image.shape[0] // 2
            x_center = bg_image.shape[1] // 2
            y_start = y_center - roi_size // 2
            x_start = x_center - roi_size // 2
            
            roi_image = bg_image[y_start:y_start+roi_size, x_start:x_start+roi_size]
            
            # Filter outlines for ROI
            roi_cells = cells_df[
                (cells_df['y'] >= y_start) & (cells_df['y'] < y_start+roi_size) &
                (cells_df['x'] >= x_start) & (cells_df['x'] < x_start+roi_size)
            ].copy()
            
            roi_nuclei = nuclei_df[
                (nuclei_df['y'] >= y_start) & (nuclei_df['y'] < y_start+roi_size) &
                (nuclei_df['x'] >= x_start) & (nuclei_df['x'] < x_start+roi_size)
            ].copy()
            
            # Adjust coordinates to ROI
            roi_cells['y'] = roi_cells['y'] - y_start
            roi_cells['x'] = roi_cells['x'] - x_start
            
            roi_nuclei['y'] = roi_nuclei['y'] - y_start
            roi_nuclei['x'] = roi_nuclei['x'] - x_start
            
            # Create overlay visualization
            fig = visualize_segmentation_overlay(
                roi_image,
                None,  # No masks, just outlines
                outlines=pd.concat([
                    roi_cells.assign(outline_type='cell'),
                    roi_nuclei.assign(outline_type='nucleus')
                ]),
                figsize=(12, 12),
                alpha=0.5,
                outline_color='red'
            )
            
            if fig:
                overlay_path = os.path.join(vis_dir, "segmentation_overlay_roi.png")
                fig.savefig(overlay_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved segmentation overlay to {overlay_path}")
    
    logger.info(f"Basic workflow completed successfully. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
