#!/usr/bin/env python
"""
Spatial alignment example for SMINT.

This script demonstrates how to align spatial transcriptomics data with 
cell segmentation results using ST Align.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

from smint.alignment import align_spatial_transcriptomics, save_alignment
from smint.utils.config import load_config

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
    parser = argparse.ArgumentParser(description='Run spatial alignment with SMINT')
    
    parser.add_argument('--reference', required=True, 
                        help='Path to the reference data file (e.g., cell positions)')
    
    parser.add_argument('--target', required=True, 
                        help='Path to the target data file to align (e.g., spatial transcriptomics)')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save alignment results')
    
    parser.add_argument('--method', default='affine', choices=['affine', 'rigid', 'similarity', 'projective'],
                        help='Alignment method')
    
    parser.add_argument('--reference-type', default='custom', choices=['visium', 'slideseq', 'custom'],
                        help='Type of reference data')
    
    parser.add_argument('--target-type', default='visium', choices=['visium', 'slideseq', 'custom'],
                        help='Type of target data')
    
    parser.add_argument('--reference-x-col', default='x',
                        help='Column name for reference x-coordinates (for custom data)')
    
    parser.add_argument('--reference-y-col', default='y',
                        help='Column name for reference y-coordinates (for custom data)')
    
    parser.add_argument('--target-x-col',
                        help='Column name for target x-coordinates (for custom data)')
    
    parser.add_argument('--target-y-col',
                        help='Column name for target y-coordinates (for custom data)')
    
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor for the alignment')
    
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots of the alignment results')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def create_alignment_plots(alignment_results, output_dir):
    """Create plots of the alignment results."""
    logger = logging.getLogger('smint')
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data from alignment results
    aligned_coords = alignment_results['aligned_coordinates']
    reference_data = alignment_results['reference_data']
    target_data = alignment_results['target_data']
    
    if reference_data is None or target_data is None:
        logger.warning("Reference or target data not available for plotting")
        return
    
    # Determine column names
    ref_x_col = None
    ref_y_col = None
    for col in reference_data.columns:
        if 'x' in col.lower():
            ref_x_col = col
        elif 'y' in col.lower():
            ref_y_col = col
    
    target_x_col = None
    target_y_col = None
    for col in target_data.columns:
        if 'x' in col.lower() and not col.startswith('aligned'):
            target_x_col = col
        elif 'y' in col.lower() and not col.startswith('aligned'):
            target_y_col = col
    
    if not all([ref_x_col, ref_y_col, target_x_col, target_y_col]):
        logger.warning("Could not determine coordinate column names for plotting")
        return
    
    # Plot 1: Overlay of reference, target, and aligned points
    plt.figure(figsize=(10, 10))
    
    # Plot reference points
    plt.scatter(
        reference_data[ref_x_col], 
        reference_data[ref_y_col], 
        s=10, alpha=0.7, color='blue', 
        label='Reference'
    )
    
    # Plot original target points
    plt.scatter(
        target_data[target_x_col], 
        target_data[target_y_col], 
        s=10, alpha=0.3, color='gray', 
        label='Original Target'
    )
    
    # Plot aligned target points
    plt.scatter(
        aligned_coords['aligned_x'], 
        aligned_coords['aligned_y'], 
        s=10, alpha=0.7, color='red', 
        label='Aligned Target'
    )
    
    plt.title('Alignment Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'alignment_overlay.png'), dpi=300)
    plt.close()
    
    # Plot 2: Before and after alignment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before alignment
    ax1.scatter(
        reference_data[ref_x_col], 
        reference_data[ref_y_col], 
        s=10, alpha=0.7, color='blue', 
        label='Reference'
    )
    ax1.scatter(
        target_data[target_x_col], 
        target_data[target_y_col], 
        s=10, alpha=0.7, color='red', 
        label='Target'
    )
    ax1.set_title('Before Alignment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After alignment
    ax2.scatter(
        reference_data[ref_x_col], 
        reference_data[ref_y_col], 
        s=10, alpha=0.7, color='blue', 
        label='Reference'
    )
    ax2.scatter(
        aligned_coords['aligned_x'], 
        aligned_coords['aligned_y'], 
        s=10, alpha=0.7, color='red', 
        label='Aligned Target'
    )
    ax2.set_title('After Alignment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'before_after_alignment.png'), dpi=300)
    plt.close()
    
    # Plot 3: Transformation visualization
    # Get a subset of points to show the transformation
    n_points = min(50, len(target_data))
    indices = np.random.choice(len(target_data), n_points, replace=False)
    
    plt.figure(figsize=(10, 10))
    
    # Plot all reference points as background
    plt.scatter(
        reference_data[ref_x_col], 
        reference_data[ref_y_col], 
        s=10, alpha=0.2, color='lightgray', 
        label='Reference (all)'
    )
    
    # Plot selected original and aligned points with connecting lines
    for i in indices:
        if i < len(target_data) and i < len(aligned_coords):
            # Original point
            orig_x = target_data.iloc[i][target_x_col]
            orig_y = target_data.iloc[i][target_y_col]
            
            # Aligned point
            aligned_x = aligned_coords.iloc[i]['aligned_x']
            aligned_y = aligned_coords.iloc[i]['aligned_y']
            
            # Plot points and connecting line
            plt.scatter(orig_x, orig_y, s=30, color='blue', marker='o')
            plt.scatter(aligned_x, aligned_y, s=30, color='red', marker='x')
            plt.plot([orig_x, aligned_x], [orig_y, aligned_y], 'k-', alpha=0.5)
    
    # Add legend with custom handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='blue', linestyle='None', markersize=8, label='Original Points'),
        Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=8, label='Aligned Points'),
        Line2D([0], [0], color='black', alpha=0.5, label='Transformation')
    ]
    plt.legend(handles=handles)
    
    plt.title('Visualization of Transformation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'transformation_visualization.png'), dpi=300)
    plt.close()
    
    logger.info(f"Saved alignment plots to {plots_dir}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, 'alignment.log')
    logger = setup_logger(log_file)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting SMINT spatial alignment")
    logger.info(f"Reference data: {args.reference}")
    logger.info(f"Target data: {args.target}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Alignment method: {args.method}")
    
    # Prepare parameters for alignment
    reference_coords_cols = None
    if args.reference_type == 'custom':
        reference_coords_cols = [args.reference_x_col, args.reference_y_col]
        logger.info(f"Using reference coordinate columns: {reference_coords_cols}")
    
    target_coords_cols = None
    if args.target_type == 'custom' and args.target_x_col and args.target_y_col:
        target_coords_cols = [args.target_x_col, args.target_y_col]
        logger.info(f"Using target coordinate columns: {target_coords_cols}")
    
    # Run alignment
    start_time = time.time()
    
    try:
        # Ensure the reference file exists
        if not os.path.exists(args.reference):
            logger.error(f"Reference file not found: {args.reference}")
            return 1
        
        # Ensure the target file exists
        if not os.path.exists(args.target):
            logger.error(f"Target file not found: {args.target}")
            return 1
        
        # Run alignment
        logger.info("Running alignment...")
        alignment_results = align_spatial_transcriptomics(
            reference_file=args.reference,
            target_file=args.target,
            output_dir=args.output_dir,
            method=args.method,
            reference_type=args.reference_type,
            target_type=args.target_type,
            scale_factor=args.scale_factor,
            reference_coords_cols=reference_coords_cols,
            target_coords_cols=target_coords_cols
        )
        
        if alignment_results is None:
            logger.error("Alignment failed")
            return 1
        
        # Save alignment results
        save_dir = os.path.join(args.output_dir, 'final_results')
        save_alignment(alignment_results, save_dir)
        logger.info(f"Saved alignment results to {save_dir}")
        
        # Log alignment statistics
        aligned_coords = alignment_results['aligned_coordinates']
        logger.info(f"Aligned {len(aligned_coords)} points")
        
        if 'transformation' in alignment_results:
            logger.info(f"Transformation matrix: {alignment_results['transformation']}")
        
        # Generate plots if requested
        if args.plot:
            logger.info("Generating alignment plots...")
            create_alignment_plots(alignment_results, args.output_dir)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Alignment completed in {elapsed_time:.2f} seconds")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during alignment: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
