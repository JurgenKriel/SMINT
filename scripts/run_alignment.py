#!/usr/bin/env python
"""
Run alignment on spatial transcriptomics data using SMINT.

This script provides a command-line interface for running the alignment
pipeline on spatial transcriptomics data using the SMINT package.
"""

import argparse
import logging
import os
import sys
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
    parser = argparse.ArgumentParser(description='Run alignment on spatial transcriptomics data using SMINT')
    
    parser.add_argument('--reference', required=True, 
                        help='Path to the reference data file (CSV or TSV)')
    
    parser.add_argument('--target', required=True, 
                        help='Path to the target data file to align (CSV or TSV)')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save alignment results')
    
    parser.add_argument('--method', default='affine', choices=['affine', 'rigid', 'similarity', 'projective'],
                        help='Alignment method')
    
    parser.add_argument('--reference-type', default='visium', choices=['visium', 'slideseq', 'custom'],
                        help='Type of reference data')
    
    parser.add_argument('--target-type', default='visium', choices=['visium', 'slideseq', 'custom'],
                        help='Type of target data')
    
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor for the alignment')
    
    parser.add_argument('--reference-x-col', 
                        help='Column name for reference x-coordinates (for custom data)')
    
    parser.add_argument('--reference-y-col', 
                        help='Column name for reference y-coordinates (for custom data)')
    
    parser.add_argument('--target-x-col', 
                        help='Column name for target x-coordinates (for custom data)')
    
    parser.add_argument('--target-y-col', 
                        help='Column name for target y-coordinates (for custom data)')
    
    parser.add_argument('--st-align-path', 
                        help='Path to the ST Align executable')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = os.path.join(args.output_dir, 'alignment.log')
    logger = setup_logger(log_file)
    logger.setLevel(log_level)
    
    logger.info("Starting SMINT alignment")
    logger.info(f"Reference data: {args.reference}")
    logger.info(f"Target data: {args.target}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Alignment method: {args.method}")
    
    # Prepare parameters for alignment
    reference_coords_cols = None
    if args.reference_type == 'custom':
        if not args.reference_x_col or not args.reference_y_col:
            logger.error("For custom reference data, --reference-x-col and --reference-y-col are required")
            sys.exit(1)
        reference_coords_cols = [args.reference_x_col, args.reference_y_col]
    
    target_coords_cols = None
    if args.target_type == 'custom':
        if not args.target_x_col or not args.target_y_col:
            logger.error("For custom target data, --target-x-col and --target-y-col are required")
            sys.exit(1)
        target_coords_cols = [args.target_x_col, args.target_y_col]
    
    # Run alignment
    start_time = time.time()
    
    try:
        alignment_results = align_spatial_transcriptomics(
            reference_file=args.reference,
            target_file=args.target,
            output_dir=args.output_dir,
            method=args.method,
            st_align_path=args.st_align_path,
            reference_type=args.reference_type,
            target_type=args.target_type,
            scale_factor=args.scale_factor,
            reference_coords_cols=reference_coords_cols,
            target_coords_cols=target_coords_cols
        )
        
        if alignment_results is None:
            logger.error("Alignment failed")
            sys.exit(1)
        
        # Save results
        result_dir = os.path.join(args.output_dir, 'final_results')
        os.makedirs(result_dir, exist_ok=True)
        
        save_alignment(alignment_results, result_dir)
        
        # Log some statistics
        aligned_coords = alignment_results['aligned_coordinates']
        logger.info(f"Alignment completed successfully")
        logger.info(f"Aligned {len(aligned_coords)} points")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Alignment completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during alignment: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
