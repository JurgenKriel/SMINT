#!/usr/bin/env python
"""
Process omics data using SMINT.

This script provides a command-line interface for processing omics data,
including preprocessing, segmentation, and integration.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from smint.preprocessing import preprocess_ome_tiff, combine_channels
from smint.segmentation import process_large_image, run_distributed_segmentation
from smint.alignment import align_spatial_transcriptomics
from smint.r_integration import run_r_script

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
    parser = argparse.ArgumentParser(description='Process omics data using SMINT')
    
    parser.add_argument('--ome-tiff', 
                        help='Path to the OME-TIFF file to preprocess')
    
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save results')
    
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing step')
    
    parser.add_argument('--segment', action='store_true',
                        help='Run segmentation step')
    
    parser.add_argument('--align', action='store_true',
                        help='Run alignment step')
    
    parser.add_argument('--integrate', action='store_true',
                        help='Run R integration step')
    
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian blur sigma for preprocessing')
    
    parser.add_argument('--distributed', action='store_true',
                        help='Run segmentation in distributed mode using Dask')
    
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for computation')
    
    parser.add_argument('--cell-model', default='cyto',
                        help='Path to cell segmentation model')
    
    parser.add_argument('--nuclei-model', default='nuclei',
                        help='Path to nuclei segmentation model')
    
    parser.add_argument('--reference-data', 
                        help='Path to reference data for alignment')
    
    parser.add_argument('--target-data', 
                        help='Path to target data for alignment')
    
    parser.add_argument('--r-script', 
                        help='Path to R script for integration')
    
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
    log_file = os.path.join(args.output_dir, 'process_omics.log')
    logger = setup_logger(log_file)
    logger.setLevel(log_level)
    
    logger.info("Starting SMINT omics processing")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Step 1: Preprocessing
    if args.preprocess:
        if not args.ome_tiff:
            logger.error("--ome-tiff is required for preprocessing")
            sys.exit(1)
            
        logger.info(f"Step 1: Preprocessing {args.ome_tiff}")
        
        try:
            # Create preprocessing output directory
            preprocess_dir = os.path.join(args.output_dir, 'preprocessing')
            os.makedirs(preprocess_dir, exist_ok=True)
            
            # Run preprocessing
            start_time = time.time()
            
            # Preprocess OME-TIFF
            preprocessed = preprocess_ome_tiff(
                image_path=args.ome_tiff,
                sigma=args.sigma,
                normalize=True
            )
            
            if not preprocessed:
                logger.error("Preprocessing failed")
                sys.exit(1)
                
            logger.info(f"Preprocessed {len(preprocessed)} channels")
            
            # Save individual channels
            for channel_name, channel_data in preprocessed.items():
                output_path = os.path.join(preprocess_dir, f"{channel_name}.tif")
                import tifffile
                tifffile.imwrite(output_path, channel_data)
                logger.info(f"Saved {channel_name} to {output_path}")
            
            # Combine specific channels if there are multiple
            if len(preprocessed) >= 2:
                channel_names = list(preprocessed.keys())
                channel1 = preprocessed[channel_names[0]]
                channel2 = preprocessed[channel_names[1]]
                
                combined = combine_channels(channel1, channel2, normalize_result=True)
                combined_path = os.path.join(preprocess_dir, f"{channel_names[0]}_{channel_names[1]}_combined.tif")
                tifffile.imwrite(combined_path, combined)
                logger.info(f"Saved combined {channel_names[0]}+{channel_names[1]} to {combined_path}")
                
                # Create RGB composite if there are at least 3 channels
                if len(preprocessed) >= 3:
                    import numpy as np
                    rgb = np.zeros((channel1.shape[0], channel1.shape[1], 3), dtype=np.uint8)
                    
                    for i, channel_name in enumerate(list(preprocessed.keys())[:3]):
                        rgb[:, :, i] = preprocessed[channel_name]
                    
                    rgb_path = os.path.join(preprocess_dir, "rgb_composite.tif")
                    tifffile.imwrite(rgb_path, rgb)
                    logger.info(f"Saved RGB composite to {rgb_path}")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Preprocessing completed in {elapsed_time:.2f} seconds")
            
            # Select the preprocessed image for segmentation
            if len(preprocessed) >= 2:
                segmentation_input = combined_path
            else:
                segmentation_input = os.path.join(preprocess_dir, f"{list(preprocessed.keys())[0]}.tif")
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}", exc_info=True)
            sys.exit(1)
    else:
        # If not preprocessing, check if we're running segmentation and need input image
        if args.segment and not args.ome_tiff:
            logger.error("Either --preprocess or --ome-tiff is required for segmentation")
            sys.exit(1)
            
        segmentation_input = args.ome_tiff
    
    # Step 2: Segmentation
    if args.segment:
        logger.info(f"Step 2: Segmentation of {segmentation_input}")
        
        try:
            # Create segmentation output directory
            segment_dir = os.path.join(args.output_dir, 'segmentation')
            os.makedirs(segment_dir, exist_ok=True)
            
            # Run segmentation
            start_time = time.time()
            
            # Determine input image
            if 'segmentation_input' not in locals():
                segmentation_input = args.ome_tiff
            
            # Live viewer settings
            tile_info_path = os.path.join(segment_dir, 'current_tile_info.txt')
            live_update_image_path = os.path.join(segment_dir, 'live_view.png')
            
            # Run segmentation based on mode
            if args.distributed:
                # Run distributed segmentation
                results = run_distributed_segmentation(
                    image_path=segmentation_input,
                    output_dir=segment_dir,
                    n_workers=None,  # Use all available
                    use_gpu=args.use_gpu,
                    chunk_size=(2048, 2048),
                    tile_info_path=tile_info_path,
                    live_update_image_path=live_update_image_path
                )
            else:
                # Run single-process segmentation
                csv_base_path = os.path.join(segment_dir, 'cell_outlines')
                
                results = process_large_image(
                    image_path=segmentation_input,
                    csv_base_path=csv_base_path,
                    chunk_size=(2048, 2048),
                    # Cell Model parameters
                    cell_model_path=args.cell_model,
                    cells_diameter=120.0,
                    cells_flow_threshold=0.4,
                    cells_cellprob_threshold=-1.5,
                    cells_channels=[1, 2],
                    # Nuclei Model parameters
                    nuclei_model_path=args.nuclei_model,
                    nuclei_diameter=40.0,
                    nuclei_flow_threshold=0.4,
                    nuclei_cellprob_threshold=-1.2,
                    nuclei_channels=[2, 0],
                    # Visualization parameters
                    visualize=True,
                    visualize_output_dir=os.path.join(segment_dir, 'visualizations'),
                    # Live update parameters
                    live_update_image_path=live_update_image_path,
                    tile_info_file_for_viewer=tile_info_path
                )
            
            if results is None or ('error' in results and results['error']):
                logger.error("Segmentation failed")
                sys.exit(1)
                
            logger.info(f"Segmentation completed successfully")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Segmentation completed in {elapsed_time:.2f} seconds")
            
            # Store result paths for alignment
            if not args.distributed:
                cells_path = results.get('cells_csv_path')
                nuclei_path = results.get('nuclei_csv_path')
            else:
                cells_path = os.path.join(segment_dir, "combined_cell_outlines.csv")
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}", exc_info=True)
            sys.exit(1)
    
    # Step 3: Alignment
    if args.align:
        if not args.reference_data or not args.target_data:
            logger.error("--reference-data and --target-data are required for alignment")
            sys.exit(1)
            
        logger.info(f"Step 3: Alignment of {args.target_data} to {args.reference_data}")
        
        try:
            # Create alignment output directory
            align_dir = os.path.join(args.output_dir, 'alignment')
            os.makedirs(align_dir, exist_ok=True)
            
            # Run alignment
            start_time = time.time()
            
            alignment_results = align_spatial_transcriptomics(
                reference_file=args.reference_data,
                target_file=args.target_data,
                output_dir=align_dir,
                method='affine'
            )
            
            if alignment_results is None:
                logger.error("Alignment failed")
                sys.exit(1)
                
            logger.info(f"Alignment completed successfully")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Alignment completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during alignment: {e}", exc_info=True)
            sys.exit(1)
    
    # Step 4: R Integration
    if args.integrate:
        if not args.r_script:
            logger.error("--r-script is required for integration")
            sys.exit(1)
            
        logger.info(f"Step 4: Integration using R script {args.r_script}")
        
        try:
            # Create integration output directory
            integrate_dir = os.path.join(args.output_dir, 'integration')
            os.makedirs(integrate_dir, exist_ok=True)
            
            # Run R script
            start_time = time.time()
            
            # Build arguments for R script
            r_args = [
                "--output-dir", integrate_dir
            ]
            
            # Add segmentation results if available
            if args.segment and 'cells_path' in locals():
                r_args.extend(["--cells", cells_path])
                
                if 'nuclei_path' in locals() and nuclei_path:
                    r_args.extend(["--nuclei", nuclei_path])
            
            # Add alignment results if available
            if args.align:
                r_args.extend(["--alignment", os.path.join(align_dir, "aligned_coordinates.csv")])
            
            # Run the R script
            return_code = run_r_script(args.r_script, r_args)
            
            if return_code != 0:
                logger.error(f"Integration failed with return code {return_code}")
                sys.exit(1)
                
            logger.info(f"Integration completed successfully")
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Integration completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during integration: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info("All processing steps completed successfully")

if __name__ == '__main__':
    main()
