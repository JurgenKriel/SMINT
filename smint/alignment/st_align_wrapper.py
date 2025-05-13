"""
ST Align wrapper for SMINT.

This module provides functions for aligning spatial transcriptomics data
using the ST Align tool.
"""

import os
import subprocess
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
import tempfile
import csv

def align_spatial_transcriptomics(
    reference_file, 
    target_file, 
    output_dir=None, 
    method="affine", 
    st_align_path=None,
    reference_type="visium",
    target_type="visium",
    scale_factor=1.0,
    reference_coords_cols=None,
    target_coords_cols=None
):
    """
    Align spatial transcriptomics data using ST Align.
    
    Args:
        reference_file (str): Path to the reference data file (CSV or TSV)
        target_file (str): Path to the target data file to align (CSV or TSV)
        output_dir (str, optional): Directory to save the alignment results
        method (str): Alignment method ('affine', 'rigid', 'similarity', 'projective')
        st_align_path (str, optional): Path to the ST Align executable
        reference_type (str): Type of reference data ('visium', 'slideseq', 'custom')
        target_type (str): Type of target data ('visium', 'slideseq', 'custom')
        scale_factor (float): Scale factor for the alignment
        reference_coords_cols (list): Column names for reference coordinates if custom
        target_coords_cols (list): Column names for target coordinates if custom
    
    Returns:
        dict: Alignment results with transformation matrix and aligned data
    """
    logger = logging.getLogger(__name__)
    
    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="stalign_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    output_dir = Path(output_dir)
    
    # Check if ST Align is available
    if st_align_path is None:
        try:
            # Try to find ST Align in PATH
            subprocess.run(["stalign", "--version"], check=True, capture_output=True)
            st_align_path = "stalign"
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("ST Align not found in PATH. Please specify the path to the ST Align executable.")
            return None
    
    # Prepare command
    cmd = [
        st_align_path,
        "--reference", str(reference_file),
        "--target", str(target_file),
        "--method", method,
        "--output", str(output_dir),
        "--reference-type", reference_type,
        "--target-type", target_type,
        "--scale-factor", str(scale_factor)
    ]
    
    # Add custom coordinate columns if needed
    if reference_type == "custom" and reference_coords_cols:
        cmd.extend(["--reference-x-col", reference_coords_cols[0]])
        cmd.extend(["--reference-y-col", reference_coords_cols[1]])
    
    if target_type == "custom" and target_coords_cols:
        cmd.extend(["--target-x-col", target_coords_cols[0]])
        cmd.extend(["--target-y-col", target_coords_cols[1]])
    
    # Run ST Align
    logger.info(f"Running ST Align: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("ST Align completed successfully")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"ST Align failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None
    
    # Load alignment results
    alignment_results = load_alignment(output_dir)
    
    return alignment_results

def load_alignment(alignment_dir):
    """
    Load alignment results from a directory.
    
    Args:
        alignment_dir (str): Directory containing alignment results
    
    Returns:
        dict: Alignment results with transformation matrix and aligned data
    """
    logger = logging.getLogger(__name__)
    alignment_dir = Path(alignment_dir)
    
    # Check for alignment files
    transform_file = alignment_dir / "transformation.json"
    aligned_file = alignment_dir / "aligned_coordinates.csv"
    
    if not transform_file.exists():
        logger.error(f"Transformation file not found: {transform_file}")
        return None
    
    if not aligned_file.exists():
        logger.error(f"Aligned coordinates file not found: {aligned_file}")
        return None
    
    # Load the transformation matrix
    try:
        with open(transform_file, 'r') as f:
            transform_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading transformation file: {e}")
        return None
    
    # Load the aligned coordinates
    try:
        aligned_coords = pd.read_csv(aligned_file)
    except Exception as e:
        logger.error(f"Error loading aligned coordinates: {e}")
        return None
    
    # Load the original reference and target files if available
    reference_file = alignment_dir / "reference.csv"
    target_file = alignment_dir / "target.csv"
    
    reference_data = None
    target_data = None
    
    if reference_file.exists():
        try:
            reference_data = pd.read_csv(reference_file)
        except Exception as e:
            logger.warning(f"Error loading reference file: {e}")
    
    if target_file.exists():
        try:
            target_data = pd.read_csv(target_file)
        except Exception as e:
            logger.warning(f"Error loading target file: {e}")
    
    # Return the alignment results
    return {
        'transformation': transform_data,
        'aligned_coordinates': aligned_coords,
        'reference_data': reference_data,
        'target_data': target_data,
        'alignment_dir': str(alignment_dir)
    }

def save_alignment(alignment_results, output_dir):
    """
    Save alignment results to a directory.
    
    Args:
        alignment_results (dict): Alignment results to save
        output_dir (str): Directory to save the results
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the transformation matrix
    try:
        transform_file = output_dir / "transformation.json"
        with open(transform_file, 'w') as f:
            json.dump(alignment_results['transformation'], f, indent=2)
    except Exception as e:
        logger.error(f"Error saving transformation file: {e}")
        return False
    
    # Save the aligned coordinates
    try:
        aligned_file = output_dir / "aligned_coordinates.csv"
        alignment_results['aligned_coordinates'].to_csv(aligned_file, index=False)
    except Exception as e:
        logger.error(f"Error saving aligned coordinates: {e}")
        return False
    
    # Save the reference data if available
    if 'reference_data' in alignment_results and alignment_results['reference_data'] is not None:
        try:
            reference_file = output_dir / "reference.csv"
            alignment_results['reference_data'].to_csv(reference_file, index=False)
        except Exception as e:
            logger.warning(f"Error saving reference data: {e}")
    
    # Save the target data if available
    if 'target_data' in alignment_results and alignment_results['target_data'] is not None:
        try:
            target_file = output_dir / "target.csv"
            alignment_results['target_data'].to_csv(target_file, index=False)
        except Exception as e:
            logger.warning(f"Error saving target data: {e}")
    
    return True

def apply_transformation(points, transformation, inverse=False):
    """
    Apply a transformation matrix to a set of points.
    
    Args:
        points (numpy.ndarray): Points to transform, shape (N, 2)
        transformation (dict): Transformation dictionary with 'matrix' key
        inverse (bool): Whether to apply the inverse transformation
    
    Returns:
        numpy.ndarray: Transformed points, shape (N, 2)
    """
    # Extract transformation matrix
    matrix = np.array(transformation['matrix'])
    
    # Invert if needed
    if inverse:
        matrix = np.linalg.inv(matrix)
    
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation
    if matrix.shape[0] == 2:  # 2x3 matrix (affine)
        # Add row for homogeneous coordinates
        matrix = np.vstack((matrix, [0, 0, 1]))
    
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    # Convert back from homogeneous coordinates
    if matrix.shape[0] == 3:  # 3x3 matrix (projective)
        # Divide by the last coordinate
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]
    else:
        transformed_points = transformed_points[:, :2]
    
    return transformed_points

def prepare_visium_data(counts_file, spatial_file, output_file=None):
    """
    Prepare Visium data for alignment.
    
    Args:
        counts_file (str): Path to the counts matrix file
        spatial_file (str): Path to the spatial coordinates file
        output_file (str, optional): Path to save the prepared data
    
    Returns:
        pandas.DataFrame: Prepared data with counts and coordinates
    """
    logger = logging.getLogger(__name__)
    
    # Load the counts matrix
    try:
        if str(counts_file).endswith('.csv'):
            counts = pd.read_csv(counts_file, index_col=0)
        elif str(counts_file).endswith('.tsv'):
            counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        else:
            logger.error(f"Unsupported file format for counts: {counts_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading counts file: {e}")
        return None
    
    # Load the spatial coordinates
    try:
        if str(spatial_file).endswith('.csv'):
            spatial = pd.read_csv(spatial_file)
        elif str(spatial_file).endswith('.tsv'):
            spatial = pd.read_csv(spatial_file, sep='\t')
        else:
            logger.error(f"Unsupported file format for spatial coordinates: {spatial_file}")
            return None
    except Exception as e:
        logger.error(f"Error loading spatial file: {e}")
        return None
    
    # Merge the data
    try:
        # Assume spatial file has barcode, x, and y columns
        spatial_cols = list(spatial.columns)
        barcode_col = next((col for col in spatial_cols if 'barcode' in col.lower()), spatial_cols[0])
        
        # Match counts indices with barcodes
        barcodes = set(spatial[barcode_col])
        common_barcodes = [bc for bc in counts.index if bc in barcodes]
        
        if not common_barcodes:
            logger.error("No matching barcodes found between counts and spatial data")
            return None
        
        # Filter counts to common barcodes
        counts_filtered = counts.loc[common_barcodes]
        
        # Create output dataframe with barcodes, coordinates, and top genes
        result = spatial[spatial[barcode_col].isin(common_barcodes)].copy()
        
        # Add gene counts (use top 50 most variable genes)
        if counts_filtered.shape[1] > 50:
            var = counts_filtered.var(axis=0)
            top_genes = var.nlargest(50).index
            for gene in top_genes:
                result[gene] = result[barcode_col].map(counts_filtered[gene])
        else:
            # Use all genes if fewer than 50
            for gene in counts_filtered.columns:
                result[gene] = result[barcode_col].map(counts_filtered[gene])
        
        logger.info(f"Prepared data with {len(result)} spots and {len(result.columns) - len(spatial_cols)} genes")
        
        # Save to file if requested
        if output_file:
            result.to_csv(output_file, index=False)
            logger.info(f"Saved prepared data to {output_file}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error preparing Visium data: {e}")
        return None
