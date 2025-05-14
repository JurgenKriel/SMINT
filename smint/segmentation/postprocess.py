"""
Post-processing utilities for cell segmentation results.

This module provides functions for processing segmentation results,
extracting contours, and saving masks and outlines.
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging

# Optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) package not available. Some functionality will be limited.")

try:
    from cellpose import io
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logging.warning("Cellpose package not available. Some functionality will be limited.")

def save_masks(masks_dict, output_path):
    """
    Save segmentation masks as PNG files.
    
    Args:
        masks_dict (dict): Dictionary containing masks from segmentation
        output_path (str): Path to save the masks
    
    Returns:
        str: Path to the saved mask file
    """
    if not CV2_AVAILABLE:
        logging.error("Cannot save masks: OpenCV (cv2) is required but not available")
        return None
        
    # Ensure masks is a numpy array
    if 'masks' in masks_dict:
        masks = masks_dict['masks']
    else:
        # If masks are stored differently, use the first key
        first_key = list(masks_dict.keys())[0]
        masks = masks_dict[first_key]
    
    masks = np.array(masks)
    
    # Create output filename
    if output_path.endswith('.npy'):
        out_file = output_path.replace('.npy', '.png')
    else:
        out_file = output_path + '.png'
    
    # Save as PNG using direct OpenCV writing
    cv2.imwrite(out_file, masks.astype(np.uint16))
    return out_file

def extract_contours(mask_file, output_csv=None):
    """
    Extract contours from a mask file and optionally save to CSV.
    
    Args:
        mask_file (str): Path to mask file
        output_csv (str, optional): Path to save CSV file
    
    Returns:
        list: List of dictionaries containing contour information
    """
    logger = logging.getLogger(__name__)
    
    if not CV2_AVAILABLE:
        logger.error("Cannot extract contours: OpenCV (cv2) is required but not available")
        return []
    
    # Read the mask image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    logger.info(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)[:10]}")
    
    # Find unique labels (excluding background - 0)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    logger.info(f"Found {len(unique_labels)} unique cell IDs")
    
    contours_data = []
    
    for label in unique_labels:
        # Create binary mask for current label
        binary_mask = (mask == label).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Extract coordinates
            coords = largest_contour.squeeze()
            
            # Handle single point case
            if len(coords.shape) == 1:
                coords = coords.reshape(1, -1)
            
            # Add to data list
            for coord in coords:
                contours_data.append({
                    'file_name': os.path.basename(mask_file),
                    'cell_id': int(label),
                    'x': int(coord[0]),
                    'y': int(coord[1])
                })
    
    # Save to CSV if requested
    if output_csv and contours_data:
        df = pd.DataFrame(contours_data)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df)} contour points to {output_csv}")
    
    return contours_data

def combine_segmentation_results(csv_files, output_csv=None):
    """
    Combine multiple segmentation result CSV files.
    
    Args:
        csv_files (list): List of CSV file paths
        output_csv (str, optional): Path to save combined CSV
    
    Returns:
        pandas.DataFrame: Combined dataframe
    """
    logger = logging.getLogger(__name__)
    
    all_dfs = []
    total_cells = 0
    current_max_id = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if we need to update cell IDs to avoid duplicates
            if current_max_id > 0:
                if 'global_cell_id' in df.columns:
                    df['global_cell_id'] += current_max_id
                elif 'cell_id' in df.columns:
                    df['cell_id'] += current_max_id
            
            # Update current_max_id
            if 'global_cell_id' in df.columns:
                current_max_id = df['global_cell_id'].max() + 1
            elif 'cell_id' in df.columns:
                current_max_id = df['cell_id'].max() + 1
            
            # Add source file
            df['source_file'] = os.path.basename(csv_file)
            
            all_dfs.append(df)
            
            # Count unique cells
            if 'global_cell_id' in df.columns:
                total_cells += df['global_cell_id'].nunique()
            elif 'cell_id' in df.columns:
                total_cells += df['cell_id'].nunique()
                
            logger.info(f"Loaded {csv_file}: {len(df)} points")
            
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if not all_dfs:
        logger.warning("No data loaded from CSV files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined {len(all_dfs)} files: {len(combined_df)} points, {total_cells} unique cells")
    
    # Save if requested
    if output_csv:
        combined_df.to_csv(output_csv, index=False)
        logger.info(f"Saved combined results to {output_csv}")
    
    return combined_df

def calculate_centroids(contours_df, output_csv=None):
    """
    Calculate centroids for cells from contour data.
    
    Args:
        contours_df (pandas.DataFrame): DataFrame with contour points
        output_csv (str, optional): Path to save centroids CSV
    
    Returns:
        pandas.DataFrame: DataFrame with cell centroids
    """
    logger = logging.getLogger(__name__)
    
    # Check for required columns
    id_col = None
    for col in ['global_cell_id', 'cell_id']:
        if col in contours_df.columns:
            id_col = col
            break
    
    if id_col is None:
        logger.error("No cell ID column found in contours DataFrame")
        return None
    
    if 'x' not in contours_df.columns or 'y' not in contours_df.columns:
        logger.error("No x/y coordinate columns found in contours DataFrame")
        return None
    
    # Group by cell ID and calculate centroids
    centroids = []
    
    for cell_id, group in contours_df.groupby(id_col):
        x_mean = group['x'].mean()
        y_mean = group['y'].mean()
        
        # Add any additional columns that might be useful
        cell_data = {'cell_id': cell_id, 'centroid_x': x_mean, 'centroid_y': y_mean}
        
        # Add extra columns if they exist
        for col in ['chunk_id', 'source_file']:
            if col in group.columns:
                cell_data[col] = group[col].iloc[0]
        
        centroids.append(cell_data)
    
    centroids_df = pd.DataFrame(centroids)
    logger.info(f"Calculated centroids for {len(centroids_df)} cells")
    
    # Save if requested
    if output_csv:
        centroids_df.to_csv(output_csv, index=False)
        logger.info(f"Saved centroids to {output_csv}")
    
    return centroids_df
