"""
Preprocessing utilities for OME-TIFF images.

This module provides functions for preprocessing and normalizing
multichannel OME-TIFF images before segmentation.
"""

import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter
import logging

def preprocess_ome_tiff(image_path, sigma=1.0, normalize=True, channel_names=None):
    """
    Preprocess a multichannel OME-TIFF image.

    Args:
        image_path (str): Path to the OME-TIFF file
        sigma (float): Gaussian blur sigma value
        normalize (bool): Whether to normalize image intensity
        channel_names (list, optional): Names of channels in the image

    Returns:
        dict: Dictionary with channel names as keys and preprocessed arrays as values
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading image: {image_path}")
    
    # Load the image
    try:
        with tifffile.TiffFile(image_path) as tif:
            # Load the full image
            image = tif.asarray()
            logger.info(f"Image shape: {image.shape}")
            logger.info(f"Image dimensions: {len(image.shape)}")
            
            # Get metadata
            if channel_names is None:
                # Try to extract channel names from metadata or use defaults
                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    # Parse OME metadata for channel names
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(tif.ome_metadata)
                        channels = []
                        for ch in root.findall(".//Channel"):
                            name = ch.get('Name') or f"Channel-{len(channels)}"
                            channels.append(name)
                        
                        if channels:
                            channel_names = channels
                            logger.info(f"Extracted channel names from OME metadata: {channel_names}")
                    except Exception as e:
                        logger.warning(f"Failed to parse OME metadata for channel names: {e}")
                        
                # If channel names still not available, use default
                if channel_names is None:
                    # Create default channel names
                    if len(image.shape) >= 3:
                        channel_names = [f'Channel-{i}' for i in range(image.shape[0])]
                    else:
                        channel_names = ['Channel-0']
                    
                    logger.info(f"Using default channel names: {channel_names}")
            
            # Extract dimensions
            dimensions = len(image.shape)
            
            # Check if dimensions are as expected
            if dimensions not in [2, 3, 4, 5]:
                raise ValueError(f"Unexpected image dimensions: {dimensions}")
            
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise

    # Dictionary to store preprocessed channels
    preprocessed_channels = {}

    # Process each channel
    for c, channel_name in enumerate(channel_names):
        if c >= image.shape[0] and dimensions >= 3:
            logger.warning(f"Channel index {c} out of bounds for image with shape {image.shape}")
            continue
            
        logger.info(f"Processing channel: {channel_name}")

        # Extract single channel based on actual dimensions
        try:
            if dimensions == 5:  # TCZYX format
                channel = image[0, c, 0, :, :]
            elif dimensions == 4:  # CZYX format
                channel = image[c, 0, :, :]
            elif dimensions == 3:  # CYX format
                channel = image[c, :, :]
            elif dimensions == 2:  # YX format (single channel)
                channel = image
            else:
                raise ValueError(f"Unexpected image dimensions: {image.shape}")
        except IndexError as e:
            logger.error(f"Error extracting channel {c}: {e}")
            continue

        # Convert to float32 for processing
        channel = channel.astype(np.float32)

        # Apply Gaussian blur using scipy
        if sigma > 0:
            logger.info(f"Applying Gaussian blur with sigma={sigma}")
            channel = gaussian_filter(channel, sigma=sigma)

        # Normalize if requested
        if normalize:
            logger.info("Normalizing intensity")
            channel = normalize_min_max(channel)

        # Convert to uint8
        channel = (channel * 255).clip(0, 255).astype(np.uint8)

        # Store result
        preprocessed_channels[channel_name] = channel
        logger.info(f"Processed channel: {channel_name} with shape {channel.shape}")

    return preprocessed_channels

def normalize_min_max(image):
    """
    Normalize image intensity to [0,1] range

    Args:
        image (numpy.ndarray): Input image

    Returns:
        numpy.ndarray: Normalized image
    """
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return image * 0  # Return zeros if image is constant

    return (image - min_val) / (max_val - min_val)

def combine_channels(channel1, channel2, normalize_result=True):
    """
    Combine two channels by adding their pixel values

    Args:
        channel1 (numpy.ndarray): First channel
        channel2 (numpy.ndarray): Second channel
        normalize_result (bool): Whether to normalize the combined result to uint8 range

    Returns:
        numpy.ndarray: Combined channel
    """
    # Add channels
    combined = channel1.astype(np.float32) + channel2.astype(np.float32)

    if normalize_result:
        # Normalize to 0-255 range
        combined = normalize_min_max(combined) * 255

    # Clip and convert to uint8
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    return combined

def create_rgb_image(channels_dict, red_channel=None, green_channel=None, blue_channel=None):
    """
    Create an RGB image from selected channels.
    
    Args:
        channels_dict (dict): Dictionary of preprocessed channels
        red_channel (str, optional): Name of channel to use for red
        green_channel (str, optional): Name of channel to use for green
        blue_channel (str, optional): Name of channel to use for blue
    
    Returns:
        numpy.ndarray: RGB image
    """
    logger = logging.getLogger(__name__)
    
    if not channels_dict:
        logger.error("No channels provided")
        return None
    
    # Get shape from the first channel
    first_channel = next(iter(channels_dict.values()))
    height, width = first_channel.shape
    
    # Create empty RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign channels
    if red_channel and red_channel in channels_dict:
        rgb_image[:, :, 0] = channels_dict[red_channel]
    
    if green_channel and green_channel in channels_dict:
        rgb_image[:, :, 1] = channels_dict[green_channel]
    
    if blue_channel and blue_channel in channels_dict:
        rgb_image[:, :, 2] = channels_dict[blue_channel]
    
    return rgb_image

def save_preprocessed_channels(channels_dict, output_dir, merged_output_path=None):
    """
    Save preprocessed channels to TIFF files.
    
    Args:
        channels_dict (dict): Dictionary of preprocessed channels
        output_dir (str): Directory to save individual channels
        merged_output_path (str, optional): Path to save merged channels
    
    Returns:
        dict: Dictionary with paths to saved files
    """
    import os
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    saved_paths = {}
    
    # Save individual channels
    for channel_name, channel_data in channels_dict.items():
        output_path = os.path.join(output_dir, f"{channel_name}_processed.tif")
        tifffile.imwrite(output_path, channel_data)
        saved_paths[channel_name] = output_path
        logger.info(f"Saved {channel_name} to {output_path}")
    
    # Save merged channels if requested
    if merged_output_path and len(channels_dict) > 1:
        # Stack channels
        channel_names = list(channels_dict.keys())
        stacked_data = np.stack([channels_dict[ch] for ch in channel_names])
        
        tifffile.imwrite(merged_output_path, stacked_data)
        saved_paths['merged'] = merged_output_path
        logger.info(f"Saved merged channels to {merged_output_path}")
    
    return saved_paths
