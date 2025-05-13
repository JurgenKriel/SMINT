import tifffile
import numpy as np

def preprocess_ome_tiff(image_path, sigma=1.0, normalize=True):
    """
    Preprocess a multichannel OME-TIFF image.

    Args:
        image_path (str): Path to the OME-TIFF file
        sigma (float): Gaussian blur sigma value
        normalize (bool): Whether to normalize image intensity

    Returns:
        dict: Dictionary with channel names as keys and preprocessed arrays as values
    """
    # Load the image
    print("Loading image...")
    with tifffile.TiffFile(image_path) as tif:
        # Load the full image
        image = tif.asarray()
        print(f"Image shape: {image.shape}")
        print(f"Image dimensions: {len(image.shape)}")

        # Get metadata
        channel_names = ['AF568-T1', 'DAPI-T2', 'AF647-T2']

    # Dictionary to store preprocessed channels
    preprocessed_channels = {}

    # Process each channel
    for c, channel_name in enumerate(channel_names):
        print(f"Processing channel: {channel_name}")

        # Extract single channel based on actual dimensions
        if len(image.shape) == 5:  # TCZYX format
            channel = image[0, c, 0, :, :]
        elif len(image.shape) == 4:  # CZYX format
            channel = image[c, 0, :, :]
        elif len(image.shape) == 3:  # CYX format
            channel = image[c, :, :]
        else:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")

        # Convert to float32 for processing
        channel = channel.astype(np.float32)

        # Apply Gaussian blur using numpy
        if sigma > 0:
            from scipy.ndimage import gaussian_filter
            channel = gaussian_filter(channel, sigma=sigma)

        # Normalize if requested
        if normalize:
            channel = normalize_min_max(channel)

        # Convert to uint8
        channel = (channel * 255).clip(0, 255).astype(np.uint8)

        # Store result
        preprocessed_channels[channel_name] = channel

    return preprocessed_channels

def normalize_min_max(image):
    """
    Normalize image intensity to [0,1] range
    """
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return image

    return (image - min_val) / (max_val - min_val)

def combine_channels(channel1, channel2, normalize_result=True):
    """
    Combine two channels by adding their pixel values

    Args:
        channel1 (np.ndarray): First channel
        channel2 (np.ndarray): Second channel
        normalize_result (bool): Whether to normalize the combined result to uint8 range

    Returns:
        np.ndarray: Combined channel
    """
    # Add channels
    combined = channel1.astype(np.float32) + channel2.astype(np.float32)

    if normalize_result:
        # Normalize to 0-255 range
        combined = normalize_min_max(combined) * 255

    # Clip and convert to uint8
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    return combined

# Usage example
if __name__ == "__main__":
    image_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/Venture_Pt5_4051_bottom-06-Orthogonal Projection-01-Stitching-03.ome.tif"

    # Preprocess channels
    print("Starting preprocessing...")
    preprocessed = preprocess_ome_tiff(image_path, sigma=1.5, normalize=True)

    # Get individual channels
    af568 = preprocessed['AF568-T1']
    af647 = preprocessed['AF647-T2']
    dapi = preprocessed['DAPI-T2']

    # Combine AF568 and AF647
    print("Combining channels...")
    combined = combine_channels(af568, af647, normalize_result=True)

    print("Processing complete!")
    print(f"Combined channel shape: {combined.shape}")

    # Optionally save the results
    # Save individual channels
    tifffile.imwrite('/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/af568_processed.tif', af568)
    tifffile.imwrite('/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/af647_processed.tif', af647)
    tifffile.imwrite('//vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_processed.tif', dapi)

    # Save combined channel
    tifffile.imwrite('combined_af568_af647.tif', combined)