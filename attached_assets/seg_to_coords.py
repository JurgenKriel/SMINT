import numpy as np
import os
from pathlib import Path
import pandas as pd
import cv2
from cellpose import io

def load_and_save_masks(seg_files):
    """
    Load .npy files and save as PNG masks using cellpose.io.save_masks
    """
    for seg_file in seg_files:
        # Load the segmentation file
        masks_dict = np.load(seg_file, allow_pickle=True).item()

        # Extract the masks array (assuming it's stored under a key like 'masks' or similar)
        # Print the keys to see what's available
        print(f"Keys in the loaded file: {masks_dict.keys()}")

        # Assuming the masks are stored under 'masks' key
        if 'masks' in masks_dict:
            masks = masks_dict['masks']
        else:
            # If masks are stored differently, we might need to adjust this
            # Let's try the first key if 'masks' isn't present
            first_key = list(masks_dict.keys())[0]
            masks = masks_dict[first_key]

        # Ensure masks is a numpy array
        masks = np.array(masks)

        # Create output filename
        out_file = str(seg_file).replace('_seg.npy', '_masks.png')

        # Save as PNG using direct OpenCV writing to avoid cellpose.io issues
        cv2.imwrite(out_file, masks.astype(np.uint16))
        print(f"Saved mask to {out_file}")

def extract_contours_from_mask(mask_file):
    """
    Extract contours from a binary mask image
    """
    # Read the mask image
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)  # Changed to IMREAD_UNCHANGED

    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, unique values: {np.unique(mask)}")

    # Find unique labels (excluding background - 0)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

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
                    'file_name': mask_file.name,
                    'cell_id': int(label),
                    'x': int(coord[0]),
                    'y': int(coord[1])
                })

    return contours_data

def main():
    # Specify input directory containing _seg.npy files
    input_dir = '/vast/scratch/users/kriel.j/venture_pt5/4051_tile_export/composite_rgb/cellpose_output/'  # Current directory, modify as needed

    # Get all _seg.npy files
    #seg_files = list(Path(input_dir).glob('*_seg.npy'))

    #print(f"Found {len(seg_files)} segmentation files")

    # Save masks as PNG
    #for seg_file in seg_files:
        #print(f"Processing {seg_file}")
        #load_and_save_masks([seg_file])

    # Get all mask PNG files
    mask_files = list(Path(input_dir).glob('*_masks.png'))
    print(f"Found {len(mask_files)} mask files")

    # Extract contours from all masks
    all_contours = []
    for mask_file in mask_files:
        print(f"Extracting contours from {mask_file}")
        contours = extract_contours_from_mask(mask_file)
        all_contours.extend(contours)

    # Create DataFrame and save to CSV
    coords_df = pd.DataFrame(all_contours)
    output_file = 'cell_coordinates.csv'
    coords_df.to_csv(output_file, index=False)

    # Print created files
    print("\nCreated/Modified files during execution:")
    print([f.name for f in Path(input_dir).glob('*_masks.png')])
    print("cell_coordinates.csv")

if __name__ == "__main__":
    main()
