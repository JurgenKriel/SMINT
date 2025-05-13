#!/bin/bash
#SBATCH --job-name=smint_preprocess
#SBATCH --partition=regular
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=%j_smint_preprocess.out
#SBATCH --error=%j_smint_preprocess.err
#SBATCH --time=12:00:00

# SMINT Preprocessing SLURM submission script
# This script submits a preprocessing job to the SLURM scheduler
# 
# Usage: sbatch slurm_preprocess.sh [ome_tiff_file] [output_dir] [sigma]
#
# Example: sbatch slurm_preprocess.sh data/image.ome.tif results/preprocessing 1.5

# Check arguments
OME_TIFF_FILE=${1}
OUTPUT_DIR=${2:-"results/preprocessing"}
SIGMA=${3:-1.0}

# Check if OME-TIFF file is provided
if [ -z "$OME_TIFF_FILE" ]; then
    echo "Error: OME-TIFF file not provided"
    echo "Usage: sbatch slurm_preprocess.sh [ome_tiff_file] [output_dir] [sigma]"
    exit 1
fi

# Print information
echo "=== SMINT Preprocessing SLURM Job ==="
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "OME-TIFF file: $OME_TIFF_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Sigma: $SIGMA"
echo "======================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Set up environment
echo "Setting up environment..."

# Set up Python environment
# Uncomment and modify the following lines according to your environment setup
# module load anaconda/3
# source activate smint

# Force immediate output flushing
export PYTHONUNBUFFERED=1

# Run the preprocessing script
echo "Running preprocessing..."

# Create a temporary Python script to run the preprocessing
TEMP_SCRIPT=$(mktemp)
cat << EOF > $TEMP_SCRIPT
import os
import sys
from smint.preprocessing import preprocess_ome_tiff, combine_channels
import tifffile
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('smint_preprocess')

# Get arguments
ome_tiff_file = "$OME_TIFF_FILE"
output_dir = "$OUTPUT_DIR"
sigma = float("$SIGMA")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Run preprocessing
logger.info(f"Processing {ome_tiff_file} with sigma={sigma}")
preprocessed = preprocess_ome_tiff(
    image_path=ome_tiff_file,
    sigma=sigma,
    normalize=True
)

if not preprocessed:
    logger.error("Preprocessing failed")
    sys.exit(1)

logger.info(f"Preprocessed {len(preprocessed)} channels")

# Save individual channels
for channel_name, channel_data in preprocessed.items():
    output_path = os.path.join(output_dir, f"{channel_name}.tif")
    tifffile.imwrite(output_path, channel_data)
    logger.info(f"Saved {channel_name} to {output_path}")

# Combine specific channels if there are multiple
if len(preprocessed) >= 2:
    channel_names = list(preprocessed.keys())
    channel1 = preprocessed[channel_names[0]]
    channel2 = preprocessed[channel_names[1]]
    
    combined = combine_channels(channel1, channel2, normalize_result=True)
    combined_path = os.path.join(output_dir, f"{channel_names[0]}_{channel_names[1]}_combined.tif")
    tifffile.imwrite(combined_path, combined)
    logger.info(f"Saved combined {channel_names[0]}+{channel_names[1]} to {combined_path}")
    
    # Create RGB composite if there are at least 3 channels
    if len(preprocessed) >= 3:
        rgb = np.zeros((channel1.shape[0], channel1.shape[1], 3), dtype=np.uint8)
        
        for i, channel_name in enumerate(list(preprocessed.keys())[:3]):
            rgb[:, :, i] = preprocessed[channel_name]
        
        rgb_path = os.path.join(output_dir, "rgb_composite.tif")
        tifffile.imwrite(rgb_path, rgb)
        logger.info(f"Saved RGB composite to {rgb_path}")

logger.info("Preprocessing completed successfully")
EOF

# Run the Python script
python $TEMP_SCRIPT

# Check if preprocessing was successful
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully"
else
    echo "Preprocessing failed with exit code $?"
    exit 1
fi

# Remove temporary script
rm $TEMP_SCRIPT

# Print summary
echo "=== Preprocessing Summary ==="
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "==========================="

exit 0
