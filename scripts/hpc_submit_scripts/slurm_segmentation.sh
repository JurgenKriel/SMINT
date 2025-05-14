#!/bin/bash
#SBATCH --job-name=smint_seg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=%j_smint_seg.out
#SBATCH --error=%j_smint_seg.err
#SBATCH --time=24:00:00

# SMINT Segmentation SLURM submission script
# This script submits a segmentation job to the SLURM scheduler
# 
# Usage: sbatch slurm_segmentation.sh [config_file] [image_file] [output_dir]
#
# Example: sbatch slurm_segmentation.sh configs/segmentation_config.yaml data/image.tif results

# Check arguments
CONFIG_FILE=${1:-"configs/segmentation_config.yaml"}
IMAGE_FILE=${2}
OUTPUT_DIR=${3:-"results/segmentation"}

# Check if image file is provided
if [ -z "$IMAGE_FILE" ]; then
    echo "Error: Image file not provided"
    echo "Usage: sbatch slurm_segmentation.sh [config_file] [image_file] [output_dir]"
    exit 1
fi

# Print information
echo "=== SMINT Segmentation SLURM Job ==="
echo "Starting at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Configuration file: $CONFIG_FILE"
echo "Image file: $IMAGE_FILE"
echo "Output directory: $OUTPUT_DIR"
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

# Run the segmentation script
echo "Running segmentation..."

# GPU options
GPU_OPTIONS="--use-gpu"

# Check if using multi-GPU
if [ -n "$SLURM_GPUS" ] && [ "$SLURM_GPUS" -gt 1 ]; then
    echo "Using multi-GPU configuration with $SLURM_GPUS GPUs"
    GPU_OPTIONS="$GPU_OPTIONS --distributed"
fi

# Run the segmentation
python -m scripts.run_segmentation \
    --image "$IMAGE_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    $GPU_OPTIONS \
    --visualize \
    --live-viewer \
    --save-config

# Check if segmentation was successful
if [ $? -eq 0 ]; then
    echo "Segmentation completed successfully"
else
    echo "Segmentation failed with exit code $?"
    exit 1
fi

# Print summary
echo "=== Segmentation Summary ==="
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Cell outlines: $OUTPUT_DIR/cell_outlines_cells.csv"
echo "Nuclei outlines: $OUTPUT_DIR/cell_outlines_nuclei.csv"
echo "==========================="

exit 0
