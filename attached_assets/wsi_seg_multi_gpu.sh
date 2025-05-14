#!/bin/bash

#SBATCH --job-name=cellpose_dual_wsi_live
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_%j.out
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kriel.j@wehi.edu.au

#############################################################################
# Cellpose Multi-GPU Segmentation SLURM Script w/ CSV Export
# ... (Usage instructions updated) ...
# Options can be passed as environment variables:
#   ... (existing options) ...
#   OUTPUT_CSV_BASE: Base path for output CSV files (e.g., /path/outlines)
#
# Example:
#   ... \
#   OUTPUT_CSV_BASE="results/my_experiment_outlines" \
#   sbatch run_cellpose_segmentation.slurm
#############################################################################

# ... (Job info printing) ...
# ... (Module loading) ...

# Set default values
IMAGE_PATH=${IMAGE_PATH:-"/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_combined_af_merged_image.ome.tif"}
OUTPUT_DIR=${OUTPUT_DIR:-"output_segmentation_combined"}
# --- ADD CSV BASE PATH ---
OUTPUT_CSV_BASE=${OUTPUT_CSV_BASE:-"${OUTPUT_DIR}/outlines"} # Default relative to output_dir

# --- Model Types and Parameters ---
MODEL_TYPES=${MODEL_TYPES:-"/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700 nuclei"}
DIAMETERS=${DIAMETERS:-"120.0 60.0"}
FLOW_THRESHOLDS=${FLOW_THRESHOLDS:-"0.4 0.4"}
CELLPROB_THRESHOLDS=${CELLPROB_THRESHOLDS:-"-1.5 -1.2"}
EVAL_CHANNELS=${EVAL_CHANNELS:-"0,0 0,0"}

# --- Other Parameters ---
CHUNK_SIZE_MB=${CHUNK_SIZE_MB:-250}
OVERLAP_PIXELS=${OVERLAP_PIXELS:-60}
PLOT_EVERY_N_CHUNKS=${PLOT_EVERY_N_CHUNKS:-0} # Default plotting off
BATCH_SIZE=${BATCH_SIZE:-1}

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$OUTPUT_CSV_BASE")" # Create dir for CSVs
mkdir -p "/vast/projects/BCRL_Multi_Omics/slurm_logs"

# Print configuration
echo "Configuration:"
echo "  IMAGE_PATH: $IMAGE_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  OUTPUT_CSV_BASE: $OUTPUT_CSV_BASE" # Print new path
echo "  MODEL_TYPES: $MODEL_TYPES"
# ... (print other params) ...

# ... (Environment setup) ...

# Define the path to your Python script
PYTHON_SCRIPT_PATH="/vast/projects/BCRL_Multi_Omics/segmentation_pipeline/cellpose_wsi_multi_gpu.py" # Adjust if needed

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT_PATH"; exit 1; fi

# Run the optimized script
echo "Starting Cellpose segmentation..."
python "$PYTHON_SCRIPT_PATH" \
  --image_path "$IMAGE_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --output_csv_base "$OUTPUT_CSV_BASE" `# <-- ADDED ARGUMENT` \
  --model_types $MODEL_TYPES \
  --diameter $DIAMETERS \
  --flow_threshold $FLOW_THRESHOLDS \
  --cellprob_threshold $CELLPROB_THRESHOLDS \
  --eval_channels $EVAL_CHANNELS \
  --chunk_size_mb $CHUNK_SIZE_MB \
  --overlap_pixels $OVERLAP_PIXELS \
  --plot_every_n_chunks $PLOT_EVERY_N_CHUNKS \
  --batch_size $BATCH_SIZE

# Check exit status
EXIT_CODE=$?
# ... (Rest of the script: exit code check, summary file, etc.) ...
# Add OUTPUT_CSV_BASE to summary file
cat >> "${OUTPUT_DIR}/job_summary.txt" << EOF

CSV Output Base: $OUTPUT_CSV_BASE
EOF

echo "Job completed at: $(date)"
exit $EXIT_CODE