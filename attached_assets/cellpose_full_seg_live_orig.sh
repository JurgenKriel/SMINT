#!/bin/bash

#SBATCH --job-name=cellpose_dual_wsi_live # Updated job name
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1        # Request 1 A100 GPU
#SBATCH --cpus-per-task=16       # CPUs for I/O, numpy etc.
#SBATCH --mem=128G               # Adjust based on chunk size and image size
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_%j.out # Log directory
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_%j.err  # Log directory
#SBATCH --time=24:00:00          # Adjust as needed
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kriel.j@wehi.edu.au

# --- Load Modules & Activate Environment ---
echo "Loading modules..."
module purge # Start with a clean environment
# module load cuda/11.8 # Or the version compatible with your Cellpose/PyTorch install

echo "Activating Python environment..."
source /vast/projects/BCRL_Multi_Omics/cellpose_env/bin/activate
# conda activate /path/to/your/conda/envs/cellpose_env # Or use conda activate

# Verify activation (optional)
echo "Python executable: $(which python)"
echo "Environment: $CONDA_DEFAULT_ENV" # Check conda env name if using conda

# Improve error handling: exit on error, treat unset variables as errors
set -e
set -u
set -o pipefail

# --- Define Paths and Parameters ---
echo "Setting up paths and parameters..."
IMAGE_FILE="/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_combined_af_merged_image.ome.tif"

# --- Model Paths ---
CELL_MODEL_PATH="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700" # Your custom cell model
NUCLEI_MODEL_PATH="nuclei" # Use built-in 'nuclei' model (or provide a path)

# --- Output Configuration ---
# Create a unique output directory based on Job ID
BASE_OUTPUT_DIR="/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/segmentation_output_dual_live_${SLURM_JOB_ID}"
OUTPUT_BASENAME="Venture5_WSI_DualSegmentation" # Base name for files within output dir
PYTHON_SCRIPT="/vast/projects/BCRL_Multi_Omics/segmentation_pipeline/wsi_cell_seg_for_viewer_orig.py" # Your Python script

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"
echo "Output directory: $BASE_OUTPUT_DIR"

# Define output base path/prefix (script appends _cells.csv, _nuclei.csv, .log)
CSV_OUTPUT_BASE="${BASE_OUTPUT_DIR}/${OUTPUT_BASENAME}"
VIS_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_BASENAME}_visualizations" # Specific subdir for summary plots

# --- Cell Model Parameters ---
CELLS_DIAMETER=120.0
CELLS_FLOW_THRESH=0.4
CELLS_CELLPROB_THRESH=-1.5
CELLS_CHANNELS="1 2" # Cytoplasm=1, Nucleus=2 (0-indexed from selected channels for model)

# --- Nuclei Model Parameters ---
NUCLEI_DIAMETER=60.0
NUCLEI_FLOW_THRESH=0.4 # This is the CONSTANT flow threshold for nuclei
NUCLEI_CELLPROB_THRESH=-1.0 # This is the INITIAL cellprob threshold for nuclei
NUCLEI_CHANNELS="2 1" # e.g., Use original TIFF channel 2 as grayscale (model ch0), original TIFF ch1 as model ch1 (ignored by grayscale model)

# --- Adaptive Nuclei Segmentation (Cellprob DECREASING) Parameters ---
ENABLE_ADAPTIVE_NUCLEI="true" # "true" or "false"
NUCLEI_ADAPTIVE_CELLPROB_LOWER_LIMIT=-6.0
NUCLEI_ADAPTIVE_CELLPROB_STEP_DECREMENT=0.2
NUCLEI_MAX_ADAPTIVE_ATTEMPTS=3
ADAPTIVE_NUCLEI_TRIGGER_RATIO=0.05

# --- Processing Parameters ---
CHUNK_Y=2048
CHUNK_X=2048

# --- Summary Visualization Parameters (End of Run) ---
ENABLE_SUMMARY_VISUALIZATION="true" # "true" to enable, "false" to disable
NUM_SUMMARY_VIS_CHUNKS=10
VIS_ROI_Y=2024 # Smaller ROI for summary plots
VIS_ROI_X=2024
VIS_BG_CHANNELS_FOR_PLOTS="0 1" # Example: OrigCh0=Green, OrigCh1=Blue (Original 0-based TIFF channel indices)

# --- Live Update Parameters ---
ENABLE_LIVE_UPDATE="true" # "true" or "false"
LIVE_UPDATE_IMAGE_FILENAME="live_chunk_segmentation.png"
LIVE_UPDATE_IMAGE_PATH="${BASE_OUTPUT_DIR}/${LIVE_UPDATE_IMAGE_FILENAME}" # Full path for the live image
# NEW: Define path for the tile info file for the live scan viewer
TILE_INFO_FILENAME="current_tile_info.txt"
TILE_INFO_FILE_PATH="${BASE_OUTPUT_DIR}/${TILE_INFO_FILENAME}"


# --- Construct Python Command ---
CMD="python \"$PYTHON_SCRIPT\" \
    \"$IMAGE_FILE\" \
    \"$CSV_OUTPUT_BASE\""

CMD+=" --cell_model_path \"$CELL_MODEL_PATH\""
CMD+=" --nuclei_model_path \"$NUCLEI_MODEL_PATH\""
CMD+=" --chunk_size $CHUNK_Y $CHUNK_X"
CMD+=" --cells_diameter $CELLS_DIAMETER"
CMD+=" --cells_flow_threshold $CELLS_FLOW_THRESH"
CMD+=" --cells_cellprob_threshold $CELLS_CELLPROB_THRESH"
CMD+=" --cells_channels \"$CELLS_CHANNELS\""
CMD+=" --nuclei_diameter $NUCLEI_DIAMETER"
CMD+=" --nuclei_flow_threshold $NUCLEI_FLOW_THRESH"
CMD+=" --nuclei_cellprob_threshold $NUCLEI_CELLPROB_THRESH"
CMD+=" --nuclei_channels \"$NUCLEI_CHANNELS\""

if [ "$ENABLE_ADAPTIVE_NUCLEI" = "true" ]; then
    CMD+=" --enable_adaptive_nuclei"
    CMD+=" --nuclei_adaptive_cellprob_lower_limit $NUCLEI_ADAPTIVE_CELLPROB_LOWER_LIMIT"
    CMD+=" --nuclei_adaptive_cellprob_step_decrement $NUCLEI_ADAPTIVE_CELLPROB_STEP_DECREMENT"
    CMD+=" --nuclei_max_adaptive_attempts $NUCLEI_MAX_ADAPTIVE_ATTEMPTS"
    CMD+=" --adaptive_nuclei_trigger_ratio $ADAPTIVE_NUCLEI_TRIGGER_RATIO"
fi

if [ "$ENABLE_SUMMARY_VISUALIZATION" = "true" ]; then
    mkdir -p "$VIS_OUTPUT_DIR"
    CMD+=" --visualize"
    CMD+=" --visualize_output_dir \"$VIS_OUTPUT_DIR\""
    CMD+=" --num_visualize_chunks $NUM_SUMMARY_VIS_CHUNKS"
    CMD+=" --visualize_roi_size $VIS_ROI_Y $VIS_ROI_X"
    CMD+=" --vis_bg_channel_indices \"$VIS_BG_CHANNELS_FOR_PLOTS\""
fi

if [ "$ENABLE_LIVE_UPDATE" = "true" ]; then
    CMD+=" --live_update_image_path \"$LIVE_UPDATE_IMAGE_PATH\""
    # Add the tile_info_file_for_viewer argument if live update is enabled
    CMD+=" --tile_info_file_for_viewer \"$TILE_INFO_FILE_PATH\""

    # Ensure vis_bg_channel_indices is passed if live update is on,
    # even if summary visualization is off, as the live update uses it for its title.
    if [ "$ENABLE_SUMMARY_VISUALIZATION" != "true" ]; then
        CMD+=" --vis_bg_channel_indices \"$VIS_BG_CHANNELS_FOR_PLOTS\""
    fi
fi

# --- Run the Python Script ---
echo "--- Running Command ---"
echo "$CMD"
echo "-----------------------"
eval "$CMD"
EXIT_CODE=$?

# --- Final Check ---
echo "--- Job Summary ---"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script finished successfully."
    echo "Output CSV base: $CSV_OUTPUT_BASE"
    if [ "$ENABLE_SUMMARY_VISUALIZATION" = "true" ]; then
        echo "Summary visualizations: $VIS_OUTPUT_DIR"
    fi
    if [ "$ENABLE_LIVE_UPDATE" = "true" ]; then
        echo "Live update image path: $LIVE_UPDATE_IMAGE_PATH"
        echo "Tile info file for viewer: $TILE_INFO_FILE_PATH"
    fi
else
    echo "Python script failed with exit code $EXIT_CODE."
    echo "Check error log: /vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_${SLURM_JOB_ID}.err"
    echo "Check output log: /vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_live_${SLURM_JOB_ID}.out"
fi
echo "-------------------"
echo "Job complete."
exit $EXIT_CODE