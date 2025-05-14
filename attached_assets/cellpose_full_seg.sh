#!/bin/bash

#SBATCH --job-name=cellpose_dual_wsi # Updated job name
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1        # Request 1 A100 GPU
#SBATCH --cpus-per-task=16       # CPUs for I/O, numpy etc.
#SBATCH --mem=128G               # Adjust based on chunk size and image size
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_%j.out # Log directory
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_%j.err  # Log directory
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
BASE_OUTPUT_DIR="/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/segmentation_output_dual_${SLURM_JOB_ID}"
OUTPUT_BASENAME="Venture5_WSI_DualSegmentation" # Base name for files within output dir
PYTHON_SCRIPT="/vast/projects/BCRL_Multi_Omics/segmentation_pipeline/adaptive_segmentation.py" # <--- UPDATE THIS PATH TO YOUR NEW SCRIPT

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"
echo "Output directory: $BASE_OUTPUT_DIR"

# Define output base path/prefix (script appends _cells.csv, _nuclei.csv, .log)
CSV_OUTPUT_BASE="${BASE_OUTPUT_DIR}/${OUTPUT_BASENAME}"
VIS_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_BASENAME}_visualizations" # Specific subdir for plots

# --- Cell Model Parameters ---
CELLS_DIAMETER=120.0
CELLS_FLOW_THRESH=0.4 # Adjusted default
CELLS_CELLPROB_THRESH=-1.5 # Adjusted default
# IMPORTANT: Set channels based on cell model needs. Format: "1 2" (Cytoplasm=1, Nucleus=2)
CELLS_CHANNELS="1 2" # Example: Assumes cell model uses Cyto (Ch1) and Nuc (Ch2)

# --- Nuclei Model Parameters ---
NUCLEI_DIAMETER=60.0
NUCLEI_FLOW_THRESH=0.4 # Adjusted default
NUCLEI_CELLPROB_THRESH=-1.5 # Adjusted default
# IMPORTANT: Set channels based on nuclei model needs. Format: "0 0" (Grayscale) or "2 0" (Use Ch2 only)
NUCLEI_CHANNELS="2 1" # Example: Assumes nuclei model runs on single channel (will use channel 0 from its input)
# If nuclei model needs channel 2 (index 1) from original TIFF, use "2 0"

# --- Processing Parameters ---
CHUNK_Y=2048
CHUNK_X=2048

# --- Visualization Parameters ---
ENABLE_VISUALIZATION="true" # Set to "true" to enable, "false" to disable
NUM_VIS_CHUNKS=10
VIS_ROI_Y=2024
VIS_ROI_X=2024
# IMPORTANT: Set background channel indices based on your OME-TIFF. Format "0 1" etc.
# These are 0-based indices from the ORIGINAL TIFF file.
VIS_BG_CHANNELS="0 1" # Example: Channel 0=Green, Channel 1=Blue

# --- Construct Python Command ---
# Start with the basic command and positional arguments
CMD="python $PYTHON_SCRIPT \
    \"$IMAGE_FILE\" \
    \"$CSV_OUTPUT_BASE\"" # Pass the base path/prefix

# Add model paths
CMD+=" --cell_model_path \"$CELL_MODEL_PATH\""
CMD+=" --nuclei_model_path \"$NUCLEI_MODEL_PATH\""

# Add processing arguments
CMD+=" --chunk_size $CHUNK_Y $CHUNK_X"

# Add cell model arguments
CMD+=" --cells_diameter $CELLS_DIAMETER"
CMD+=" --cells_flow_threshold $CELLS_FLOW_THRESH"
CMD+=" --cells_cellprob_threshold $CELLS_CELLPROB_THRESH"
CMD+=" --cells_channels \"$CELLS_CHANNELS\"" # Pass as space-separated string

# Add nuclei model arguments
CMD+=" --nuclei_diameter $NUCLEI_DIAMETER"
CMD+=" --nuclei_flow_threshold $NUCLEI_FLOW_THRESH"
CMD+=" --nuclei_cellprob_threshold $NUCLEI_CELLPROB_THRESH"
CMD+=" --nuclei_channels \"$NUCLEI_CHANNELS\"" # Pass as space-separated string

# Add optional visualization arguments
if [ "$ENABLE_VISUALIZATION" = "true" ]; then
    mkdir -p "$VIS_OUTPUT_DIR" # Create vis dir only if needed
    CMD+=" --visualize"
    CMD+=" --visualize_output_dir \"$VIS_OUTPUT_DIR\""
    CMD+=" --num_visualize_chunks $NUM_VIS_CHUNKS"
    CMD+=" --visualize_roi_size $VIS_ROI_Y $VIS_ROI_X"
    CMD+=" --vis_bg_channel_indices \"$VIS_BG_CHANNELS\"" # Pass as space-separated string
fi

# --- Run the Python Script ---
echo "--- Running Command ---"
echo "$CMD"
echo "-----------------------"

eval $CMD # Use eval to handle spaces in paths correctly

EXIT_CODE=$? # Capture exit code of the python script

# --- Final Check ---
echo "--- Job Summary ---"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script finished successfully."
    echo "Output Cell CSV: ${CSV_OUTPUT_BASE}_cells.csv"
    echo "Output Nuclei CSV: ${CSV_OUTPUT_BASE}_nuclei.csv"
    if [ "$ENABLE_VISUALIZATION" = "true" ]; then
        echo "Visualizations: $VIS_OUTPUT_DIR"
    fi
    echo "Log file: ${CSV_OUTPUT_BASE}.log"
else
    echo "Python script failed with exit code $EXIT_CODE."
    echo "Check SLURM error file: /vast/projects/BCRL_Multi_Omics/slurm_logs/cellpose_dual_wsi_${SLURM_JOB_ID}.err"
    echo "Check Python log file (if created): ${CSV_OUTPUT_BASE}.log"

fi
echo "-------------------"

# Deactivate environment (optional but good practice)
# deactivate # If using venv
# conda deactivate # If using conda

echo "Job complete."
exit $EXIT_CODE # Exit SLURM job with the python script's exit code