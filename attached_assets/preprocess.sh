#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=regular    
#SBATCH --cpus-per-task=56
#SBATCH --mem=256G
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/preprocess_%j.out
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/preprocess_%j.err
#SBATCH --time=24:00:00         # Added time limit (24 hours)
#SBATCH --mail-type=BEGIN,END,FAIL  # Optional: email notifications
#SBATCH --mail-user=kriel.j@wehi.edu.au  # Optional: replace with your email

# Load any required modules (if needed)
# module load python

# Force immediate output flushing
export PYTHONUNBUFFERED=1

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CPU cores allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

# Run the Python script
python /vast/projects/BCRL_Multi_Omics/segmentation_pipeline/preprocess_split_ometiff.py

# Print completion message
echo "Job completed at: $(date)"