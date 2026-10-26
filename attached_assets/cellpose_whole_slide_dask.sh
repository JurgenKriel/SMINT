#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100
#SBATCH --cpus-per-task 48
#SBATCH --mem=315G
#SBATCH --output=/path/to/project/cellpose_dask_%j.out
#SBATCH --error=/path/to/project/cellpose_dask_%j.err
#SBATCH --time=24:00:00       
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=you@example.org  

# Activate the environment
#source /path/to/project/cellpose_env/bin/activate


# Run the Python script
python /path/to/project/segmentation_pipeline/distributed_seg.py