#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100
#SBATCH --cpus-per-task 48
#SBATCH --mem=315G
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/cellpose_dask_%j.out
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/cellpose_dask_%j.err
#SBATCH --time=24:00:00       
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=kriel.j@wehi.edu.au  

# Activate the environment
#source /vast/projects/BCRL_Multi_Omics/cellpose_env/bin/activate


# Run the Python script
python /vast/projects/BCRL_Multi_Omics/segmentation_pipeline/distributed_seg.py