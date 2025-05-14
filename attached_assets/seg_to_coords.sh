#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --partition=regular
#SBATCH --cpus-per-task 64
#SBATCH --mem=128G
#SBATCH --output=/vast/projects/BCRL_Multi_Omics/seg_to_coords_%j.out
#SBATCH --error=/vast/projects/BCRL_Multi_Omics/seg_to_coords_%j.err

python /vast/projects/BCRL_Multi_Omics/mask_to_coords.py