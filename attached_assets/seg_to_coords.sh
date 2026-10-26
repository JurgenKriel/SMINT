#!/bin/bash
#SBATCH --job-name=cellpose
#SBATCH --partition=regular
#SBATCH --cpus-per-task 64
#SBATCH --mem=128G
#SBATCH --output=/path/to/project/seg_to_coords_%j.out
#SBATCH --error=/path/to/project/seg_to_coords_%j.err

python /path/to/project/mask_to_coords.py