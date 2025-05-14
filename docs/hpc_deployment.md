# HPC Deployment

SMINT is designed to run efficiently on High-Performance Computing (HPC) clusters, taking advantage of multiple CPUs and GPUs for processing large spatial omics datasets. This page explains how to deploy SMINT on SLURM-based HPC systems.

## Overview

Running SMINT on an HPC cluster involves the following steps:

1. Install SMINT on the HPC system
2. Prepare your configuration file
3. Submit jobs using SLURM submission scripts
4. Monitor job progress
5. Collect and analyze results

## Installation on HPC

### Using Environment Modules

Most HPC systems use environment modules. Load the required modules and install SMINT:

```bash
# Load required modules
module load python/3.8
module load cuda/11.3
module load miniconda3

# Create a conda environment
conda create -n smint python=3.8
conda activate smint

# Install SMINT
pip install smint
