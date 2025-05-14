# SMINT: Spatial Multi-Omics Integration

SMINT is a Python package for Spatial Multi-Omics Integration with enhanced segmentation capabilities and streamlined workflow.

## Overview

SMINT provides a comprehensive toolkit for processing and analyzing spatial omics data, including:

- Multi-GPU cell segmentation for whole-slide images
- Distributed segmentation using Dask for improved performance
- Live segmentation monitoring with intuitive visualization tools
- Streamlined alignment workflow using ST Align
- Integration with R analysis scripts
- Comprehensive documentation with step-by-step guides
- HPC deployment scripts for SLURM-based clusters

![SMINT Workflow](SpatialSegPaper_v2.png)

## Key Features

### Enhanced Segmentation

- **Multi-GPU Support**: Utilize multiple GPUs for faster processing of large whole-slide images
- **Distributed Computing**: Use Dask to distribute segmentation tasks across multiple nodes
- **Live Monitoring**: Track segmentation progress in real-time with the built-in viewer
- **Adaptive Segmentation**: Automatically adjust segmentation parameters for optimal results
- **Dual-Model Segmentation**: Simultaneously segment cells and nuclei with specialized models

### Streamlined Alignment

- **ST Align Integration**: Seamlessly align spatial transcriptomics data with the ST Align tool
- **Multiple Transformation Types**: Support for affine, rigid, similarity, and projective transformations
- **Multiple Data Types**: Compatible with Visium, Slide-seq, and custom spatial data formats

### R Integration

- **Seamless Python-R Bridge**: Call R scripts and functions directly from Python
- **Data Transfer**: Convert data between Python and R formats
- **Existing R Scripts**: Use your existing R analysis scripts within the SMINT workflow

### Visualization

- **Live Viewer**: Monitor segmentation progress with a live viewer
- **Segmentation Overlays**: Visualize segmentation results overlaid on the original image
- **Feature Plots**: Generate feature plots and spatial heatmaps

### HPC Deployment

- **SLURM Integration**: Ready-to-use SLURM submission scripts for HPC deployment
- **Resource Management**: Optimized resource allocation for different processing stages
- **Checkpointing**: Resume processing from checkpoints after interruptions

## Getting Started

- [Installation](installation.md): Install SMINT and its dependencies
- [Segmentation](segmentation.md): Run cell segmentation on whole-slide images
- [Alignment](alignment.md): Align spatial transcriptomics data
- [R Integration](r_integration.md): Use R scripts and functions with SMINT
- [Examples](examples.md): Complete examples of SMINT workflows
- [Configuration](configuration.md): Configure SMINT for your specific needs
- [HPC Deployment](hpc_deployment.md): Run SMINT on HPC clusters

## Citation

If you use SMINT in your research, please cite:

