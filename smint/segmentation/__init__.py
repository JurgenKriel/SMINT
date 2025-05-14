"""
Segmentation module for SMINT.

This module provides tools for cell and nuclear segmentation from whole slide images,
with multi-GPU and distributed computing support.
"""

from .wsi_cell_segmentation import process_large_image
from .distributed_seg import run_distributed_segmentation
from .cell_utils import get_cell_outlines, segment_chunk
from .postprocess import extract_contours, save_masks

__all__ = [
    'process_large_image',
    'run_distributed_segmentation',
    'get_cell_outlines',
    'segment_chunk',
    'extract_contours',
    'save_masks',
]
