"""
Alignment module for SMINT.

This module provides tools for aligning different omics data types
and for registering images using ST Align.
"""

from .st_align_wrapper import align_spatial_transcriptomics, load_alignment, save_alignment

# Import Xenium-Metabolomics alignment functionality
try:
    from .xenium_metabolomics import (
        align_xenium_to_metabolomics,
        read_xenium_data,
        read_sm_matrix,
        visualize_alignment
    )
    XENIUM_METABOLOMICS_AVAILABLE = True
except ImportError:
    # STalign or other dependencies might be missing
    XENIUM_METABOLOMICS_AVAILABLE = False

# Only include available functionalities in __all__
__all__ = [
    'align_spatial_transcriptomics',
    'load_alignment',
    'save_alignment'
]

# Add Xenium-Metabolomics alignment if available
if XENIUM_METABOLOMICS_AVAILABLE:
    __all__.extend([
        'align_xenium_to_metabolomics',
        'read_xenium_data',
        'read_sm_matrix',
        'visualize_alignment'
    ])
