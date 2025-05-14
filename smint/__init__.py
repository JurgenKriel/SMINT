"""
SMINT - Spatial Multi-Omics Integration

A Python package for Spatial Multi-Omics Integration with enhanced segmentation capabilities
and streamlined workflow.
"""

__version__ = '0.1.0'

# Import sub-packages to make them available at the top level
from . import segmentation
from . import preprocessing
from . import visualization
from . import utils
from . import alignment
from . import r_integration

__all__ = [
    'segmentation',
    'preprocessing',
    'visualization',
    'utils',
    'alignment',
    'r_integration',
]
