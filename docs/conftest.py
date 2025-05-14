"""
This file helps with mocking imports during documentation generation.
"""
import sys
from unittest.mock import MagicMock

# List of modules to mock
MOCK_MODULES = [
    'rpy2', 'rpy2.robjects', 'rpy2.robjects.packages', 'rpy2.robjects.conversion',
    'cellpose', 'cv2', 'dask.distributed', 'dask_cuda', 'dask.array',
    'distributed', 'distributed.client'
]

# Create mock for each module
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()