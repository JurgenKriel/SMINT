#!/usr/bin/env python
"""
Helper script to build documentation with proper mocking of dependencies.
This is used by the GitHub Actions workflow to ensure all dependencies
are properly mocked during the documentation build process.
"""
import os
import sys
import subprocess
from unittest.mock import MagicMock

# List of modules to mock
MOCK_MODULES = [
    'rpy2', 'rpy2.robjects', 'rpy2.robjects.packages', 'rpy2.robjects.conversion',
    'cellpose', 'cv2', 'dask.distributed', 'dask_cuda', 'dask.array',
    'distributed', 'distributed.client', 'dask', 'numpy', 'numpy.core',
    'matplotlib', 'pandas', 'sklearn', 'skimage', 'scipy', 'tifffile'
]

def mock_modules():
    """Mock all required modules for documentation generation."""
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()
        print(f"Mocked module: {mod_name}")

def main():
    """Run mkdocs build or deploy command with all dependencies mocked."""
    # Set environment variables
    os.environ['DOCS_BUILDING'] = '1'
    
    # Mock all required modules
    mock_modules()
    
    # Determine command to run (build or deploy)
    command = "mkdocs gh-deploy --force" if "--deploy" in sys.argv else "mkdocs build"
    
    # Run the command
    print(f"Running command: {command}")
    return subprocess.call(command, shell=True)

if __name__ == "__main__":
    sys.exit(main())
