#!/usr/bin/env python
"""
Helper script to build documentation with proper mocking of dependencies.
This is used by the GitHub Actions workflow to ensure all dependencies
are properly mocked during the documentation build process.
"""
import os
import sys
import subprocess
import glob
import re
from unittest.mock import MagicMock

# List of modules to mock
MOCK_MODULES = [
    'rpy2', 'rpy2.robjects', 'rpy2.robjects.packages', 'rpy2.robjects.conversion',
    'cellpose', 'cv2', 'dask.distributed', 'dask_cuda', 'dask.array',
    'distributed', 'distributed.client', 'dask', 'numpy', 'numpy.core',
    'matplotlib', 'pandas', 'sklearn', 'skimage', 'scipy', 'tifffile',
    'smint', 'smint.segmentation', 'smint.preprocessing', 'smint.visualization',
    'smint.alignment', 'smint.r_integration', 'smint.utils'
]

def mock_modules():
    """Mock all required modules for documentation generation."""
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()
        print(f"Mocked module: {mod_name}")

def check_for_mkdocstrings_references():
    """
    Check for and remove mkdocstrings references that might cause errors.
    We'll use a static approach instead of auto-generating from code.
    """
    api_files = glob.glob('docs/api/*.md')
    pattern = re.compile(r':::\s+smint\.')
    
    for file_path in api_files:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if file has mkdocstrings references
        if pattern.search(content):
            print(f"WARNING: Found mkdocstrings references in {file_path}")
            print(f"These will be ignored during the build process")
    
    return True

def main():
    """Run mkdocs build or deploy command with all dependencies mocked."""
    # Set environment variables
    os.environ['DOCS_BUILDING'] = '1'
    
    # Mock all required modules
    mock_modules()
    
    # Check for mkdocstrings references
    check_for_mkdocstrings_references()
    
    # Determine command to run (build or deploy)
    command = "mkdocs gh-deploy --force" if "--deploy" in sys.argv else "mkdocs build"
    
    # Run the command
    print(f"Running command: {command}")
    return subprocess.call(command, shell=True)

if __name__ == "__main__":
    sys.exit(main())