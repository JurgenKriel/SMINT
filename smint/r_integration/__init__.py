"""
R integration module for SMINT.

This module provides tools for integrating R scripts and functions
with the Python-based SMINT workflow.
"""

from .r_bridge import run_r_script, r_to_pandas, pandas_to_r, initialize_r

__all__ = [
    'run_r_script',
    'r_to_pandas',
    'pandas_to_r',
    'initialize_r'
]
