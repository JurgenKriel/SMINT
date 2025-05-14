"""
R integration bridge for SMINT.

This module provides functions for running R scripts and for
transferring data between Python and R.
"""

import os
import logging
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Check if rpy2 is available
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    import rpy2.robjects.packages as rpackages
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    ro = None
    pandas2ri = None
    localconverter = None
    rpackages = None
    logging.warning("rpy2 package not available. R integration functionality will be limited.")

def initialize_r(packages=None):
    """
    Initialize the R environment with necessary packages.
    
    Args:
        packages (list, optional): List of R packages to load.
    
    Returns:
        bool: True if initialization successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return False
    
    # Initialize R
    try:
        # Activate pandas converter
        pandas2ri.activate()
        
        # Install and load required packages
        if packages:
            utils = rpackages.importr('utils')
            base = rpackages.importr('base')
            
            # Get installed packages
            installed_packages = ro.r('installed.packages()[,"Package"]')
            
            # Install missing packages
            for package in packages:
                if package not in installed_packages:
                    logger.info(f"Installing R package: {package}")
                    utils.install_packages(package)
                
                # Load the package
                try:
                    rpackages.importr(package)
                    logger.info(f"Loaded R package: {package}")
                except Exception as e:
                    logger.error(f"Error loading R package {package}: {e}")
                    return False
        
        logger.info("R environment initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing R environment: {e}")
        return False

def run_r_script(script_path, args=None, return_output=False):
    """
    Run an R script using the system's R interpreter.
    
    Args:
        script_path (str): Path to the R script.
        args (list, optional): Command-line arguments to pass to the script.
        return_output (bool): Whether to return stdout/stderr.
    
    Returns:
        int or tuple: Return code or (return_code, stdout, stderr).
    """
    logger = logging.getLogger(__name__)
    
    # Check if script exists
    if not os.path.exists(script_path):
        logger.error(f"R script not found: {script_path}")
        return 1 if not return_output else (1, "", "Script not found")
    
    # Prepare command
    cmd = ["Rscript", script_path]
    if args:
        cmd.extend(args)
    
    # Run the script
    logger.info(f"Running R script: {' '.join(cmd)}")
    try:
        if return_output:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            logger.info(f"R script completed with return code {result.returncode}")
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=False)
            logger.info(f"R script completed with return code {result.returncode}")
            return result.returncode
    
    except Exception as e:
        logger.error(f"Error running R script: {e}")
        return 1 if not return_output else (1, "", str(e))

def r_to_pandas(r_object):
    """
    Convert an R object to pandas DataFrame.
    
    Args:
        r_object: An R object.
    
    Returns:
        pandas.DataFrame: Pandas DataFrame.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return None
    
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_df = ro.conversion.rpy2py(r_object)
        return pd_df
    
    except Exception as e:
        logger.error(f"Error converting R object to pandas DataFrame: {e}")
        return None

def pandas_to_r(pd_df):
    """
    Convert a pandas DataFrame to an R data.frame.
    
    Args:
        pd_df (pandas.DataFrame): Pandas DataFrame.
    
    Returns:
        rpy2.robjects.DataFrame: R data.frame.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return None
    
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(pd_df)
        return r_df
    
    except Exception as e:
        logger.error(f"Error converting pandas DataFrame to R data.frame: {e}")
        return None

def exec_r_code(r_code, input_data=None, output_vars=None):
    """
    Execute R code and optionally return specified variables.
    
    Args:
        r_code (str): R code to execute.
        input_data (dict, optional): Dictionary mapping variable names to pandas DataFrames.
        output_vars (list, optional): List of variable names to return.
    
    Returns:
        dict: Dictionary mapping variable names to pandas DataFrames.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return None
    
    try:
        # Convert input data to R
        if input_data:
            for var_name, pd_df in input_data.items():
                r_df = pandas_to_r(pd_df)
                ro.globalenv[var_name] = r_df
        
        # Run the R code
        ro.r(r_code)
        
        # Convert output variables to pandas
        result = {}
        if output_vars:
            for var_name in output_vars:
                if var_name in ro.globalenv:
                    r_obj = ro.globalenv[var_name]
                    pd_obj = r_to_pandas(r_obj)
                    result[var_name] = pd_obj
                else:
                    logger.warning(f"Variable '{var_name}' not found in R environment")
        
        return result
    
    except Exception as e:
        logger.error(f"Error executing R code: {e}")
        return None

def r_function_to_python(r_function_name, *args, **kwargs):
    """
    Call an R function with Python arguments and convert the result to Python.
    
    Args:
        r_function_name (str): Name of the R function.
        *args: Positional arguments to pass to the R function.
        **kwargs: Keyword arguments to pass to the R function.
    
    Returns:
        The result of the R function converted to Python objects.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return None
    
    try:
        # Get the R function
        r_function = ro.r[r_function_name]
        
        # Convert Python arguments to R
        r_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                r_args.append(pandas_to_r(arg))
            elif isinstance(arg, (list, tuple)):
                r_args.append(ro.vectors.StrVector(arg) if all(isinstance(x, str) for x in arg) else 
                             ro.vectors.FloatVector(arg) if all(isinstance(x, (int, float)) for x in arg) else
                             arg)
            else:
                r_args.append(arg)
        
        # Convert Python keyword arguments to R
        r_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                r_kwargs[key] = pandas_to_r(value)
            elif isinstance(value, (list, tuple)):
                if all(isinstance(x, str) for x in value):
                    r_kwargs[key] = ro.vectors.StrVector(value)
                elif all(isinstance(x, (int, float)) for x in value):
                    r_kwargs[key] = ro.vectors.FloatVector(value)
                else:
                    r_kwargs[key] = value
            else:
                r_kwargs[key] = value
        
        # Call the R function
        result = r_function(*r_args, **r_kwargs)
        
        # Convert the result to Python
        if hasattr(result, "rclass") and "data.frame" in result.rclass:
            return r_to_pandas(result)
        else:
            # Try to convert other types
            try:
                return ro.conversion.rpy2py(result)
            except Exception:
                return result
    
    except Exception as e:
        logger.error(f"Error calling R function '{r_function_name}': {e}")
        return None

def load_r_analysis_script(script_path):
    """
    Load and execute an R analysis script that defines functions.
    
    Args:
        script_path (str): Path to the R script.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    if not HAS_RPY2:
        logger.error("rpy2 is not installed. Install with 'pip install rpy2'")
        return False
    
    try:
        # Convert script path to absolute path
        script_path = os.path.abspath(script_path)
        
        # Source the R script
        ro.r(f'source("{script_path}")')
        
        logger.info(f"Loaded R script: {script_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error loading R script {script_path}: {e}")
        return False
