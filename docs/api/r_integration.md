# R Integration API

The R integration module provides functions for calling R scripts and functions from Python and for transferring data between Python and R.

## Overview

The R integration module includes the following key functionalities:

- **run_r_script**: Run an R script from Python
- **initialize_r**: Initialize the R environment
- **r_to_pandas**: Convert R objects to pandas DataFrames
- **pandas_to_r**: Convert pandas DataFrames to R objects
- **run_r_function**: Call R functions from Python

## Function Reference

```python
def run_r_script(script_path, args=None, return_output=False):
    """
    Run an R script from Python.
    
    Parameters
    ----------
    script_path : str
        Path to the R script
    args : list, optional
        Command line arguments to pass to the R script
    return_output : bool, optional
        Whether to return the script's output
        
    Returns
    -------
    str or None
        Output of the R script if return_output is True
    """
    pass

def initialize_r(packages=None):
    """
    Initialize the R environment.
    
    Parameters
    ----------
    packages : list, optional
        List of R packages to load
        
    Returns
    -------
    bool
        Whether initialization was successful
    """
    pass

def r_to_pandas(r_object):
    """
    Convert R objects to pandas DataFrames.
    
    Parameters
    ----------
    r_object : rpy2.robjects.vectors.DataFrame or str
        R DataFrame object or name of R object
        
    Returns
    -------
    pandas.DataFrame
        Pandas DataFrame
    """
    pass

def pandas_to_r(df, r_variable_name=None):
    """
    Convert pandas DataFrames to R objects.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame to convert
    r_variable_name : str, optional
        Name to assign to the R object
        
    Returns
    -------
    rpy2.robjects.vectors.DataFrame
        R DataFrame object
    """
    pass

def run_r_function(function_name, *args, **kwargs):
    """
    Call R functions from Python.
    
    Parameters
    ----------
    function_name : str
        Name of the R function to call
    *args : tuple
        Positional arguments to pass to the R function
    **kwargs : dict
        Keyword arguments to pass to the R function
        
    Returns
    -------
    object
        Result of the R function call
    """
    pass
```

## Dependency Handling

The R integration module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | Direct R integration | Fallback to subprocess R script execution |
| pandas | Data manipulation | Required dependency |
| numpy | Matrix operations | Required dependency |
| R | R script execution | Required for any R functionality |

When rpy2 is missing, SMINT will still allow you to run R scripts through subprocess calls, but the direct function calling and data transfer capabilities will be limited.
