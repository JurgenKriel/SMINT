# Utilities API

The utilities module provides helper functions for file operations, logging, and other common tasks.

## Overview

The utilities module includes the following key functionalities:

- **list_files**: List files in a directory with optional filtering
- **create_directory**: Create a directory if it doesn't exist
- **get_file_info**: Get information about a file
- **setup_logger**: Set up a logger for consistent logging
- **log_parameters**: Log parameters for reproducibility
- **load_config**: Load configuration from a file
- **save_config**: Save configuration to a file

## Function Reference

```python
def list_files(directory, pattern=None, recursive=False):
    """
    List files in a directory with optional filtering.
    
    Parameters
    ----------
    directory : str
        Directory to list files from
    pattern : str, optional
        Glob pattern to filter files
    recursive : bool, optional
        Whether to search recursively
        
    Returns
    -------
    list
        List of file paths
    """
    pass

def create_directory(directory, parents=True, exist_ok=True):
    """
    Create a directory if it doesn't exist.
    
    Parameters
    ----------
    directory : str
        Directory to create
    parents : bool, optional
        Whether to create parent directories
    exist_ok : bool, optional
        Whether to ignore if directory exists
        
    Returns
    -------
    str
        Path to created directory
    """
    pass

def get_file_info(file_path):
    """
    Get information about a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file
        
    Returns
    -------
    dict
        Dictionary with file information
    """
    pass

def setup_logger(name, log_file=None, level=logging.INFO, format=None):
    """
    Set up a logger for consistent logging.
    
    Parameters
    ----------
    name : str
        Name of the logger
    log_file : str, optional
        Path to log file
    level : int, optional
        Logging level
    format : str, optional
        Log message format
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    pass

def log_parameters(logger, parameters, prefix=''):
    """
    Log parameters for reproducibility.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger to use
    parameters : dict
        Parameters to log
    prefix : str, optional
        Prefix for parameter names
        
    Returns
    -------
    None
    """
    pass

def load_config(config_path, format=None):
    """
    Load configuration from a file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    format : str, optional
        Format of the file (yaml, json, etc.)
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    pass

def save_config(config, config_path, format=None):
    """
    Save configuration to a file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str
        Path to save configuration to
    format : str, optional
        Format to save in (yaml, json, etc.)
        
    Returns
    -------
    str
        Path to saved configuration
    """
    pass
```

## Dependency Handling

The utilities module is designed to have minimal external dependencies, making it more robust across different environments. The core functionality requires only standard Python libraries, with some enhanced features available when optional dependencies are present.

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| PyYAML | Advanced config loading/saving | Fallback to JSON for configuration |
| tqdm | Progress tracking | Simple text-based progress updates |
| pandas | Data manipulation | Limited data handling capabilities |

The utilities module is designed to be a reliable foundation for the rest of the SMINT package, with minimal external dependencies to ensure robustness.
