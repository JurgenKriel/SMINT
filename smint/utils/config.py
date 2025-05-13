"""
Configuration utilities for SMINT.

This module provides functions for loading and saving configuration files
in YAML format.
"""

import yaml
import os
import logging
from pathlib import Path

def get_default_config_path():
    """
    Get the default path for the configuration file.
    
    Returns:
        Path: Path to the default configuration file.
    """
    # Look for config in the standard locations
    possible_paths = [
        # Current directory
        Path("./configs/segmentation_config.yaml"),
        # User's home directory
        Path.home() / ".smint" / "config.yaml",
        # Package directory
        Path(__file__).parent.parent.parent / "configs" / "segmentation_config.yaml",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Return the package config path even if it doesn't exist (for writing)
    return possible_paths[2]

def load_config(config_path=None):
    """
    Load a configuration file.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, searches in default locations.
    
    Returns:
        dict: Configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    
    # If no path provided, use default
    if config_path is None:
        config_path = get_default_config_path()
    
    # Convert to Path object
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    # Read and parse the configuration file
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config or {}
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}

def save_config(config, config_path=None):
    """
    Save a configuration to a file.
    
    Args:
        config (dict): Configuration dictionary.
        config_path (str, optional): Path to save the configuration file.
            If None, uses the default location.
    
    Returns:
        bool: True if successfully saved, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    # If no path provided, use default
    if config_path is None:
        config_path = get_default_config_path()
    
    # Convert to Path object
    config_path = Path(config_path)
    
    # Ensure the directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the configuration file
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False

def merge_configs(base_config, override_config):
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config (dict): Base configuration dictionary.
        override_config (dict): Configuration dictionary with overrides.
    
    Returns:
        dict: Merged configuration dictionary.
    """
    # Start with a copy of the base config
    result = base_config.copy()
    
    # Recursively update with the override config
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = recursive_update(d[k], v)
            else:
                d[k] = v
        return d
    
    return recursive_update(result, override_config)

def validate_config(config, schema=None):
    """
    Validate a configuration against a schema.
    
    Args:
        config (dict): Configuration dictionary to validate.
        schema (dict, optional): Schema dictionary. If None, uses a basic schema.
    
    Returns:
        tuple: (is_valid, errors) where is_valid is a boolean and errors is a list of error messages.
    """
    logger = logging.getLogger(__name__)
    
    # Basic schema if none provided
    if schema is None:
        schema = {
            'required': [
                'model_paths',
                'output_dir'
            ],
            'optional': [
                'model_params',
                'chunk_size',
                'preprocessing',
                'visualization',
                'tile_info_path',
                'live_update_image_path'
            ]
        }
    
    errors = []
    
    # Check required fields
    for required_field in schema.get('required', []):
        if required_field not in config:
            errors.append(f"Missing required field: {required_field}")
    
    # Check field types (can be expanded with more specific type checking)
    if 'model_paths' in config and not isinstance(config['model_paths'], list):
        errors.append("'model_paths' must be a list")
    
    if 'chunk_size' in config and not isinstance(config['chunk_size'], list):
        errors.append("'chunk_size' must be a list")
    
    # Log the validation results
    if errors:
        logger.warning(f"Configuration validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("Configuration validated successfully")
    
    return len(errors) == 0, errors

def get_config_value(config, key_path, default=None):
    """
    Get a value from a nested configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary.
        key_path (str): Path to the key, using dots to separate levels (e.g., 'model_params.diameter').
        default: Default value to return if the key is not found.
    
    Returns:
        The value at the specified key path, or the default if not found.
    """
    keys = key_path.split('.')
    result = config
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    
    return result
