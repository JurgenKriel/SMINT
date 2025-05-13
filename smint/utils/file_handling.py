"""
File handling utilities for SMINT.

This module provides functions for managing files and directories,
finding files with specific patterns, and extracting file information.
"""

import os
import glob
import logging
from pathlib import Path
import shutil
import re

def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str or Path): Directory path.
    
    Returns:
        Path: Path object for the directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def find_files(directory, pattern="*", recursive=False):
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory (str or Path): Directory to search in.
        pattern (str): Glob pattern to match files.
        recursive (bool): Whether to search recursively.
    
    Returns:
        list: List of Path objects for matching files.
    """
    directory = Path(directory)
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return []
    
    # Construct the glob pattern
    search_pattern = str(directory / pattern)
    
    # Find matching files
    if recursive:
        matches = glob.glob(search_pattern, recursive=True)
    else:
        matches = glob.glob(search_pattern)
    
    return [Path(match) for match in matches]

def get_file_info(file_path):
    """
    Get information about a file.
    
    Args:
        file_path (str or Path): Path to the file.
    
    Returns:
        dict: Dictionary with file information.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"File does not exist: {file_path}")
        return {}
    
    stats = file_path.stat()
    
    return {
        'path': str(file_path),
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'size': stats.st_size,
        'modified': stats.st_mtime,
        'directory': str(file_path.parent),
    }

def safe_move(src, dst, overwrite=False):
    """
    Safely move a file from source to destination.
    
    Args:
        src (str or Path): Source file path.
        dst (str or Path): Destination file path.
        overwrite (bool): Whether to overwrite existing destination.
    
    Returns:
        bool: True if moved successfully, False otherwise.
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        logging.error(f"Source file does not exist: {src}")
        return False
    
    if dst.exists() and not overwrite:
        logging.warning(f"Destination file exists and overwrite is False: {dst}")
        return False
    
    try:
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(src), str(dst))
        return True
    
    except Exception as e:
        logging.error(f"Error moving file from {src} to {dst}: {e}")
        return False

def extract_tile_coordinates(filename):
    """
    Extract tile coordinates from a filename.
    
    Args:
        filename (str): Filename containing tile coordinates.
    
    Returns:
        tuple or None: (y, x) coordinates if found, None otherwise.
    """
    # Common patterns for tile coordinates in filenames
    patterns = [
        r'_y(\d+)_x(\d+)',  # Format: _y123_x456
        r'_(\d+)_(\d+)',    # Format: _123_456
        r'tile_(\d+)_(\d+)'  # Format: tile_123_456
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                y, x = map(int, match.groups())
                return (y, x)
            except ValueError:
                continue
    
    return None

def get_file_size_human_readable(size_in_bytes):
    """
    Convert file size in bytes to human-readable format.
    
    Args:
        size_in_bytes (int): File size in bytes.
    
    Returns:
        str: Human-readable file size.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            break
        size_in_bytes /= 1024.0
    
    return f"{size_in_bytes:.2f} {unit}"

def organize_files_by_extension(directory, move=False, target_dir=None):
    """
    Organize files in a directory by their extensions.
    
    Args:
        directory (str or Path): Directory containing files.
        move (bool): Whether to move files to subdirectories.
        target_dir (str or Path, optional): Target directory for organized files.
    
    Returns:
        dict: Dictionary mapping extensions to lists of files.
    """
    directory = Path(directory)
    if not directory.exists():
        logging.error(f"Directory does not exist: {directory}")
        return {}
    
    # Group files by extension
    files_by_extension = {}
    
    for file in directory.glob("*"):
        if file.is_file():
            ext = file.suffix.lower()[1:]  # Get extension without dot
            if not ext:
                ext = "no_extension"
            
            if ext not in files_by_extension:
                files_by_extension[ext] = []
            
            files_by_extension[ext].append(file)
    
    # Move files to subdirectories if requested
    if move:
        target_dir = Path(target_dir) if target_dir else directory
        
        for ext, files in files_by_extension.items():
            ext_dir = target_dir / ext
            ext_dir.mkdir(exist_ok=True)
            
            for file in files:
                dst = ext_dir / file.name
                if not dst.exists():
                    try:
                        shutil.move(str(file), str(dst))
                    except Exception as e:
                        logging.error(f"Error moving {file} to {dst}: {e}")
    
    return files_by_extension
