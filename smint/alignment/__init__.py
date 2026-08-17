"""
Alignment module for SMINT.

This module provides tools for aligning different omics data types
and for registering images using ST Align.
"""

from .st_align_wrapper import align_spatial_transcriptomics, load_alignment, save_alignment

# ST <-> SM registration (STalign LDDMM, with a manual landmark stage).
# This is the supported entry point for registering spatial metabolomics onto
# spatial transcriptomics; see smint.alignment.st_sm_registration.
from .st_sm_registration import (
    prepare_landmark_inputs,
    register_sm_to_st,
    read_sm_matrix,
    read_st_annotations,
    sm_coordinates,
    st_coordinates,
    rasterize_coordinates,
    save_rasterized,
    load_landmarks,
    check_landmark_pair,
    affine_from_landmarks,
    run_lddmm_alignment,
    transform_points,
    save_transformed_data,
    find_sm_matrix_files,
    stalign_available,
)

# Coarse pre-registration: scale to a reference coordinate system, plus
# rotation and flips, so datasets start roughly aligned before STalign runs.
from .pretransform import (
    build_pretransform,
    apply_pretransform,
    fit_scale_to_reference,
    describe_pretransform,
    overlap_score,
    compose,
    rotation_matrix,
    flip_matrix,
    scale_matrix,
    translation_matrix,
)

# Centroid-to-centroid registration for post-staining on the SAME section,
# where centroids correspond one-to-one (RANSAC / affine / TPS, scored by TRE).
# For sequential sections use the STalign LDDMM path above instead.
from .centroid_registration import (
    match_centroids,
    split_pairs,
    target_registration_error,
    estimate_affine,
    estimate_affine_ransac,
    apply_affine,
    fit_tps,
    apply_tps,
    register_centroids,
    register_centroid_files,
)

# Deprecated Xenium-Metabolomics module, superseded by st_sm_registration.
# Note `read_sm_matrix` is intentionally NOT re-exported here: the newer one
# above returns a DataFrame and applies the scale/rotate/prefix steps the
# registration workflow depends on.
try:
    from .xenium_metabolomics import (
        align_xenium_to_metabolomics,
        read_xenium_data,
        visualize_alignment
    )
    XENIUM_METABOLOMICS_AVAILABLE = True
except ImportError:
    # STalign or other dependencies might be missing
    XENIUM_METABOLOMICS_AVAILABLE = False

# Backward compatibility functions
def run_alignment(source_data, target_data, method="similarity", output_dir=None, **kwargs):
    """
    Backward compatibility function for align_spatial_transcriptomics.
    
    Run alignment to transform source data coordinates to match target data.
    
    Parameters
    ----------
    source_data : str or DataFrame
        Path to source data CSV or DataFrame with coordinates
    target_data : str or DataFrame
        Path to target data CSV or DataFrame with coordinates
    method : str, optional
        Transformation method, one of "rigid", "similarity", "affine", "projective"
    output_dir : str, optional
        Directory to save the alignment results
    **kwargs : dict
        Additional parameters to pass to align_spatial_transcriptomics
        
    Returns
    -------
    dict
        Dictionary containing transformation matrix and alignment metrics
    """
    # For backward compatibility, rename parameters
    return align_spatial_transcriptomics(
        reference_file=target_data,  # In old API, target was the reference
        target_file=source_data,     # In old API, source was to be aligned
        method=method,
        output_dir=output_dir,
        **kwargs
    )

def transform_coordinates(coordinates, transformation_matrix):
    """
    Transform coordinates using a transformation matrix.
    
    Parameters
    ----------
    coordinates : DataFrame or array-like
        Coordinates to transform, should have x and y columns or be a numpy array
    transformation_matrix : array-like
        Transformation matrix from alignment
        
    Returns
    -------
    DataFrame or array-like
        Transformed coordinates in the same format as input
    """
    import numpy as np
    import pandas as pd
    
    # Check if input is a DataFrame
    is_dataframe = isinstance(coordinates, pd.DataFrame)
    
    # Get coordinates as numpy array
    if is_dataframe:
        # Try to find x and y columns
        if 'x' in coordinates.columns and 'y' in coordinates.columns:
            x_col, y_col = 'x', 'y'
        elif 'X' in coordinates.columns and 'Y' in coordinates.columns:
            x_col, y_col = 'X', 'Y'
        else:
            # Use first two columns
            x_col, y_col = coordinates.columns[:2]
            
        # Extract coordinates
        coords = coordinates[[x_col, y_col]].values
    else:
        coords = np.array(coordinates)
    
    # Ensure coordinates are 2D
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    
    # If transformation matrix is 3x3 (homogeneous), apply homogeneous transformation
    if hasattr(transformation_matrix, 'shape') and transformation_matrix.shape == (3, 3):
        # Add homogeneous coordinate (z=1)
        homogeneous_coords = np.ones((coords.shape[0], 3))
        homogeneous_coords[:, 0:2] = coords[:, 0:2]
        
        # Apply transformation
        transformed = np.dot(homogeneous_coords, transformation_matrix.T)
        
        # Convert back from homogeneous coordinates
        transformed_coords = transformed[:, :2] / transformed[:, 2:]
    else:
        # Direct application of transformation matrix
        transformed_coords = np.dot(coords, transformation_matrix.T)
    
    # Return in the same format as input
    if is_dataframe:
        result = coordinates.copy()
        result[x_col] = transformed_coords[:, 0]
        result[y_col] = transformed_coords[:, 1]
        return result
    else:
        # For numpy arrays and lists
        return transformed_coords

# Update the __all__ list to include both new and old functions
__all__ = [
    'align_spatial_transcriptomics',
    'load_alignment',
    'save_alignment',
    # ST <-> SM registration
    'prepare_landmark_inputs',
    'register_sm_to_st',
    'read_sm_matrix',
    'read_st_annotations',
    'sm_coordinates',
    'st_coordinates',
    'rasterize_coordinates',
    'save_rasterized',
    'load_landmarks',
    'check_landmark_pair',
    'affine_from_landmarks',
    'run_lddmm_alignment',
    'transform_points',
    'save_transformed_data',
    'find_sm_matrix_files',
    'stalign_available',
    # Coarse pre-registration
    'build_pretransform',
    'apply_pretransform',
    'fit_scale_to_reference',
    'describe_pretransform',
    'overlap_score',
    'compose',
    'rotation_matrix',
    'flip_matrix',
    'scale_matrix',
    'translation_matrix',
    # Centroid-to-centroid registration (same section)
    'match_centroids',
    'split_pairs',
    'target_registration_error',
    'estimate_affine',
    'estimate_affine_ransac',
    'apply_affine',
    'fit_tps',
    'apply_tps',
    'register_centroids',
    'register_centroid_files',
    # Backward compatibility
    'run_alignment',
    'transform_coordinates'
]

# Add deprecated Xenium-Metabolomics helpers if available
if XENIUM_METABOLOMICS_AVAILABLE:
    __all__.extend([
        'align_xenium_to_metabolomics',
        'read_xenium_data',
        'visualize_alignment'
    ])
