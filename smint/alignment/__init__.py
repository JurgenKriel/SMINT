"""
Alignment module for SMINT.

Three capabilities, in the order they are normally used:

``pretransform``
    Coarse scale / rotation / flip to bring two modalities into roughly the
    same coordinate system.
``st_sm_registration``
    STalign LDDMM for **sequential sections**, around a manual landmark stage.
``centroid_registration``
    Correspondence fitting (affine / RANSAC / TPS) for post-staining on the
    **same section**, scored by held-out Target Registration Error.

Registration can be run in-process or handed to a worker via
:mod:`smint.alignment.jobs`.
"""

# ST <-> SM registration (STalign LDDMM, with a manual landmark stage).
# This is the supported entry point for registering spatial metabolomics onto
# spatial transcriptomics; see smint.alignment.st_sm_registration.
from .st_sm_registration import (
    prepare_landmark_inputs,
    register_sm_to_st,
    read_sm_matrix,
    read_st_annotations,
    resolve_st_columns,
    list_columns,
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
    save_transform,
    load_transform,
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

__all__ = [
    # Coarse pre-registration
    'build_pretransform',
    'apply_pretransform',
    'fit_scale_to_reference',
    'describe_pretransform',
    'overlap_score',
    'save_transform',
    'load_transform',
    'compose',
    'rotation_matrix',
    'flip_matrix',
    'scale_matrix',
    'translation_matrix',
    # ST <-> SM registration (sequential sections)
    'prepare_landmark_inputs',
    'register_sm_to_st',
    'read_sm_matrix',
    'read_st_annotations',
    'resolve_st_columns',
    'list_columns',
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
]

# Deprecated helpers, only if their optional dependencies import.
if XENIUM_METABOLOMICS_AVAILABLE:
    __all__.extend([
        'align_xenium_to_metabolomics',
        'read_xenium_data',
        'visualize_alignment',
    ])
