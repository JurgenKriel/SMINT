# Preprocessing API

The preprocessing module provides functions for preparing and transforming spatial omics data for analysis.

## OME-TIFF Preprocessing

::: smint.preprocessing.preprocess_ome.split_large_ometiff
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.preprocessing.preprocess_ome.extract_channels
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Data Normalization

::: smint.preprocessing.normalization.normalize_expression
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.preprocessing.normalization.scale_data
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Dependency Handling

The preprocessing module relies on several libraries for optimal performance, but can operate with limited functionality when some dependencies are missing:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| tifffile | OME-TIFF reading/writing | Stub implementation with helpful error messages |
| scikit-image | Image transformations | Limited image processing capabilities |
| OpenCV (cv2) | Fast image I/O | Slower fallback implementation |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.