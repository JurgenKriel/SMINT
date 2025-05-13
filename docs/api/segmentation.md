# Segmentation API

The segmentation module provides functions for segmenting cells and nuclei in whole-slide images.

## Core Segmentation Functions

::: smint.segmentation.process_large_image
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.segmentation.run_distributed_segmentation
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Cell Utilities

::: smint.segmentation.get_cell_outlines
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.segmentation.segment_chunk
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Post-processing

::: smint.segmentation.extract_contours
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.segmentation.save_masks
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Optional Dependencies

The segmentation module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| Cellpose | Cell segmentation models | Stub implementation with helpful error messages |
| OpenCV (cv2) | Image I/O and contour extraction | Limited visualization, no mask saving/loading |
| Dask | Distributed processing | Single-process implementation only |
| Distributed | Multi-node computation | Single-node implementation only |
| CUDA | GPU acceleration | CPU-only implementation |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.