# Visualization API

The visualization module provides functions for visualizing segmentation results and spatial data.

## RGB Composites

::: smint.visualization.visualization_utils.create_rgb_composite
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Segmentation Visualization

::: smint.visualization.visualization_utils.visualize_segmentation_overlay
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.visualization.visualization_utils.visualize_cell_outlines
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Feature Visualization

::: smint.visualization.visualization_utils.plot_cell_features
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Animation

::: smint.visualization.visualization_utils.create_segmentation_animation
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Live Scan Viewer

::: smint.visualization.live_scan_viewer.LiveScanViewer
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Dependency Handling

The visualization module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| matplotlib | All plotting functionality | Stub implementation with helpful error messages |
| OpenCV (cv2) | Image I/O and processing | Limited visualization capabilities |
| tkinter | Live Scan Viewer GUI | Command-line only interface |
| numpy | Data manipulation | Required dependency |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.