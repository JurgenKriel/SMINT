# Alignment API

The alignment module provides functions for aligning spatial omics data using ST Align.

## ST Align Wrapper

::: smint.alignment.st_align_wrapper.run_alignment
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.alignment.st_align_wrapper.transform_coordinates
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Configuration

::: smint.alignment.st_align_wrapper.create_config
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Validation

::: smint.alignment.st_align_wrapper.validate_alignment
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Dependency Handling

The alignment module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | R bridge functionality | Stub implementation with helpful error messages |
| pandas | Data manipulation | Required dependency |
| numpy | Matrix operations | Required dependency |
| matplotlib | Visualization | Limited visualization capabilities |

When a dependency is missing, SMINT will log a warning but continue to operate with limited functionality.