# R Integration API

The R integration module provides functions for calling R scripts and functions from Python and for transferring data between Python and R.

## R Bridge

::: smint.r_integration.r_bridge.run_r_script
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.r_integration.r_bridge.initialize_r
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Data Conversion

::: smint.r_integration.r_bridge.r_to_pandas
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.r_integration.r_bridge.pandas_to_r
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## R Function Calls

::: smint.r_integration.r_bridge.run_r_function
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Dependency Handling

The R integration module has several optional dependencies:

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | Direct R integration | Fallback to subprocess R script execution |
| pandas | Data manipulation | Required dependency |
| numpy | Matrix operations | Required dependency |
| R | R script execution | Required for any R functionality |

When rpy2 is missing, SMINT will still allow you to run R scripts through subprocess calls, but the direct function calling and data transfer capabilities will be limited.