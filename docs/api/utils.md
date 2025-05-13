# Utilities API

The utilities module provides helper functions for file operations, logging, and other common tasks.

## File Operations

::: smint.utils.file_utils.list_files
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.utils.file_utils.create_directory
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.utils.file_utils.get_file_info
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Logging Utilities

::: smint.utils.logging_utils.setup_logger
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.utils.logging_utils.log_parameters
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Configuration Utilities

::: smint.utils.config_utils.load_config
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

::: smint.utils.config_utils.save_config
    options:
      show_root_heading: true
      show_source: true
      show_signature_annotations: true

## Dependency Handling

The utilities module is designed to have minimal external dependencies, making it more robust across different environments. The core functionality requires only standard Python libraries, with some enhanced features available when optional dependencies are present.

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| PyYAML | Advanced config loading/saving | Fallback to JSON for configuration |
| tqdm | Progress tracking | Simple text-based progress updates |
| pandas | Data manipulation | Limited data handling capabilities |

The utilities module is designed to be a reliable foundation for the rest of the SMINT package, with minimal external dependencies to ensure robustness.