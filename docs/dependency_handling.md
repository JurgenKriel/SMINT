# Dependency Handling

SMINT is designed with graceful dependency handling to ensure robust operation across different environments, even when some optional dependencies are missing.

## Design Philosophy

The core design principles for dependency handling in SMINT are:

1. **Graceful Degradation**: The package should continue to function with reduced capabilities when optional dependencies are missing.
2. **Clear Feedback**: Users should receive informative messages about missing dependencies and their impact.
3. **Minimal Core Requirements**: The essential functionality should work with a minimal set of dependencies.
4. **Easy Extension**: Adding optional dependencies should enable additional features without code changes.

## How Dependency Handling Works

### Detection Mechanism

SMINT uses a consistent pattern across all modules to detect and handle optional dependencies:

```python
# Check for optional dependency
try:
    import some_package
    SOME_PACKAGE_AVAILABLE = True
except ImportError:
    SOME_PACKAGE_AVAILABLE = False
    logging.warning("some_package not available. Some functionality will be limited.")
```

### Implementation Pattern

Functions that require optional dependencies check their availability before execution:

```python
def function_requiring_dependency():
    if not SOME_PACKAGE_AVAILABLE:
        logging.error("This function requires some_package to be installed.")
        return {"error": "Missing required dependency: some_package"}
    
    # Normal function implementation
    ...
```

## Module-Specific Dependencies

### Segmentation Module

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| cellpose | Cell segmentation models | Error message with installation instructions |
| opencv-python | Image I/O and contour extraction | Limited visualization, no mask saving/loading |
| dask[distributed] | Distributed processing | Single-process implementation only |
| dask-cuda | GPU acceleration | CPU-only implementation |

### Visualization Module

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| matplotlib | All plotting functionality | Error message with installation instructions |
| opencv-python | Image I/O and processing | Limited visualization capabilities |
| tkinter | Live Scan Viewer GUI | Command-line only interface |

### R Integration Module

| Dependency | Required For | Fallback Behavior |
| ---------- | ------------ | ----------------- |
| rpy2 | Direct R integration | Fallback to subprocess R script execution |
| pandas | Data transfer between Python and R | Required dependency |
| R (system) | All R functionality | Error message with installation instructions |

## Checking Dependency Status

You can check the status of optional dependencies programmatically:

```python
import importlib

def check_dependency(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

# Check for cellpose
cellpose_available = check_dependency("cellpose")
print(f"Cellpose available: {cellpose_available}")

# Check for opencv
opencv_available = check_dependency("cv2")
print(f"OpenCV available: {opencv_available}")

# Check for rpy2
rpy2_available = check_dependency("rpy2")
print(f"rpy2 available: {rpy2_available}")
```

## Warnings and Error Messages

SMINT provides detailed warnings and error messages when dependencies are missing:

1. **Import-time warnings**: When SMINT is imported, it will log warnings about missing optional dependencies.
2. **Function-specific errors**: Functions that require missing dependencies will provide clear error messages.
3. **Installation instructions**: Error messages include instructions for installing missing dependencies.

## Customizing Error Handling

You can customize the behavior when dependencies are missing by modifying the logging configuration:

```python
import logging

# Set logging level to ERROR to suppress warnings about missing dependencies
logging.basicConfig(level=logging.ERROR)

# Or capture logs to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='smint.log'
)
```

## Best Practices for Dependency Management

1. **Install only what you need**: Use selective installation to minimize dependencies:
   ```bash
   pip install "smint[segmentation]"  # Only install segmentation dependencies
   ```

2. **Check availability before calling functions**:
   ```python
   import smint
   from smint.segmentation import run_distributed_segmentation
   
   # Check if distributed processing is available
   try:
      import distributed
      distributed_available = True
   except ImportError:
      distributed_available = False
   
   # Use appropriate function based on availability
   if distributed_available:
      result = run_distributed_segmentation(...)
   else:
      result = smint.segmentation.process_large_image(...)
   ```

3. **Provide feedback to users**: When building applications on top of SMINT, forward dependency warnings to users.