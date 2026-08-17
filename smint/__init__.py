"""
SMINT - Spatial Multi-Omics Integration

A Python package for Spatial Multi-Omics Integration with enhanced segmentation capabilities
and streamlined workflow.
"""

__version__ = '0.1.0'

import importlib

# Sub-packages are imported lazily (PEP 562). Importing them eagerly meant
# `import smint` -- or even `from smint.alignment import ...` -- pulled in the
# segmentation stack (cellpose -> segment_anything -> torchvision -> torch),
# so a broken or absent optional dependency in any one environment made the
# whole package unimportable. Attribute access still works exactly as before:
# `smint.alignment` resolves on first use.
_SUBPACKAGES = (
    'segmentation',
    'preprocessing',
    'visualization',
    'utils',
    'alignment',
    'r_integration',
)


def __getattr__(name):
    if name in _SUBPACKAGES:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))


__all__ = list(_SUBPACKAGES)
