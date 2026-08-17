"""
napari plugin for SMINT registration.

Provides dock widgets to load ST/SM (or centroid) datasets, place paired
landmarks, and launch registration as a batch or local job.

Widgets are imported lazily so that merely having the plugin installed does not
pull Qt and magicgui into non-GUI sessions -- ``smint.cli.register`` runs on
compute nodes with no display.
"""

import importlib

_WIDGETS = ("load_datasets", "PreRegisterWidget", "LandmarkWidget", "registration_widget")

__all__ = list(_WIDGETS)


def __getattr__(name):
    if name in _WIDGETS:
        module = importlib.import_module("._widgets", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_WIDGETS))
