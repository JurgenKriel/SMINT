"""
Cellpose 3 / Cellpose 4 compatibility.

The two major versions differ in ways that silently change results rather than
raising, so every model load and ``eval`` call in SMINT goes through here:

- Cellpose 4 ships a single model, Cellpose-SAM (``cpsam``). ``model_type`` is
  ignored, so asking for ``"cyto"`` or ``"nuclei"`` quietly loads cpsam.
- Cellpose 4 ignores ``channels``; it takes the first three channels of the
  input as given. Channel selection therefore has to happen before ``eval``.
- ``use_gpu`` lives in ``cellpose.core`` in both versions; ``cellpose.models``
  does not re-export it.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Built-in model names that only exist in Cellpose <= 3.
LEGACY_BUILTIN_MODELS = {"cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3"}


def cellpose_major_version():
    """Return the installed Cellpose major version as an int, or None if absent."""
    try:
        from importlib.metadata import version
        return int(version("cellpose").split(".")[0])
    except Exception:
        return None


def use_gpu():
    """Whether Cellpose can use a GPU in this process."""
    from cellpose import core
    return bool(core.use_gpu())


def is_known_model(model):
    """True if ``model`` is a file path or a name Cellpose can load."""
    if model is None:
        return False
    return os.path.exists(str(model)) or str(model).lower() in LEGACY_BUILTIN_MODELS | {"cpsam"}


def load_model(model, gpu=None):
    """
    Load a Cellpose model by built-in name or file path.

    Args:
        model (str): A path to a trained model, or a built-in name. On Cellpose 3
            the names are ``cyto``, ``cyto2``, ``cyto3``, ``nuclei`` etc. On
            Cellpose 4 the only built-in is ``cpsam``; legacy names fall back to
            it with a warning.
        gpu (bool, optional): Use the GPU. Defaults to whatever is available.

    Returns:
        cellpose.models.CellposeModel: The loaded model.
    """
    from cellpose import models

    if gpu is None:
        gpu = use_gpu()
    model = str(model)
    major = cellpose_major_version()

    if os.path.exists(model):
        return models.CellposeModel(gpu=gpu, pretrained_model=model)

    if major is not None and major >= 4:
        if model.lower() != "cpsam":
            logger.warning(
                "Cellpose %s has no '%s' model; using the built-in Cellpose-SAM (cpsam) "
                "model instead. Pass a model path, or install cellpose<4, for the "
                "legacy models.", major, model,
            )
        return models.CellposeModel(gpu=gpu)

    if model.lower() == "cpsam":
        raise ValueError("The 'cpsam' model needs Cellpose 4 (pip install 'cellpose>=4').")
    return models.CellposeModel(gpu=gpu, model_type=model)


def select_channels(image, channels):
    """
    Pick the channels Cellpose should see, using Cellpose 3's convention.

    ``channels`` is ``[main, optional_nuclear]`` with 1-based indices into the
    channel axis and 0 meaning "none": ``[0, 0]`` is grayscale (first channel),
    ``[2, 1]`` is channel index 1 as the main channel with index 0 as the
    nuclear channel.

    Args:
        image (numpy.ndarray): ``(Y, X)`` or ``(C, Y, X)`` array.
        channels (sequence of int): Cellpose-style channel pair.

    Returns:
        numpy.ndarray: ``(Y, X)`` for a single channel, otherwise ``(2, Y, X)``.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected a (Y, X) or (C, Y, X) array, got shape {image.shape}")

    channels = list(channels or [0, 0]) + [0, 0]
    main, nuclear = int(channels[0]), int(channels[1])
    n_channels = image.shape[0]
    for c in (main, nuclear):
        if c > n_channels:
            raise ValueError(f"Channel {c} requested but the image has {n_channels} channels")

    main_plane = image[main - 1] if main > 0 else image[0]
    if nuclear > 0:
        return np.stack([main_plane, image[nuclear - 1]], axis=0)
    return main_plane


def run_eval(model, image, channels=(0, 0), **kwargs):
    """
    Run ``model.eval`` with channel handling that behaves the same on 3 and 4.

    Args:
        model (cellpose.models.CellposeModel): A loaded Cellpose model, e.g.
            from [`load_model`][smint.segmentation.cellpose_compat.load_model].
        image (numpy.ndarray): ``(Y, X)`` or ``(C, Y, X)`` array.
        channels (sequence of int): Cellpose 3 style channel pair, see
            [`select_channels`][smint.segmentation.cellpose_compat.select_channels].
        **kwargs (Any): Passed through to ``eval`` (``diameter``, ``flow_threshold``,
            ``cellprob_threshold`` ...).

    Returns:
        numpy.ndarray: Integer label mask.
    """
    x = select_channels(image, channels)
    major = cellpose_major_version()

    if x.ndim == 3:
        kwargs["channel_axis"] = 0
        if major is None or major < 4:
            kwargs["channels"] = [1, 2]
    elif major is None or major < 4:
        kwargs["channels"] = [0, 0]

    masks = model.eval(x, **kwargs)[0]
    return np.asarray(masks)
