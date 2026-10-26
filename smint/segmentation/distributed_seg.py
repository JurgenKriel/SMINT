"""
Distributed whole-slide segmentation with Dask.

The image is cut into a grid of core tiles. Each Dask worker segments one tile
*expanded* by an overlap margin, so cells on a tile boundary are seen whole in
both neighbouring tiles:

1. The worker writes only the tile's core region into shared label-mask Zarr
   arrays, so no pixel is written twice.
2. It extracts outlines from the full expanded mask, so each boundary cell is
   reported, complete, by both tiles.
3. The coordinator stitches those duplicates and reconciles cells with nuclei
   (:mod:`smint.segmentation.stitching`, a port of ``stitching/stitching.Rmd``),
   then writes outline and summary CSVs per object type.

Workers come from a local cluster (CPU, or one worker per GPU via dask_cuda),
from SLURM via dask_jobqueue, or from an existing scheduler. Workers read the
image themselves, by path, so the image must be on storage they can all see.
"""

import json
import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import dask.array as da
import tifffile

from ..utils.config import load_config
from .cellpose_compat import is_known_model, load_model, run_eval
from .stitching import (
    filter_by_nuclei,
    polygons_from_outlines,
    reconcile_cells_and_nuclei,
    stitch_polygons,
)

logger = logging.getLogger(__name__)

# Optional dependencies. Attempt the real import rather than find_spec: a
# package whose own dependency chain is broken passes find_spec and then raises.
try:
    from distributed import Client, LocalCluster, as_completed
    DISTRIBUTED_AVAILABLE = True
except Exception as exc:
    DISTRIBUTED_AVAILABLE = False
    logger.debug("'distributed' unavailable (%s); distributed segmentation disabled.", exc)

try:
    import zarr
    ZARR_AVAILABLE = True
except Exception as exc:
    ZARR_AVAILABLE = False
    logger.debug("'zarr' unavailable (%s); distributed segmentation disabled.", exc)

try:
    import cellpose  # noqa: F401
    CELLPOSE_AVAILABLE = True
except Exception as exc:
    CELLPOSE_AVAILABLE = False
    logger.debug("'cellpose' unavailable (%s).", exc)

DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_OVERLAP = 256
MAX_CELLS_PER_TILE = 100_000


def setup_logging(output_dir=None):
    """
    Set up logging for a distributed run.

    Args:
        output_dir (str, optional): Directory for a timestamped log file. If
            None, log to the console only.

    Returns:
        logging.Logger: Configured logger instance
    """
    log = logging.getLogger('smint.distributed_seg')
    log.setLevel(logging.INFO)
    # Stitching reports its merge / trim / reconcile counts on its own logger
    logging.getLogger('smint.segmentation.stitching').setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Log to the console only if nothing up the hierarchy already does, so a
    # calling script's logging setup doesn't print every line twice.
    for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
        log.removeHandler(h)
        h.close()
    stitch_log = logging.getLogger('smint.segmentation.stitching')
    if not log.hasHandlers() and not logging.getLogger().handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)
        stitch_log.addHandler(console_handler)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, f'distributed_seg_{time.strftime("%Y%m%d-%H%M%S")}.log'))
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        for h in [h for h in stitch_log.handlers if isinstance(h, logging.FileHandler)]:
            stitch_log.removeHandler(h)
        stitch_log.addHandler(file_handler)

    return log


# ---------------------------------------------------------------------------
# Image access
# ---------------------------------------------------------------------------

def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _ome_zarr_multiscale(path):
    """Return (dataset path, axis names) from OME-Zarr metadata, or (None, None)."""
    attrs = _read_json(os.path.join(path, '.zattrs'))
    if attrs is None:
        meta = _read_json(os.path.join(path, 'zarr.json')) or {}
        attrs = meta.get('attributes', {})
        attrs = attrs.get('ome', attrs)
    try:
        ms = attrs['multiscales'][0]
        axes = ms.get('axes')
        if axes and isinstance(axes[0], dict):
            axes = [a['name'] for a in axes]
        return ms['datasets'][0]['path'], ''.join(axes).upper() if axes else None
    except (KeyError, IndexError, TypeError):
        return None, None


def _open_zarr(path):
    """Open the full-resolution array of a Zarr or OME-Zarr store."""
    try:
        return da.from_zarr(path), None
    except Exception:
        pass

    component, axes = _ome_zarr_multiscale(path)
    if component is not None:
        return da.from_zarr(path, component=component), axes

    group = zarr.open_group(path, mode='r')
    for key in ('0', '0/0'):
        try:
            if hasattr(group[key], 'shape'):
                return da.from_zarr(path, component=key), None
        except KeyError:
            continue
    keys = list(group.array_keys())
    if keys:
        return da.from_zarr(path, component=keys[0]), None
    raise ValueError(f"No image array found in Zarr store: {path}")


def _to_cyx(arr, axes=None):
    """Reduce an array to (C, Y, X), taking index 0 of any T/Z axes."""
    if axes and len(axes) == arr.ndim and 'Y' in axes and 'X' in axes:
        channel_axis = next((a for a in 'CSIQ' if a in axes), None)
        index = []
        for a in axes:
            index.append(slice(None) if a in ('Y', 'X', channel_axis) else 0)
        arr = arr[tuple(index)]
        kept = [a for a in axes if a in ('Y', 'X', channel_axis)]
        if channel_axis is None:
            return arr[None]
        return arr.transpose([kept.index(channel_axis), kept.index('Y'), kept.index('X')])

    if arr.ndim == 2:
        return arr[None]
    while arr.ndim > 3:
        arr = arr[0]
    if arr.shape[-1] <= 4 < arr.shape[0]:  # (Y, X, C), e.g. RGB
        return arr.transpose(2, 0, 1)
    return arr


def open_image(image_path):
    """
    Open a large image lazily as a ``(C, Y, X)`` Dask array.

    Supports Zarr / OME-Zarr stores and TIFF / OME-TIFF files. Nothing is read
    until the array is sliced and computed.

    Args:
        image_path (str): Path to the image.

    Returns:
        dask.array.Array: ``(C, Y, X)`` array.
    """
    path = str(image_path).rstrip('/')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    if path.lower().endswith(('.tif', '.tiff')):
        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes
        arr = da.from_zarr(tifffile.imread(path, aszarr=True, series=0, level=0))
    else:
        if not ZARR_AVAILABLE:
            raise ImportError("Reading Zarr images requires zarr (pip install zarr)")
        arr, axes = _open_zarr(path)
    return _to_cyx(arr, axes)


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

def _worker_cache():
    """Per-worker dict that survives between tasks, so models load once."""
    try:
        from distributed import get_worker
        w = get_worker()
        if not hasattr(w, '_smint_cache'):
            w._smint_cache = {}
        return w._smint_cache
    except Exception:
        return _LOCAL_CACHE


_LOCAL_CACHE = {}


def _needed_channels(*channel_specs):
    """0-based channel indices used by Cellpose-style channel pairs."""
    needed = {c - 1 for spec in channel_specs if spec for c in spec if c > 0}
    return sorted(needed) or [0]


def _remap_channels(spec, needed):
    """Rewrite a 1-based channel pair to index into the subset ``needed``."""
    return [needed.index(c - 1) + 1 if c > 0 else 0 for c in (list(spec) + [0, 0])[:2]]


def _split_multinucleated(cell_masks, nuclei_masks):
    """Split cells containing several nuclei by a watershed seeded on the nuclei."""
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt, find_objects

    cell_masks = cell_masks.astype(np.int32)
    nuclei_masks = nuclei_masks.astype(np.int32)
    next_id = int(cell_masks.max()) + 1

    for idx, sl in enumerate(find_objects(cell_masks)):
        if sl is None:
            continue
        cid = idx + 1
        local_cell = cell_masks[sl] == cid
        local_nuclei = nuclei_masks[sl] * local_cell
        nuclei_ids = np.unique(local_nuclei)
        nuclei_ids = nuclei_ids[nuclei_ids > 0]
        if len(nuclei_ids) < 2:
            continue
        labels = watershed(-distance_transform_edt(local_cell), local_nuclei, mask=local_cell)
        view = cell_masks[sl]
        for i, nid in enumerate(nuclei_ids):
            if i == 0:
                view[labels == nid] = cid
            else:
                view[labels == nid] = next_id
                next_id += 1
    return cell_masks


def _mask_outlines(masks, x_origin, y_origin, id_offset):
    """Outline every label in ``masks`` in global image coordinates."""
    import cv2
    from skimage.measure import regionprops

    rows = []
    for prop in regionprops(masks):
        r0, c0, r1, c1 = prop.bbox
        local = (masks[r0:r1, c0:c1] == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        c = max(contours, key=len).squeeze()
        if c.ndim < 2:
            continue
        rows.append({
            'global_id': int(prop.label) + id_offset,
            'x_coords': c[:, 0].astype(float) + c0 + x_origin,
            'y_coords': c[:, 1].astype(float) + r0 + y_origin,
        })
    return rows


def segment_tile(tile_id, tile_coords, image_path, image_shape, overlap,
                 cell_model, cell_params, nuclei_model, nuclei_params,
                 cells_zarr_path, nuclei_zarr_path, use_gpu=None,
                 split_multinucleated=True, max_cells_per_tile=MAX_CELLS_PER_TILE):
    """
    Segment one tile. Runs on a Dask worker.

    Args:
        tile_id (int): Tile index; offsets label IDs so tiles never collide.
        tile_coords (tuple): Core region ``(y_start, x_start, y_end, x_end)``.
        image_path (str): Image path, readable from the worker.
        image_shape (tuple): Full image ``(height, width)``.
        overlap (int): Pixels the tile is expanded by on each side.
        cell_model (str or None): Cell model name or path; None skips cells.
        cell_params (dict): ``diameter``, ``flow_threshold``,
            ``cellprob_threshold``, ``channels`` for the cell model.
        nuclei_model (str or None): Nuclei model name or path; None skips nuclei.
        nuclei_params (dict): As ``cell_params``, for the nuclei model.
        cells_zarr_path (str): Label-mask Zarr for cells.
        nuclei_zarr_path (str): Label-mask Zarr for nuclei.
        use_gpu (bool, optional): Use the GPU. Defaults to what is available.
        split_multinucleated (bool): Split cells containing several nuclei.
        max_cells_per_tile (int): Label ID offset between tiles.

    Returns:
        dict: ``success``, ``error``, ``n_cells``, ``n_nuclei`` and the tile's
        ``cell_outlines`` / ``nuclei_outlines``.
    """
    y_start, x_start, y_end, x_end = tile_coords
    height, width = image_shape
    result = {'tile_id': tile_id, 'tile_coords': tile_coords, 'success': False,
              'error': None, 'n_cells': 0, 'n_nuclei': 0,
              'cell_outlines': [], 'nuclei_outlines': []}

    y0, y1 = max(0, y_start - overlap), min(height, y_end + overlap)
    x0, x1 = max(0, x_start - overlap), min(width, x_end + overlap)
    yo, xo = y_start - y0, x_start - x0
    core_h, core_w = y_end - y_start, x_end - x_start

    cache = _worker_cache()
    try:
        for name in (cell_model, nuclei_model):
            if name is not None and ('model', name) not in cache:
                cache[('model', name)] = load_model(name, gpu=use_gpu)
    except Exception as e:
        result['error'] = f"Model load failed: {e}"
        return result

    cell_params = dict(cell_params or {})
    nuclei_params = dict(nuclei_params or {})
    cell_channels = cell_params.pop('channels', [0, 0])
    nuclei_channels = nuclei_params.pop('channels', [0, 0])
    needed = _needed_channels(cell_channels if cell_model else None,
                              nuclei_channels if nuclei_model else None)

    try:
        if ('image', image_path) not in cache:
            cache[('image', image_path)] = open_image(image_path)
        # Read in this process: a worker's default scheduler is the cluster,
        # which would try to ship the open file handle back over the network.
        tile = np.asarray(cache[('image', image_path)][needed, y0:y1, x0:x1]
                          .compute(scheduler='synchronous'))
    except Exception as e:
        result['error'] = f"Image read failed: {e}"
        return result

    try:
        shape = tile.shape[1:]
        cell_masks = np.zeros(shape, np.int32)
        nuclei_masks = np.zeros(shape, np.int32)
        if cell_model is not None:
            cell_masks = run_eval(cache[('model', cell_model)], tile,
                                  _remap_channels(cell_channels, needed), **cell_params).astype(np.int32)
        if nuclei_model is not None:
            nuclei_masks = run_eval(cache[('model', nuclei_model)], tile,
                                    _remap_channels(nuclei_channels, needed), **nuclei_params).astype(np.int32)
        if split_multinucleated and cell_model is not None and nuclei_model is not None:
            cell_masks = _split_multinucleated(cell_masks, nuclei_masks)
    except Exception as e:
        result['error'] = f"Segmentation failed: {e}"
        return result

    id_offset = tile_id * max_cells_per_tile
    for masks, path in ((cell_masks, cells_zarr_path), (nuclei_masks, nuclei_zarr_path)):
        core = masks[yo:yo + core_h, xo:xo + core_w]
        out = np.where(core > 0, core.astype(np.uint32) + id_offset, 0).astype(np.uint32)
        try:
            zarr.open_array(path, mode='r+')[y_start:y_end, x_start:x_end] = out
        except Exception as e:
            result['error'] = f"Zarr write failed: {e}"
            return result

    result['cell_outlines'] = _mask_outlines(cell_masks, x0, y0, id_offset)
    result['nuclei_outlines'] = _mask_outlines(nuclei_masks, x0, y0, id_offset)
    result['n_cells'] = len(result['cell_outlines'])
    result['n_nuclei'] = len(result['nuclei_outlines'])
    result['success'] = True
    return result


# ---------------------------------------------------------------------------
# Outputs (coordinator side)
# ---------------------------------------------------------------------------

def stitch_and_reconcile(cell_outlines, nuclei_outlines, merge_threshold=0.5,
                         trim_threshold=0.25, reconcile=True):
    """
    Turn raw per-tile outlines into final, non-overlapping cells and nuclei.

    Nuclei are stitched first; cells are filtered to those touching a stitched
    nucleus, then stitched; finally cells and nuclei are reconciled so every
    nucleus sits inside exactly one cell. See :mod:`smint.segmentation.stitching`.

    Args:
        cell_outlines (list or None): Raw cell outline dicts, or None to skip cells.
        nuclei_outlines (list or None): Raw nucleus outline dicts, or None.
        merge_threshold (float): Merge pairs overlapping by more than this.
        trim_threshold (float): Trim below, remove smaller at or above this.
        reconcile (bool): Run cell / nucleus reconciliation when both exist.

    Returns:
        tuple: ``(cells, nuclei, nucleus_cell)``; ``nucleus_cell`` holds each
        nucleus's index into ``cells``, or None when not reconciled.
    """
    log = logging.getLogger('smint.distributed_seg')
    cells = nuclei = nucleus_cell = None
    if nuclei_outlines is not None:
        log.info(f"Stitching {len(nuclei_outlines)} nucleus outlines")
        nuclei = stitch_polygons(polygons_from_outlines(nuclei_outlines),
                                 merge_threshold, trim_threshold)
    if cell_outlines is not None:
        cells = polygons_from_outlines(cell_outlines)
        if nuclei is not None:
            n = len(cells)
            cells = filter_by_nuclei(cells, nuclei)
            log.info(f"{n - len(cells)} of {n} cell outlines touch no nucleus and were dropped")
        log.info(f"Stitching {len(cells)} cell outlines")
        cells = stitch_polygons(cells, merge_threshold, trim_threshold)
    if reconcile and cells is not None and nuclei is not None:
        cells, nuclei, nucleus_cell = reconcile_cells_and_nuclei(cells, nuclei)
    return cells, nuclei, nucleus_cell


def polygons_to_dataframe(polygons):
    """Flatten polygons to SMINT's ``global_cell_id, x, y`` vertex table, IDs from 1."""
    if len(polygons) == 0:
        return pd.DataFrame(columns=['global_cell_id', 'x', 'y'])
    coords = [np.asarray(p.exterior.coords) for p in polygons]
    return pd.DataFrame({
        'global_cell_id': np.repeat(np.arange(1, len(coords) + 1), [len(c) for c in coords]),
        'x': np.concatenate([c[:, 0] for c in coords]),
        'y': np.concatenate([c[:, 1] for c in coords]),
    })


def polygon_summary(polygons, **columns):
    """One row per polygon: ``global_cell_id``, centroid, area, plus ``columns``."""
    import shapely
    centroids = shapely.centroid(np.asarray(polygons, dtype=object)) if len(polygons) else []
    table = pd.DataFrame({
        'global_cell_id': np.arange(1, len(polygons) + 1),
        'x_centroid': shapely.get_x(centroids) if len(polygons) else [],
        'y_centroid': shapely.get_y(centroids) if len(polygons) else [],
        'area': shapely.area(np.asarray(polygons, dtype=object)) if len(polygons) else [],
    })
    for name, values in columns.items():
        table[name] = values
    return table


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

def setup_dask_client(backend='local', n_workers=None, use_gpu=True,
                      scheduler_address=None, slurm=None, worker_wait_timeout=600):
    """
    Create a Dask client.

    Args:
        backend (str): ``'local'`` or ``'slurm'``. Ignored when
            ``scheduler_address`` is given.
        n_workers (int, optional): Number of workers. Local default: one per
            GPU with dask_cuda, otherwise ``min(4, cpu_count)``. SLURM default: 4.
        use_gpu (bool): Request GPU workers.
        scheduler_address (str, optional): Connect to an existing scheduler.
        slurm (dict, optional): ``dask_jobqueue.SLURMCluster`` options. SMINT
            defaults: ``queue='gpuq'``, ``memory='64GB'``,
            ``walltime='12:00:00'``, ``gpu_type`` (adds
            ``--gres=gpu:<type>:1``), ``python`` (the current interpreter).
        worker_wait_timeout (int): Seconds to wait for the first worker.

    Returns:
        tuple: ``(client, cluster)``; ``cluster`` is None for an external scheduler.
    """
    if not DISTRIBUTED_AVAILABLE:
        raise ImportError("Distributed segmentation requires 'distributed' "
                          "(pip install 'smint[distributed]').")
    log = logging.getLogger('smint.distributed_seg')

    if scheduler_address:
        client = Client(scheduler_address)
        log.info(f"Connected to scheduler at {scheduler_address}")
        return client, None

    if backend == 'slurm':
        try:
            from dask_jobqueue import SLURMCluster
        except ImportError as e:
            raise ImportError("backend='slurm' requires dask-jobqueue "
                              "(pip install 'smint[distributed]').") from e
        opts = {'queue': 'gpuq', 'cores': 1, 'processes': 1, 'memory': '64GB',
                'walltime': '12:00:00', 'python': sys.executable,
                'log_directory': 'dask_logs'}
        opts.update(slurm or {})
        gpu_type = opts.pop('gpu_type', None)
        if use_gpu:
            gres = f'--gres=gpu:{gpu_type}:1' if gpu_type else '--gres=gpu:1'
            opts['job_extra_directives'] = list(opts.get('job_extra_directives', [])) + [gres]
        os.makedirs(opts['log_directory'], exist_ok=True)
        cluster = SLURMCluster(**opts)
        cluster.scale(jobs=n_workers or 4)
        log.info(f"SLURM worker job script:\n{cluster.job_script()}")
    elif backend == 'local':
        cluster = None
        if use_gpu:
            try:
                from dask_cuda import LocalCUDACluster
                cluster = LocalCUDACluster(n_workers=n_workers, threads_per_worker=1)
            except Exception as e:
                log.warning(f"No LocalCUDACluster ({e}); using CPU workers. "
                            "Install dask-cuda for one worker per GPU.")
        if cluster is None:
            cluster = LocalCluster(n_workers=n_workers or min(4, os.cpu_count() or 1),
                                   threads_per_worker=1)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'local' or 'slurm'")

    client = Client(cluster)
    log.info(f"Dask dashboard: {client.dashboard_link}")
    try:
        client.wait_for_workers(n_workers=1, timeout=worker_wait_timeout)
    except Exception:
        client.close()
        cluster.close()
        raise TimeoutError(f"No Dask workers came online within {worker_wait_timeout}s")
    log.info(f"{len(client.scheduler_info()['workers'])} worker(s) ready")
    return client, cluster


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def generate_tiles(image_shape, tile_size):
    """Non-overlapping core tile grid as ``[(tile_id, (y0, x0, y1, x1)), ...]``."""
    height, width = image_shape
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tiles.append((len(tiles), (y, x, min(y + tile_size, height), min(x + tile_size, width))))
    return tiles


def _model_params(params):
    """Keep only the keys passed to Cellpose."""
    keys = ('diameter', 'flow_threshold', 'cellprob_threshold', 'channels')
    return {k: params[k] for k in keys if k in params and params[k] is not None}


def _write_live_update(cells_zarr_path, tile_coords, image_shape,
                       live_update_image_path, tile_info_path):
    """Write a tile's cell boundaries and the live viewer's tile-info line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    y0, x0, y1, x1 = tile_coords
    masks = np.asarray(zarr.open_array(cells_zarr_path, mode='r')[y0:y1, x0:x1])
    plt.imsave(live_update_image_path, find_boundaries(masks), cmap='gray')
    if tile_info_path:
        with open(tile_info_path, 'w') as f:
            f.write(f"{y0},{x0},{y1 - y0},{x1 - x0},{image_shape[0]},{image_shape[1]},"
                    f"{os.path.basename(live_update_image_path)}")


def run_distributed_segmentation(image_path, output_dir, config_path=None,
                                 n_workers=None, use_gpu=None, chunk_size=None,
                                 tile_info_path=None, live_update_image_path=None,
                                 backend=None, scheduler_address=None, tile_overlap=None,
                                 cell_model=None, nuclei_model=None,
                                 cell_params=None, nuclei_params=None,
                                 split_multinucleated=None, reconcile_nuclei=None,
                                 stitch_merge_thresh=None, stitch_trim_thresh=None,
                                 slurm=None, worker_wait_timeout=None):
    """
    Segment cells and nuclei across a whole slide with Dask workers.

    Arguments left as None come from the YAML config, then from defaults.
    Config keys: ``model_paths`` (``[cell_model, nuclei_model]``),
    ``cell_model_params`` / ``nuclei_model_params`` (``diameter``,
    ``flow_threshold``, ``cellprob_threshold``, ``channels``), ``n_workers``,
    ``use_gpu``, ``chunk_size``, and an optional ``distributed`` section with
    ``backend``, ``scheduler_address``, ``tile_overlap``,
    ``split_multinucleated``, ``reconcile_nuclei``, ``stitch_merge_thresh``,
    ``stitch_trim_thresh``, ``worker_wait_timeout`` and ``slurm``.

    Args:
        image_path (str): Zarr / OME-Zarr store or TIFF / OME-TIFF file. Must be
            readable by every worker (for SLURM, on shared storage).
        output_dir (str): Output directory, shared with the workers.
        config_path (str, optional): YAML configuration file.
        n_workers (int, optional): Number of Dask workers.
        use_gpu (bool, optional): Use GPU workers. Default True.
        chunk_size (tuple, optional): Core tile ``(height, width)``; tiles are
            square, so the first value is used. Default 2048.
        tile_info_path (str, optional): Tile-info file for the live viewer.
        live_update_image_path (str, optional): PNG updated as tiles finish.
        backend (str, optional): ``'local'`` (default) or ``'slurm'``.
        scheduler_address (str, optional): Use an existing Dask scheduler.
        tile_overlap (int, optional): Expansion on each tile side; must exceed
            the largest cell radius. Default 256.
        cell_model (str, optional): Cell model name or path; ``False`` skips cells.
        nuclei_model (str, optional): Nuclei model name or path; ``False``
            skips nuclei.
        cell_params (dict, optional): Cell model parameters.
        nuclei_params (dict, optional): Nuclei model parameters.
        split_multinucleated (bool, optional): Split cells with several nuclei.
            Default True.
        reconcile_nuclei (bool, optional): Reconcile cells with nuclei and
            assign each nucleus to a cell. Default True.
        stitch_merge_thresh (float, optional): Merge above this overlap. Default 0.5.
        stitch_trim_thresh (float, optional): Trim below this overlap, remove
            the smaller object at or above it. Default 0.25.
        slurm (dict, optional): SLURMCluster options, see :func:`setup_dask_client`.
        worker_wait_timeout (int, optional): Seconds to wait for workers. Default 600.

    Returns:
        dict: ``cells_csv_path``, ``nuclei_csv_path``, ``cells_summary_path``,
        ``nuclei_summary_path``, ``cells_zarr_path``, ``nuclei_zarr_path``,
        ``total_cells``, ``total_nuclei``, ``n_tiles``, ``failed_tiles`` and
        ``processing_time``.

    Raises:
        ImportError: A required package is missing.
        RuntimeError: Every tile failed.

    Outputs in ``output_dir``:

    - ``cells_outlines.csv`` / ``nuclei_outlines.csv`` -- one row per outline
      vertex: ``global_cell_id, x, y``, IDs from 1 after stitching.
    - ``cells_summary.csv`` -- ``global_cell_id, x_centroid, y_centroid, area,
      n_nuclei``; ``nuclei_summary.csv`` -- the same with ``cell_id`` (the
      containing cell's ``global_cell_id``) in place of ``n_nuclei``.
    - ``cells.zarr`` / ``nuclei.zarr`` -- raw label masks. Their IDs are
      per-tile and are not the stitched IDs.
    """
    for ok, pkg in ((DISTRIBUTED_AVAILABLE, 'distributed'), (ZARR_AVAILABLE, 'zarr'),
                    (CELLPOSE_AVAILABLE, 'cellpose')):
        if not ok:
            raise ImportError(f"run_distributed_segmentation requires '{pkg}' "
                              "(pip install 'smint[distributed]').")

    log = setup_logging(output_dir)
    start = time.time()

    config = load_config(config_path) if config_path else {}
    if config_path:
        log.info(f"Loaded configuration from {config_path}")
    dist = config.get('distributed') or {}

    def pick(value, key, default, section=dist):
        if value is not None:
            return value
        return section.get(key, default)

    model_paths = list(config.get('model_paths') or ['cyto', 'nuclei']) + [None, None]
    cell_model = pick(cell_model, None, model_paths[0], {})
    nuclei_model = pick(nuclei_model, None, model_paths[1], {})
    cell_model = cell_model or None
    nuclei_model = nuclei_model or None
    if cell_model is None and nuclei_model is None:
        raise ValueError("No cell or nuclei model given")
    for m in (cell_model, nuclei_model):
        if m is not None and not is_known_model(m):
            raise ValueError(f"Model '{m}' is neither a file nor a built-in Cellpose model")

    cell_params = _model_params(pick(cell_params, 'cell_model_params', {}, config))
    nuclei_params = _model_params(pick(nuclei_params, 'nuclei_model_params', {}, config))
    use_gpu = pick(use_gpu, 'use_gpu', True, config)
    n_workers = pick(n_workers, 'n_workers', None, config)
    chunk_size = pick(chunk_size, 'chunk_size', DEFAULT_TILE_SIZE, config)
    tile_size = int(chunk_size[0] if isinstance(chunk_size, (list, tuple)) else chunk_size)
    backend = pick(backend, 'backend', 'local')
    scheduler_address = pick(scheduler_address, 'scheduler_address', None)
    tile_overlap = int(pick(tile_overlap, 'tile_overlap', DEFAULT_TILE_OVERLAP))
    split_multinucleated = pick(split_multinucleated, 'split_multinucleated', True)
    reconcile_nuclei = bool(pick(reconcile_nuclei, 'reconcile_nuclei', True))
    stitch_merge_thresh = float(pick(stitch_merge_thresh, 'stitch_merge_thresh', 0.5))
    stitch_trim_thresh = float(pick(stitch_trim_thresh, 'stitch_trim_thresh', 0.25))
    slurm = pick(slurm, 'slurm', None)
    worker_wait_timeout = int(pick(worker_wait_timeout, 'worker_wait_timeout', 600))

    image = open_image(image_path)
    image_shape = tuple(int(s) for s in image.shape[1:])
    tiles = generate_tiles(image_shape, tile_size)
    log.info(f"Image {image_path}: {image.shape[0]} channel(s), {image_shape[0]} x {image_shape[1]} px")
    log.info(f"Tiles: {len(tiles)} of {tile_size} px, {tile_overlap} px overlap")
    log.info(f"Cell model: {cell_model} {cell_params}")
    log.info(f"Nuclei model: {nuclei_model} {nuclei_params}")
    if len(tiles) * MAX_CELLS_PER_TILE >= 2 ** 32:
        raise ValueError(f"{len(tiles)} tiles would overflow uint32 label IDs; use a larger tile size")

    os.makedirs(output_dir, exist_ok=True)
    cells_zarr_path = os.path.join(output_dir, 'cells.zarr')
    nuclei_zarr_path = os.path.join(output_dir, 'nuclei.zarr')
    for path in (cells_zarr_path, nuclei_zarr_path):
        zarr.open_array(path, mode='w', shape=image_shape, chunks=(tile_size, tile_size),
                        dtype='uint32', fill_value=0)

    client, cluster = setup_dask_client(backend, n_workers, use_gpu, scheduler_address,
                                        slurm, worker_wait_timeout)
    results = []
    try:
        futures = [
            client.submit(segment_tile, tid, coords, os.path.abspath(image_path), image_shape,
                          tile_overlap, cell_model, cell_params, nuclei_model, nuclei_params,
                          os.path.abspath(cells_zarr_path), os.path.abspath(nuclei_zarr_path),
                          use_gpu, split_multinucleated, MAX_CELLS_PER_TILE,
                          key=f"smint-tile-{tid}", pure=False)
            for tid, coords in tiles
        ]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r['success']:
                log.info(f"Tile {len(results)}/{len(tiles)} {r['tile_coords']}: "
                         f"{r['n_cells']} cells, {r['n_nuclei']} nuclei")
                if live_update_image_path:
                    try:
                        _write_live_update(cells_zarr_path, r['tile_coords'], image_shape,
                                           live_update_image_path, tile_info_path)
                    except Exception as e:
                        log.warning(f"Live update failed: {e}")
            else:
                log.error(f"Tile {r['tile_id']} {r['tile_coords']} failed: {r['error']}")
    finally:
        client.close()
        if cluster is not None:
            cluster.close()

    failed = [r for r in results if not r['success']]
    if len(failed) == len(results):
        raise RuntimeError(f"All {len(results)} tiles failed; first error: {failed[0]['error']}")

    output = {'cells_zarr_path': cells_zarr_path, 'nuclei_zarr_path': nuclei_zarr_path,
              'cells_csv_path': None, 'nuclei_csv_path': None,
              'cells_summary_path': None, 'nuclei_summary_path': None,
              'total_cells': 0, 'total_nuclei': 0, 'n_tiles': len(tiles),
              'failed_tiles': [r['tile_coords'] for r in failed]}

    def collect(key, model):
        if model is None:
            return None
        return [o for r in results if r['success'] for o in r[key]]

    cells, nuclei, nucleus_cell = stitch_and_reconcile(
        collect('cell_outlines', cell_model), collect('nuclei_outlines', nuclei_model),
        stitch_merge_thresh, stitch_trim_thresh, reconcile_nuclei)

    for kind, polys in (('cells', cells), ('nuclei', nuclei)):
        if polys is None:
            continue
        csv_path = os.path.join(output_dir, f'{kind}_outlines.csv')
        polygons_to_dataframe(polys).to_csv(csv_path, index=False)
        extra = {}
        if nucleus_cell is not None:
            extra = ({'n_nuclei': np.bincount(nucleus_cell, minlength=len(cells))}
                     if kind == 'cells' else {'cell_id': nucleus_cell + 1})
        summary_path = os.path.join(output_dir, f'{kind}_summary.csv')
        polygon_summary(polys, **extra).to_csv(summary_path, index=False)
        output[f'{kind}_csv_path'] = csv_path
        output[f'{kind}_summary_path'] = summary_path
        output[f'total_{kind}'] = len(polys)
        log.info(f"{len(polys)} {kind} -> {csv_path}")

    output['processing_time'] = time.time() - start
    if failed:
        log.warning(f"{len(failed)} of {len(tiles)} tiles failed; their regions are empty")
    log.info(f"Done in {output['processing_time'] / 60:.1f} min")
    return output


def main(argv=None):
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Distributed whole-slide segmentation with Dask")
    parser.add_argument("--image_path", required=True, help="Zarr/OME-Zarr store or TIFF/OME-TIFF")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config_path", help="YAML configuration file")
    parser.add_argument("--backend", choices=["local", "slurm"], help="Where workers run (default: local)")
    parser.add_argument("--scheduler_address", help="Connect to an existing Dask scheduler")
    parser.add_argument("--n_workers", type=int, help="Number of workers")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=None)
    gpu.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--tile_size", type=int, help="Core tile size in pixels (default 2048)")
    parser.add_argument("--tile_overlap", type=int, help="Tile overlap in pixels (default 256)")
    parser.add_argument("--cell_model", help="Cell model name or path")
    parser.add_argument("--nuclei_model", help="Nuclei model name or path")
    parser.add_argument("--slurm_gpu_type", help="SLURM GPU type, e.g. A30")
    parser.add_argument("--tile_info_path", help="Tile-info file for the live viewer")
    parser.add_argument("--live_update_image_path", help="PNG updated as tiles finish")
    args = parser.parse_args(argv)

    run_distributed_segmentation(
        args.image_path, args.output_dir, args.config_path,
        n_workers=args.n_workers, use_gpu=args.use_gpu,
        chunk_size=args.tile_size, backend=args.backend,
        scheduler_address=args.scheduler_address, tile_overlap=args.tile_overlap,
        cell_model=args.cell_model, nuclei_model=args.nuclei_model,
        slurm={'gpu_type': args.slurm_gpu_type} if args.slurm_gpu_type else None,
        tile_info_path=args.tile_info_path,
        live_update_image_path=args.live_update_image_path,
    )


if __name__ == "__main__":
    main()
