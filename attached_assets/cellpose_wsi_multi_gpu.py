#!/usr/bin/env python3
"""
Optimized Multi-GPU Cellpose Segmentation Script with Outline CSV Export

This script performs efficient parallel segmentation of large microscopy images using Cellpose
across multiple GPUs and exports the object outline coordinates to CSV files.
"""

import os
import sys
import logging
import argparse
import time
import tifffile
import numpy as np
# Import necessary libraries for outline extraction
from skimage import measure
from skimage.segmentation import find_boundaries # Keep for potential future use
from cellpose import models, utils, io
import torch
import torch.multiprocessing as mp
from queue import Empty
import gc
from contextlib import contextmanager
import pandas as pd # Import pandas for CSV export

# --- Default Configuration ---
DEFAULT_IMAGE_PATH = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_combined_af_merged_image.ome.tif"
DEFAULT_OUTPUT_DIR = "output_segmentation_combined"
DEFAULT_CSV_BASE_PATH = "output_segmentation_combined/outlines" # Default base for CSV files
# Default models and their corresponding parameters
DEFAULT_MODEL_TYPES = ['/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700', 'nuclei']
DEFAULT_DIAMETERS = [120.0, 60.0]
DEFAULT_FLOW_THRESHOLDS = [0.4, 0.4]
DEFAULT_CELLPROB_THRESHOLDS = [-1.5, -1.0]
DEFAULT_EVAL_CHANNELS = ["0,0", "0,0"] # Default to grayscale processing [seg_chan, nucleus_chan=None]

DEFAULT_CHUNK_SIZE_MB = 250
DEFAULT_OVERLAP_PIXELS = 60
DEFAULT_PLOT_EVERY_N_CHUNKS = 0 # Disable plotting by default when saving CSVs
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = None

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimized Multi-GPU Cellpose Segmentation with Outline CSV Export')
    # Input/Output
    parser.add_argument('--image_path', type=str, default=DEFAULT_IMAGE_PATH,
                        help='Path to the OME-TIFF image')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save segmentation results (masks, logs)')
    parser.add_argument('--output_csv_base', type=str, default=DEFAULT_CSV_BASE_PATH,
                        help='Base path for saving outline CSV files (e.g., /path/outlines -> /path/outlines_model1.csv)')

    # Model Selection
    parser.add_argument('--model_types', type=str, nargs='+', default=DEFAULT_MODEL_TYPES,
                        help='Cellpose models to use (e.g., cyto3, nuclei, or custom model path)')

    # Cellpose Parameters (per model)
    parser.add_argument('--diameter', type=float, nargs='+', default=DEFAULT_DIAMETERS,
                        help='Diameter parameter for each model type. Use 0.0 for auto-detection.')
    parser.add_argument('--flow_threshold', type=float, nargs='+', default=DEFAULT_FLOW_THRESHOLDS,
                        help='Flow threshold parameter for each model type.')
    parser.add_argument('--cellprob_threshold', type=float, nargs='+', default=DEFAULT_CELLPROB_THRESHOLDS,
                        help='Cell probability threshold parameter for each model type.')
    parser.add_argument('--eval_channels', type=str, nargs='+', default=DEFAULT_EVAL_CHANNELS,
                        help='Channels to use for model evaluation (e.g., "0,0" for grayscale, "1,2" for cyto/nucleus). '
                             'Provide one per model type. Format: "chan1,chan2". ')

    # Processing Parameters
    parser.add_argument('--chunk_size_mb', type=int, default=DEFAULT_CHUNK_SIZE_MB,
                        help='Target memory size for each 2D chunk in MB')
    parser.add_argument('--overlap_pixels', type=int, default=DEFAULT_OVERLAP_PIXELS,
                        help='Overlap between chunks in pixels')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Number of chunks to process in a single batch')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of worker processes (default: auto-detect based on GPUs)')

    # Optional Plotting
    parser.add_argument('--plot_every_n_chunks', type=int, default=DEFAULT_PLOT_EVERY_N_CHUNKS,
                        help='Generate a debug plot every N chunks (0 to disable)')

    args = parser.parse_args()

    # --- Validate Parameter List Lengths ---
    num_models = len(args.model_types)
    param_lengths = {
        'diameter': len(args.diameter),
        'flow_threshold': len(args.flow_threshold),
        'cellprob_threshold': len(args.cellprob_threshold),
        'eval_channels': len(args.eval_channels)
    }
    for param, length in param_lengths.items():
        if length != num_models:
            parser.error(f"Number of values for --{param} ({length}) must match the number of --model_types ({num_models})")

    # --- Parse Eval Channels ---
    try:
        args.parsed_eval_channels = []
        for chan_str in args.eval_channels:
            chans = [int(c.strip()) for c in chan_str.split(',')]
            if len(chans) != 2:
                raise ValueError(f"Invalid format for --eval_channels: '{chan_str}'. Must be 'int,int'.")
            args.parsed_eval_channels.append(chans)
    except ValueError as e:
        parser.error(f"Error parsing --eval_channels: {e}")

    return args

# --- Logging Setup ---
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "segmentation_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(process)d-%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("cellpose_segmentation")
    logger.info(f"Logging to: {log_file_path}")
    return logger

# --- CUDA Memory Management ---
@contextmanager
def torch_cuda_memory_management():
    try: yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); gc.collect()

# --- GPU Detection and SLURM Awareness ---
def get_available_devices():
    devices = []
    slurm_gpus = os.environ.get('SLURM_JOB_GPUS')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if torch.cuda.is_available():
        if slurm_gpus:
            logging.info(f"SLURM allocated GPUs: {slurm_gpus}")
            try:
                if ',' in slurm_gpus: gpu_indices = [int(idx) for idx in slurm_gpus.split(',')]
                elif '-' in slurm_gpus: start, end = map(int, slurm_gpus.split('-')); gpu_indices = list(range(start, end + 1))
                else: gpu_indices = [int(slurm_gpus)]
                for idx in gpu_indices: devices.append(torch.device(f'cuda:{idx}'))
            except ValueError:
                logging.warning(f"Could not parse SLURM_JOB_GPUS: {slurm_gpus}. Using all available GPUs.")
                devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        elif cuda_visible_devices:
            logging.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] # PyTorch re-indexes these
        else:
            devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        logging.info(f"Found {len(devices)} GPU(s). Using: {devices}")
    if not devices:
        devices.append(torch.device('cpu')); logging.info("No GPU found. Using CPU.")
    return devices

# --- Model Initialization ---
def initialize_model_on_device(device, model_type):
    logger = logging.getLogger(f"init_model_{device}")
    logger.info(f"Initializing model '{model_type}' on {device}...")
    try:
        with torch_cuda_memory_management():
            use_gpu = (device.type == 'cuda')
            # Ensure the size model file exists or handle potential errors during init
            model = models.Cellpose(gpu=use_gpu, model_type=model_type, device=device if use_gpu else None)
            if use_gpu: # Dummy eval only needed if GPU involved
                dummy_size = min(int(model.diam_mean * 2) if hasattr(model, 'diam_mean') and model.diam_mean > 0 else 64, 128)
                dummy_eval_img = np.zeros((dummy_size, dummy_size), dtype=np.uint8)
                eval_diam = model.diam_mean if hasattr(model, 'diam_mean') and model.diam_mean > 0 else 30.0
                _ = model.eval(dummy_eval_img, channels=[0,0], diameter=eval_diam, progress=None)
            logger.info(f"Model '{model_type}' initialized successfully on {device}.")
            return model
    except Exception as e:
        logger.error(f"Error initializing model '{model_type}' on {device}: {e}", exc_info=True)
        return None # Return None if initialization fails

# --- Outline Extraction Helper ---
def get_cell_outlines(masks):
    """
    Extracts outline coordinates for each unique object ID in a mask array.
    Uses skimage.measure.find_contours.
    Returns a list of dictionaries, each with 'cell_id', 'x_coords', 'y_coords'.
    Coordinates are relative to the input mask array.
    """
    cells = []
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids != 0] # Exclude background label 0

    for cell_id in cell_ids:
        cell_mask = (masks == cell_id)
        # Find contours for the current cell mask
        contours = measure.find_contours(cell_mask, 0.5) # 0.5 level is standard for binary masks

        if len(contours) > 0:
            # Typically, find_contours returns one contour for a simple object mask
            # We take the first one if multiple are found (e.g., donut shapes)
            contour = contours[0]
            # Note: find_contours returns (row, column), i.e., (y, x)
            cells.append({
                'cell_id': cell_id, # This is the LOCAL ID from the mask
                'y_coords': contour[:, 0], # Row indices
                'x_coords': contour[:, 1], # Column indices
            })
    return cells

# --- Segmentation Worker Process ---
def segmentation_worker(device_id, model_types, task_queue, result_queue, done_event, config):
    process_name = f"worker_gpu{device_id}"
    mp.current_process().name = process_name
    logger = logging.getLogger(process_name)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
    else: device = torch.device('cpu')
    logger.info(f"Worker started on device: {device}")

    models_dict = {}
    for model_type in model_types:
        models_dict[model_type] = initialize_model_on_device(device, model_type)

    if not any(models_dict.values()):
        logger.error(f"No models could be initialized on device {device}. Worker exiting.")
        return

    batch_size = config['batch_size']
    overlap_pixels = config['overlap_pixels']
    model_params = config['model_params']

    while not (done_event.is_set() and task_queue.empty()):
        try:
            tasks = []
            try:
                task = task_queue.get(timeout=5)
                tasks.append(task)
                for _ in range(batch_size - 1):
                    if not task_queue.empty(): tasks.append(task_queue.get_nowait())
                    else: break
            except Empty:
                if done_event.is_set(): break
                continue

            logger.info(f"Processing batch of {len(tasks)} tasks")
            for task in tasks:
                with torch_cuda_memory_management():
                    # Call segment_chunk which now returns outlines too
                    result = segment_chunk(
                        task['chunk_data_yx'], task['z_index'],
                        task['chunk_rel_y_coords'], task['chunk_rel_x_coords'],
                        task['abs_y_start'], task['abs_x_start'],
                        models_dict, model_params, task['output_dir_z'],
                        task['original_chunk_shape_yx'], task['is_first_y_chunk'],
                        task['is_last_y_chunk'], task['is_first_x_chunk'],
                        task['is_last_x_chunk'], overlap_pixels, logger
                    )
                    result_queue.put(result) # Put the extended result onto the queue
                if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
        except Exception as e: logger.error(f"Error in worker process: {e}", exc_info=True)
    logger.info(f"Worker on device {device} shutting down")

# --- Segmentation Function (Modified to return outlines) ---
def segment_chunk(chunk_data_yx, z_index, chunk_rel_y_coords, chunk_rel_x_coords,
                  abs_y_start, abs_x_start, models_dict, model_params,
                  output_dir_z, original_chunk_shape_yx,
                  is_first_y_chunk, is_last_y_chunk, is_first_x_chunk, is_last_x_chunk,
                  overlap_pixels, logger):
    base_filename_prefix = f"z{z_index}_y{abs_y_start}_x{abs_x_start}"
    valid_model = next((m for m in models_dict.values() if m is not None), None)
    device_name = valid_model.device if valid_model else 'N/A (Init Failed?)'
    logger.info(f"Processing chunk {base_filename_prefix} on device {device_name} shape {chunk_data_yx.shape}")

    if chunk_data_yx.ndim != 2:
        logger.error(f"Chunk data {base_filename_prefix} not 2D: {chunk_data_yx.shape}")
        return base_filename_prefix, {}, {}, {} # Return empty dicts for masks, info, outlines

    processed_masks = {}
    processed_outlines = {} # Dictionary to store outlines per model

    for model_name, model in models_dict.items():
        if model is None:
            logger.warning(f"Model '{model_name}' not initialized for {base_filename_prefix}. Skipping.")
            processed_masks[model_name] = None
            processed_outlines[model_name] = [] # Empty list for this model
            continue

        params = model_params.get(model_name)
        if not params:
            logger.error(f"Params not found for model '{model_name}' in {base_filename_prefix}. Skipping.")
            processed_masks[model_name] = None
            processed_outlines[model_name] = []
            continue

        diameter = params['diameter'] if params['diameter'] > 0 else None
        flow_threshold = params['flow_threshold']
        cellprob_threshold = params['cellprob_threshold']
        eval_channels = params['eval_channels']
        logger.debug(f"Using params for '{model_name}': diam={diameter}, flow={flow_threshold}, cellprob={cellprob_threshold}, chans={eval_channels}")

        try:
            masks_raw, flows, styles, diams = model.eval(
                chunk_data_yx, diameter=diameter, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, channels=eval_channels, progress=None
            )

            core_y_start = 0 if is_first_y_chunk else overlap_pixels // 2
            core_y_end = original_chunk_shape_yx[0] + core_y_start
            core_x_start = 0 if is_first_x_chunk else overlap_pixels // 2
            core_x_end = original_chunk_shape_yx[1] + core_x_start
            core_y_end = min(core_y_end, masks_raw.shape[0])
            core_x_end = min(core_x_end, masks_raw.shape[1])

            masks_cropped = masks_raw[core_y_start:core_y_end, core_x_start:core_x_end]

            # --- Extract and Offset Outlines ---
            outlines_relative = get_cell_outlines(masks_cropped)
            outlines_global = []
            for outline_info in outlines_relative:
                # Add absolute offsets to relative coordinates
                global_y = outline_info['y_coords'] + abs_y_start
                global_x = outline_info['x_coords'] + abs_x_start
                outlines_global.append({
                    'local_id': outline_info['cell_id'], # Keep local ID for now
                    'y_coords': global_y,
                    'x_coords': global_x,
                    'z_index': z_index,
                    'chunk_id': base_filename_prefix # Store chunk identifier
                })
            processed_outlines[model_name] = outlines_global
            logger.info(f"Extracted {len(outlines_global)} outlines for '{model_name}' in {base_filename_prefix}")
            # --- End Outline Processing ---

            # Save cropped masks (optional, can be disabled if only CSV needed)
            mask_output_path = os.path.join(output_dir_z, f"{base_filename_prefix}_{os.path.basename(model_name)}_masks.tif")
            tifffile.imwrite(mask_output_path, masks_cropped)
            processed_masks[model_name] = masks_cropped # Keep mask data if needed later

            logger.info(f"Saved '{model_name}' cropped masks for {base_filename_prefix} shape {masks_cropped.shape}")

        except Exception as e:
            logger.error(f"Error segmenting {base_filename_prefix} with '{model_name}': {e}", exc_info=True)
            processed_masks[model_name] = None
            processed_outlines[model_name] = [] # Ensure empty list on error

    chunk_info_for_plot = {
        "z": z_index, "y_abs": abs_y_start, "x_abs": abs_x_start,
        "orig_shape": original_chunk_shape_yx,
        "input_data_for_plot": chunk_data_yx
    }

    # Return outlines along with other info
    return base_filename_prefix, processed_masks, chunk_info_for_plot, processed_outlines

# --- Optional Plotting Function ---
# (No changes needed here, but plotting might be disabled by default now)
def generate_debug_plot(base_filename, chunk_info, processed_masks_dict, model_to_plot,
                        model_params, output_dir, overlap_pixels):
    # ... (plotting code remains the same) ...
    pass # Keep the function definition even if unused

# --- Main Processing Logic (Modified for CSV export) ---
def main():
    args = parse_arguments()
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting segmentation for {args.image_path}...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Outline CSV base path: {args.output_csv_base}")
    # ... (log other parameters) ...
    process_start_time = time.time()

    # --- Load Image ---
    # (Image loading logic remains the same)
    try:
        # ... (image loading code) ...
        with tifffile.TiffFile(args.image_path) as tif:
             # ... (rest of image loading and axis handling) ...
             series_index = np.argmax([s.size for s in tif.series])
             img_full_data = tif.series[series_index].asarray() # Load as numpy array for simplicity here
             axes_order = tif.series[series_index].axes.upper()
             logger.info(f"Image loaded: {args.image_path}, Shape: {img_full_data.shape}, Axes: {axes_order}")
             # --- Axis and Channel Handling ---
             img_processed_data = None
             # ... (axis handling logic to get 3D ZYX numpy array 'img_processed_data') ...
             if axes_order == 'ZYX': img_processed_data = img_full_data
             elif axes_order == 'YX': img_processed_data = np.expand_dims(img_full_data, axis=0)
             elif axes_order == 'ZCYX': img_processed_data = img_full_data[:, 0, :, :]; logger.info(f"Selected C=0 from ZCYX.")
             elif axes_order == 'TZCYX': img_processed_data = img_full_data[0, :, 0, :, :]; logger.info(f"Selected T=0, C=0 from TZCYX.")
             elif axes_order == 'TCYX': img_processed_data = np.expand_dims(img_full_data[0, 0, :, :], axis=0); logger.info(f"Selected T=0, C=0 from TCYX + added Z.")
             elif axes_order == 'CYX': img_processed_data = np.expand_dims(img_full_data[0, :, :], axis=0); logger.info(f"Selected C=0 from CYX + added Z.")
             else: # Attempt guess
                 logger.warning(f"Unknown axes order '{axes_order}'. Attempting ZYX inference.")
                 if img_full_data.ndim == 3: img_processed_data = img_full_data
                 elif img_full_data.ndim == 2: img_processed_data = np.expand_dims(img_full_data, axis=0)
                 elif img_full_data.ndim >= 4: # Simplistic guess: take last 3 dims
                     img_processed_data = img_full_data[tuple([0]*(img_full_data.ndim-3) + [slice(None)]*3)]

             if img_processed_data is None or img_processed_data.ndim != 3:
                 raise ValueError(f"Could not process image into 3D (Z,Y,X). Axes: {axes_order}, Shape: {img_full_data.shape}")
             original_shape_z_y_x = img_processed_data.shape
             logger.info(f"Data for segmentation prepared (Z,Y,X): {original_shape_z_y_x}")

    except FileNotFoundError: logger.error(f"Image file not found: {args.image_path}"); return
    except Exception as e: logger.error(f"Error loading/processing image: {e}", exc_info=True); return

    # --- Calculate Chunking Strategy ---
    # (Chunking logic remains the same)
    bytes_per_pixel = img_processed_data.dtype.itemsize
    pixels_per_chunk_yx_target = (args.chunk_size_mb * 1024 * 1024) / bytes_per_pixel
    chunk_dim_yx_ideal = int(np.sqrt(pixels_per_chunk_yx_target))
    chunk_y_size_no_overlap = max(1, min(chunk_dim_yx_ideal, original_shape_z_y_x[1]))
    chunk_x_size_no_overlap = max(1, min(chunk_dim_yx_ideal, original_shape_z_y_x[2]))
    logger.info(f"Base chunk YX size (no overlap): {chunk_y_size_no_overlap}x{chunk_x_size_no_overlap}")

    # --- Initialize Multiprocessing ---
    available_devices = get_available_devices()
    if not available_devices: logger.error("No processing devices found."); return
    num_devices = len(available_devices)
    num_workers = args.num_workers if args.num_workers is not None else num_devices
    logger.info(f"Using {num_workers} worker processes across {num_devices} devices")

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    done_event = mp.Event()

    # --- Create Model Parameter Dictionary ---
    model_params = {}
    for i, model_type in enumerate(args.model_types):
        model_params[model_type] = {
            'diameter': args.diameter[i], 'flow_threshold': args.flow_threshold[i],
            'cellprob_threshold': args.cellprob_threshold[i], 'eval_channels': args.parsed_eval_channels[i]
        }
    logger.info(f"Model parameters packaged: {model_params}")

    config = {
        'batch_size': args.batch_size, 'overlap_pixels': args.overlap_pixels,
        'plot_every_n_chunks': args.plot_every_n_chunks, 'model_params': model_params
    }

    # --- Start Workers ---
    workers = []
    for i in range(num_workers):
        device_index = i % num_devices
        device_id = available_devices[device_index].index if available_devices[device_index].type == 'cuda' else 'cpu'
        p = mp.Process(target=segmentation_worker, args=(device_id, args.model_types, task_queue, result_queue, done_event, config))
        p.start(); workers.append(p)

    # --- Prepare Tasks ---
    total_chunks_submitted = 0
    # (Task preparation loop remains the same)
    try:
        for z_idx in range(original_shape_z_y_x[0]):
            logger.info(f"Preparing tasks for Z-slice: {z_idx}")
            output_dir_z = os.path.join(args.output_dir, f"Z_{z_idx:03d}")
            os.makedirs(output_dir_z, exist_ok=True)
            for y_start_abs in range(0, original_shape_z_y_x[1], chunk_y_size_no_overlap):
                # ... (calculate chunk boundaries with overlap: eff_y_start, eff_y_end, etc.) ...
                is_first_y_chunk = (y_start_abs == 0)
                y_end_abs = min(y_start_abs + chunk_y_size_no_overlap, original_shape_z_y_x[1])
                is_last_y_chunk = (y_end_abs == original_shape_z_y_x[1])
                for x_start_abs in range(0, original_shape_z_y_x[2], chunk_x_size_no_overlap):
                    is_first_x_chunk = (x_start_abs == 0)
                    x_end_abs = min(x_start_abs + chunk_x_size_no_overlap, original_shape_z_y_x[2])
                    is_last_x_chunk = (x_end_abs == original_shape_z_y_x[2])
                    eff_y_start = max(0, y_start_abs if is_first_y_chunk else y_start_abs - args.overlap_pixels // 2)
                    eff_y_end = min(original_shape_z_y_x[1], y_end_abs if is_last_y_chunk else y_end_abs + args.overlap_pixels // 2)
                    eff_x_start = max(0, x_start_abs if is_first_x_chunk else x_start_abs - args.overlap_pixels // 2)
                    eff_x_end = min(original_shape_z_y_x[2], x_end_abs if is_last_x_chunk else x_end_abs + args.overlap_pixels // 2)

                    chunk_data_with_overlap = img_processed_data[z_idx, eff_y_start:eff_y_end, eff_x_start:eff_x_end].copy()
                    rel_y_start_no_overlap = 0 if is_first_y_chunk else args.overlap_pixels // 2
                    rel_y_end_no_overlap = rel_y_start_no_overlap + (y_end_abs - y_start_abs)
                    rel_x_start_no_overlap = 0 if is_first_x_chunk else args.overlap_pixels // 2
                    rel_x_end_no_overlap = rel_x_start_no_overlap + (x_end_abs - x_start_abs)
                    original_chunk_shape_yx_for_this_chunk = (y_end_abs - y_start_abs, x_end_abs - x_start_abs)

                    task = {
                        'chunk_data_yx': chunk_data_with_overlap, 'z_index': z_idx,
                        'chunk_rel_y_coords': (rel_y_start_no_overlap, rel_y_end_no_overlap),
                        'chunk_rel_x_coords': (rel_x_start_no_overlap, rel_x_end_no_overlap),
                        'abs_y_start': y_start_abs, 'abs_x_start': x_start_abs,
                        'output_dir_z': output_dir_z,
                        'original_chunk_shape_yx': original_chunk_shape_yx_for_this_chunk,
                        'is_first_y_chunk': is_first_y_chunk, 'is_last_y_chunk': is_last_y_chunk,
                        'is_first_x_chunk': is_first_x_chunk, 'is_last_x_chunk': is_last_x_chunk,
                    }
                    task_queue.put(task); total_chunks_submitted += 1
            logger.info(f"Tasks for Z={z_idx} submitted. Total: {total_chunks_submitted}")

        # --- Process Results and Collect Outlines ---
        logger.info(f"All {total_chunks_submitted} tasks submitted. Processing results...")
        total_chunks_completed = 0
        # Initialize lists to store outline fragments from all chunks for each model
        all_outline_fragments = {model_name: [] for model_name in args.model_types}

        while total_chunks_completed < total_chunks_submitted:
            try:
                result_data = result_queue.get(timeout=120)
                # Expecting: base_filename, processed_masks, chunk_info, processed_outlines
                if result_data is None or len(result_data) != 4:
                     logger.warning(f"Received invalid result from queue: {result_data}. Skipping.")
                     continue # Or implement more robust error tracking

                base_filename, processed_masks_dict, chunk_info, processed_outlines_dict = result_data
                total_chunks_completed += 1
                logger.info(f"Completed chunk {total_chunks_completed}/{total_chunks_submitted}: {base_filename}")

                # --- Collect Outline Fragments ---
                for model_name, outlines_list in processed_outlines_dict.items():
                    if model_name in all_outline_fragments:
                        all_outline_fragments[model_name].extend(outlines_list)
                    else:
                        logger.warning(f"Received outlines for unexpected model '{model_name}' in chunk {base_filename}")

                # --- Optional Plotting ---
                if args.plot_every_n_chunks and args.plot_every_n_chunks > 0 and \
                   total_chunks_completed % args.plot_every_n_chunks == 0:
                    # ... (plotting logic can remain, using processed_masks_dict and chunk_info) ...
                    pass

            except Empty:
                all_workers_dead = not any(w.is_alive() for w in workers)
                if all_workers_dead and total_chunks_completed < total_chunks_submitted:
                    logger.error(f"Workers died prematurely. {total_chunks_completed}/{total_chunks_submitted} chunks completed.")
                    break
                else: logger.info(f"Waiting for results... ({total_chunks_completed}/{total_chunks_submitted} completed)")
            except Exception as e: logger.error(f"Error processing result: {e}", exc_info=True)

    except KeyboardInterrupt: logger.info("Keyboard interrupt. Shutting down...")
    except Exception as e: logger.error(f"Error in main process: {e}", exc_info=True)
    finally:
        logger.info("Signaling workers to finish...")
        done_event.set()
        for i, worker in enumerate(workers):
            logger.info(f"Waiting for worker {i} (PID {worker.pid})...")
            worker.join(timeout=60)
            if worker.is_alive(): logger.warning(f"Worker {i} unresponsive. Terminating..."); worker.kill(); worker.join(5)
            logger.info(f"Worker {i} finished with exit code {worker.exitcode}.")

    # --- Post-processing: Assign Global IDs and Save CSVs ---
    logger.info("Processing collected outlines and saving CSVs...")
    output_csv_dir = os.path.dirname(args.output_csv_base)
    if output_csv_dir: os.makedirs(output_csv_dir, exist_ok=True)

    for model_name, outline_fragments in all_outline_fragments.items():
        logger.info(f"Processing outlines for model: {model_name} ({len(outline_fragments)} fragments found)")
        if not outline_fragments:
            logger.warning(f"No outline fragments collected for model '{model_name}'. Skipping CSV.")
            continue

        all_rows_for_df = []
        global_id_counter = 1
        for fragment in outline_fragments:
            local_id = fragment['local_id']
            chunk_id = fragment['chunk_id']
            z_index = fragment['z_index']
            # Assign the current global ID to all points in this fragment
            current_global_id = global_id_counter
            for y, x in zip(fragment['y_coords'], fragment['x_coords']):
                all_rows_for_df.append({
                    'global_id': current_global_id,
                    'x': x,
                    'y': y,
                    'z': z_index, # Add Z index
                    'chunk_local_id': local_id, # Optional: keep local ID for debugging
                    'chunk_id': chunk_id # Optional: keep chunk ID
                })
            global_id_counter += 1 # Increment AFTER processing all points for one fragment

        if not all_rows_for_df:
            logger.warning(f"No valid outline points generated for model '{model_name}'. Skipping CSV.")
            continue

        df_outlines = pd.DataFrame(all_rows_for_df)
        num_unique_objects = df_outlines['global_id'].nunique()
        # Sanitize model name for filename
        safe_model_name = os.path.basename(model_name).replace('.', '_')
        csv_filename = f"{args.output_csv_base}_{safe_model_name}.csv"
        try:
            df_outlines.to_csv(csv_filename, index=False)
            logger.info(f"Saved outlines for '{model_name}' to {csv_filename} ({len(df_outlines)} points, {num_unique_objects} objects)")
        except Exception as e:
            logger.error(f"Failed to save CSV for '{model_name}' to {csv_filename}: {e}", exc_info=True)

    process_end_time = time.time()
    logger.info(f"Segmentation process completed in {process_end_time - process_start_time:.2f} seconds.")
    logger.info(f"Total chunks submitted: {total_chunks_submitted}, Total chunks completed: {total_chunks_completed}")
    if total_chunks_completed < total_chunks_submitted: logger.warning("Potential issue: Not all submitted chunks were processed.")

# --- Create Dummy Test Data ---
# (Dummy data function remains the same)
def create_dummy_test_data(image_path):
    # ... (dummy data creation code) ...
    pass

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError as e: logging.warning(f"MP start method: {e}")
    args_for_setup = parse_arguments()
    if not os.path.exists(args_for_setup.image_path) and args_for_setup.image_path == DEFAULT_IMAGE_PATH:
        # ... (dummy data creation call if needed) ...
        pass
    main()