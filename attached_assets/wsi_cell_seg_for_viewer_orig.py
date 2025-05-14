import dask.array as da
import tifffile
import numpy as np
from cellpose import models, utils
import os
import pandas as pd
from skimage import measure
from skimage.exposure import rescale_intensity
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import random
import argparse

# --- Helper Functions ---
def get_cell_outlines(masks):
    cells = []
    cell_ids = np.unique(masks)[1:]
    for cell_id in cell_ids:
        cell_mask = masks == cell_id
        contours = measure.find_contours(cell_mask, 0.5)
        if len(contours) > 0:
            contour = contours[0]
            cells.append({
                'cell_id': cell_id,
                'x_coords': contour[:, 1],
                'y_coords': contour[:, 0],
            })
    return cells

def segment_chunk(chunk_data, model_instance, chunk_position=None, diameter=80,
                  flow_threshold=0.8, cellprob_threshold=-3.5, channels=[0,0], object_type="object"):
    logger = logging.getLogger(__name__) # Ensure logger is accessible
    if model_instance is None:
        logger.error(f"Model not provided to segment_chunk for {object_type}.")
        return np.array([], dtype=np.uint16), []

    if chunk_data.ndim > 2 and channels == [0,0]:
        if chunk_data.ndim == 3: chunk_data = chunk_data[0]
        elif chunk_data.ndim == 4: chunk_data = chunk_data[0,0]
        else:
            logger.error(f"Cannot handle {object_type} chunk dimension {chunk_data.ndim} for grayscale model.")
            return np.zeros(chunk_data.shape[-2:], dtype=np.uint16), []
    elif chunk_data.ndim < 2:
        logger.error(f"{object_type} chunk has insufficient dimensions at {chunk_position}: shape {chunk_data.shape}")
        return np.array([], dtype=np.uint16), []
    if chunk_data.size == 0:
        logger.warning(f"Empty {object_type} chunk at position {chunk_position}")
        return np.array([], dtype=np.uint16), []
    if any(s < 10 for s in chunk_data.shape[-2:]):
        logger.warning(f"{object_type} chunk spatial dimensions too small at {chunk_position}: shape {chunk_data.shape[-2:]}")
        return np.zeros(chunk_data.shape[-2:], dtype=np.uint16), []

    try:
        masks, flows, styles = model_instance.eval(
            chunk_data,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels
        )
        outlines_info = get_cell_outlines(masks)

        if chunk_position is not None:
            y_offset, x_offset = chunk_position
            for item_info in outlines_info:
                item_info['x_coords'] = np.array(item_info['x_coords']) + x_offset
                item_info['y_coords'] = np.array(item_info['y_coords']) + y_offset
                item_info['chunk_id'] = f"{y_offset}_{x_offset}"
        return masks.astype(np.uint16), outlines_info
    except Exception as e:
        logger.error(f"Error processing {object_type} chunk at {chunk_position} with shape {chunk_data.shape}: {e}", exc_info=True)
        return np.zeros(chunk_data.shape[-2:] if chunk_data.ndim >=2 else (0,0), dtype=np.uint16), []


# --- Visualization Functions ---
def create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices=[0, 1]):
    if image_data_stack.ndim != 3 or image_data_stack.shape[0] < max(vis_ch_indices) + 1 :
        raise ValueError(f"Image stack has shape {image_data_stack.shape}, need channels at indices {vis_ch_indices} from the provided stack.")

    channels_data = []
    for ch_idx in vis_ch_indices:
        channels_data.append(image_data_stack[ch_idx, y_start:y_end, x_start:x_end])

    roi_ch_g_data = channels_data[0]
    roi_ch_b_data = channels_data[1] if len(channels_data) > 1 else np.zeros_like(channels_data[0])
    roi_ch_r_data = channels_data[2] if len(channels_data) > 2 else np.zeros_like(channels_data[0])

    p_low, p_high = 1, 99
    def normalize_channel(channel_data):
        if channel_data.size == 0: return channel_data
        c_min, c_max = np.percentile(channel_data, (p_low, p_high))
        return rescale_intensity(channel_data, in_range=(c_min, c_max if c_max > c_min else c_max + 1e-6), out_range=(0.0, 1.0))

    roi_ch_g_norm = normalize_channel(roi_ch_g_data)
    roi_ch_b_norm = normalize_channel(roi_ch_b_data)
    roi_ch_r_norm = normalize_channel(roi_ch_r_data)

    rgb_image = np.zeros((roi_ch_g_norm.shape[0], roi_ch_g_norm.shape[1], 3), dtype=float)

    if len(vis_ch_indices) > 2: # R, G, B
        rgb_image[..., 0] = roi_ch_r_norm
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_b_norm
    elif len(vis_ch_indices) > 1: # G, B
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_b_norm
    elif len(vis_ch_indices) > 0: # G (grayscale)
        rgb_image[..., 0] = roi_ch_g_norm
        rgb_image[..., 1] = roi_ch_g_norm
        rgb_image[..., 2] = roi_ch_g_norm
    return np.clip(rgb_image, 0, 1)


def visualize_roi_combined(image_data_stack, roi_position, roi_size,
                           df_cells_outlines=None, df_nuclei_outlines=None,
                           vis_ch_indices=[0, 1], figsize=(10, 10),
                           original_bg_channels_for_title=[0,1]):
    logger = logging.getLogger(__name__)
    y_start_param, x_start_param = roi_position
    height_param, width_param = roi_size

    if image_data_stack.ndim != 3:
        logger.error(f"visualize_roi_combined expects 3D (C,Y,X) stack, got {image_data_stack.shape}")
        return None

    img_c, img_h, img_w = image_data_stack.shape

    y_start_roi = max(0, y_start_param)
    x_start_roi = max(0, x_start_param)
    y_end_roi = min(y_start_param + height_param, img_h)
    x_end_roi = min(x_start_param + width_param, img_w)

    actual_roi_height = y_end_roi - y_start_roi
    actual_roi_width = x_end_roi - x_start_roi

    if actual_roi_height <= 0 or actual_roi_width <= 0:
        logger.warning(f"ROI at {roi_position} size {roi_size} results in zero area. Stack shape: {image_data_stack.shape}")
        return None
    try:
        rgb_roi = create_rgb_roi(image_data_stack, y_start_roi, y_end_roi, x_start_roi, x_end_roi, vis_ch_indices)
    except ValueError as e:
        logger.error(f"Error creating RGB ROI: {e}"); return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb_roi)

    ch_map_str_list = []
    if len(original_bg_channels_for_title) == 1:
        ch_map_str_list.append(f"Display: G=OrigCh{original_bg_channels_for_title[0]}")
    elif len(original_bg_channels_for_title) == 2:
        ch_map_str_list.append(f"Display: G=OrigCh{original_bg_channels_for_title[0]}, B=OrigCh{original_bg_channels_for_title[1]}")
    elif len(original_bg_channels_for_title) >= 3:
        ch_map_str_list.append(f"Display: R=OrigCh{original_bg_channels_for_title[0]}, G=OrigCh{original_bg_channels_for_title[1]}, B=OrigCh{original_bg_channels_for_title[2]}")

    title_bg_str = ", ".join(ch_map_str_list) if ch_map_str_list else "N/A"
    title = (f'ROI ({y_start_roi},{x_start_roi}) Size ({actual_roi_width}x{actual_roi_height}) | '
             f'{title_bg_str}')

    outline_info = []
    x_offset_for_plot = x_start_roi
    y_offset_for_plot = y_start_roi

    if df_cells_outlines is not None and not df_cells_outlines.empty:
        roi_df_cells = df_cells_outlines[
            (df_cells_outlines['x'] >= x_start_roi) & (df_cells_outlines['x'] < x_end_roi) &
            (df_cells_outlines['y'] >= y_start_roi) & (df_cells_outlines['y'] < y_end_roi)
        ].copy()
        if not roi_df_cells.empty:
            roi_df_cells['x_rel'] = roi_df_cells['x'] - x_offset_for_plot
            roi_df_cells['y_rel'] = roi_df_cells['y'] - y_offset_for_plot
            for obj_id, group in roi_df_cells.groupby('global_cell_id'):
                ax.plot(group['x_rel'], group['y_rel'], 'r-', linewidth=1.0, alpha=0.7)
            outline_info.append(f"Cells({roi_df_cells['global_cell_id'].nunique()}):Red")

    if df_nuclei_outlines is not None and not df_nuclei_outlines.empty:
        roi_df_nuclei = df_nuclei_outlines[
            (df_nuclei_outlines['x'] >= x_start_roi) & (df_nuclei_outlines['x'] < x_end_roi) &
            (df_nuclei_outlines['y'] >= y_start_roi) & (df_nuclei_outlines['y'] < y_end_roi)
        ].copy()
        if not roi_df_nuclei.empty:
            roi_df_nuclei['x_rel'] = roi_df_nuclei['x'] - x_offset_for_plot
            roi_df_nuclei['y_rel'] = roi_df_nuclei['y'] - y_offset_for_plot
            for obj_id, group in roi_df_nuclei.groupby('global_cell_id'):
                ax.plot(group['x_rel'], group['y_rel'], 'cyan-', linewidth=1.0, alpha=0.7)
            outline_info.append(f"Nuclei({roi_df_nuclei['global_cell_id'].nunique()}):Cyan")

    if outline_info:
        title += " | Outlines: " + ", ".join(outline_info)
    ax.set_title(title, fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    return fig

# --- Main Processing Function ---
def process_large_image(
    image_path,
    csv_base_path,
    chunk_size=(2048, 2048),
    # Cell Model parameters
    cell_model_path="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700",
    cells_diameter=120.0,
    cells_flow_threshold=0.4,
    cells_cellprob_threshold=-1.5,
    cells_channels=[1,2],
    # Nuclei Model parameters
    nuclei_model_path="nuclei",
    nuclei_diameter=40.0,
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=-1.2,
    nuclei_channels=[2,0],
    # Adaptive Nuclei Segmentation parameters
    enable_adaptive_nuclei=False,
    nuclei_adaptive_cellprob_lower_limit=-6.0,
    nuclei_adaptive_cellprob_step_decrement=0.2,
    nuclei_max_adaptive_attempts=3,
    adaptive_nuclei_trigger_ratio=0.05,
    # Visualization parameters
    visualize=True,
    visualize_output_dir=None,
    num_visualize_chunks=5,
    visualize_roi_size=(2024, 2024),
    vis_bg_channel_indices=[0,1],
    # Live update parameters
    live_update_image_path=None,
    tile_info_file_for_viewer=None  # <-- NEW PARAMETER
    ):

    log_file = csv_base_path + '.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    start_time = datetime.now()
    logger.info("="*50)
    logger.info(f"Starting DUAL MODEL image processing at {start_time}")
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output CSV base: {csv_base_path}")
    logger.info(f"Chunk size: {chunk_size}")

    logger.info(f"Cell Model Path: {cell_model_path}")
    logger.info(f"Cells Diameter: {cells_diameter}, Flow: {cells_flow_threshold}, Prob: {cells_cellprob_threshold}, Channels: {cells_channels}")

    logger.info(f"Nuclei Model Path: {nuclei_model_path}")
    logger.info(f"Nuclei Diameter: {nuclei_diameter}, Flow: {nuclei_flow_threshold}, Initial Prob: {nuclei_cellprob_threshold}, Channels: {nuclei_channels}")

    if enable_adaptive_nuclei:
        logger.info("Adaptive Nuclei Segmentation (Cellprob DECREASING): ENABLED")
        logger.info(f"  Cellprob Lower Limit: {nuclei_adaptive_cellprob_lower_limit}, Step Decrement: {nuclei_adaptive_cellprob_step_decrement}, Max Attempts: {nuclei_max_adaptive_attempts}")
        logger.info(f"  Trigger Ratio (nuclei/cells): < {adaptive_nuclei_trigger_ratio}")
    else:
        logger.info("Adaptive Nuclei Segmentation: DISABLED")

    logger.info(f"Visualization Enabled: {visualize}")
    if visualize:
        if visualize_output_dir is None:
            visualize_output_dir = os.path.join(os.path.dirname(csv_base_path) or ".", "visualizations")
        logger.info(f"Visualization Output Dir: {visualize_output_dir}")
        logger.info(f"Num Chunks to Visualize: {num_visualize_chunks}, ROI Size: {visualize_roi_size}")
        logger.info(f"Visualization BG Channels (Original TIFF, 0-indexed): {vis_bg_channel_indices}")

    if live_update_image_path:
        logger.info(f"Live update image will be saved to: {live_update_image_path}")
        live_update_dir = os.path.dirname(live_update_image_path)
        if live_update_dir and not os.path.exists(live_update_dir):
            try:
                os.makedirs(live_update_dir)
                logger.info(f"Created directory for live update image: {live_update_dir}")
            except Exception as e:
                logger.error(f"Could not create directory {live_update_dir} for live update image: {e}. Live updates disabled.")
                live_update_image_path = None

    if tile_info_file_for_viewer: # <-- NEW BLOCK
        logger.info(f"Tile info for live viewer will be written to: {tile_info_file_for_viewer}")
        tile_info_dir = os.path.dirname(tile_info_file_for_viewer)
        if tile_info_dir and not os.path.exists(tile_info_dir):
            try:
                os.makedirs(tile_info_dir)
                logger.info(f"Created directory for tile info file: {tile_info_dir}")
            except Exception as e:
                logger.error(f"Could not create directory {tile_info_dir} for tile info file: {e}. Tile info updates disabled.")
                tile_info_file_for_viewer = None
    logger.info("="*50)


    logger.info("Initializing Cellpose models...")
    try:
        cell_model = models.CellposeModel(pretrained_model=cell_model_path, gpu=True)
        logger.info("Cell model initialized successfully.")
        nuclei_model = models.CellposeModel(pretrained_model=nuclei_model_path, gpu=True)
        logger.info("Nuclei model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize one or more Cellpose models: {e}", exc_info=True)
        return

    logger.info("Reading OME-TIFF file structure...")
    try:
        with tifffile.TiffFile(image_path) as tiff:
            s0 = tiff.series[0]
            s0_shape = s0.shape; s0_ndim = s0.ndim
            logger.info(f"TIFF series 0 shape: {s0_shape}, ndim: {s0_ndim}")
            try:
                data = da.from_zarr(s0.aszarr())
                logger.info(f"Using Zarr store. Dask array chunks: {data.chunks}")
            except Exception as zarr_err:
                logger.warning(f"Could not open as Zarr ({zarr_err}), falling back to asarray.")
                dask_chunks_fb = chunk_size if s0_ndim == 2 else ((1,) + chunk_size if s0_ndim == 3 else ((1,1) + chunk_size if s0_ndim == 4 else 'auto'))
                data = da.from_array(s0.asarray(), chunks=dask_chunks_fb)
                logger.info(f"Using asarray. Dask array chunks: {data.chunks}")
    except FileNotFoundError: logger.error(f"Input image not found: {image_path}"); return
    except Exception as e: logger.error(f"Failed to read TIFF {image_path}: {e}", exc_info=True); return

    logger.info(f"Original data shape: {data.shape}")
    img_shape_yx = data.shape[-2:]
    logger.info(f"Processing based on YX shape: {img_shape_yx}")

    n_chunks_y = (img_shape_yx[0] + chunk_size[0] - 1) // chunk_size[0]
    n_chunks_x = (img_shape_yx[1] + chunk_size[1] - 1) // chunk_size[1]
    total_chunks = n_chunks_y * n_chunks_x
    logger.info(f"Total chunks: {total_chunks} ({n_chunks_y}x{n_chunks_x})")

    final_proc_chunking = chunk_size if data.ndim == 2 else ((data.shape[0],) + chunk_size if data.ndim == 3 else ((data.shape[0], data.shape[1]) + chunk_size if data.ndim == 4 else 'auto'))
    try:
        if final_proc_chunking != 'auto' and data.chunks[-2:] != final_proc_chunking[-2:]:
            logger.info(f"Rechunking data with YX chunks: {chunk_size} -> Dask chunks: {final_proc_chunking}")
            data = data.rechunk(final_proc_chunking)
            logger.info(f"Data rechunked. New Dask array chunks: {data.chunks}")
    except Exception as e: logger.error(f"Failed to rechunk dask array: {e}", exc_info=True)

    all_cells_outlines_info = []
    all_nuclei_outlines_info = []
    processed_chunk_coords_for_vis = []
    global_cell_id_counter = 1
    global_nuclei_id_counter = 1
    processed_chunks_count = 0

    logger.info("Starting chunk processing loop...")
    for y_idx in range(n_chunks_y):
        for x_idx in range(n_chunks_x):
            chunk_start_time = datetime.now()
            processed_chunks_count += 1
            y_start, x_start = y_idx * chunk_size[0], x_idx * chunk_size[1]
            y_end, x_end = min(y_start + chunk_size[0], img_shape_yx[0]), min(x_start + chunk_size[1], img_shape_yx[1])
            current_chunk_height, current_chunk_width = y_end - y_start, x_end - x_start
            if current_chunk_height <=0 or current_chunk_width <=0: continue

            logger.info(f"Processing chunk [{processed_chunks_count}/{total_chunks}] YX: ({y_start},{x_start}) to ({y_end},{x_end})")

            if tile_info_file_for_viewer: # <-- WRITE TILE INFO
                try:
                    # Format: y_start,x_start,tile_h,tile_w,scan_h,scan_w
                    info_line = f"{y_start},{x_start},{current_chunk_height},{current_chunk_width},{img_shape_yx[0]},{img_shape_yx[1]}"
                    with open(tile_info_file_for_viewer, 'w') as tif_viewer_info:
                        tif_viewer_info.write(info_line)
                    # logger.debug(f"Updated tile info file: {tile_info_file_for_viewer} with {info_line}")
                except Exception as e_ti:
                    logger.error(f"Error writing to tile info file {tile_info_file_for_viewer}: {e_ti}")

            if data.ndim == 2: chunk_slice = (slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 3: chunk_slice = (slice(None), slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 4: chunk_slice = (slice(None), slice(None), slice(y_start, y_end), slice(x_start, x_end))
            else: logger.error(f"Cannot slice for data dim {data.ndim}"); continue

            try:
                chunk_data_computed = data[chunk_slice].compute()
                if chunk_data_computed.size == 0 or chunk_data_computed.ndim < 2:
                    logger.warning(f"Chunk data empty or invalid dims at YX ({y_start},{x_start}). Shape: {chunk_data_computed.shape}. Skipping."); continue

                _, cells_in_chunk_global = segment_chunk(
                    chunk_data_computed, cell_model, (y_start, x_start),
                    cells_diameter, cells_flow_threshold, cells_cellprob_threshold, cells_channels, "cell"
                )
                current_nuclei_cellprob = nuclei_cellprob_threshold
                _, nuclei_in_chunk_global = segment_chunk(
                    chunk_data_computed, nuclei_model, (y_start, x_start),
                    nuclei_diameter, nuclei_flow_threshold, current_nuclei_cellprob, nuclei_channels, "nucleus"
                )
                num_cells_found = len(cells_in_chunk_global)
                num_nuclei_found = len(nuclei_in_chunk_global)

                if enable_adaptive_nuclei and num_cells_found > 0:
                    trigger_adaptive = False
                    if num_nuclei_found == 0:
                        trigger_adaptive = True
                        logger.info(f"Chunk [{processed_chunks_count}/{total_chunks}] YX: ({y_start},{x_start}) - {num_cells_found} cells, 0 nuclei. Triggering adaptive nuclei (cellprob DECREASING) segmentation.")
                    elif adaptive_nuclei_trigger_ratio > 0 and (num_nuclei_found / num_cells_found) < adaptive_nuclei_trigger_ratio:
                        trigger_adaptive = True
                        logger.info(f"Chunk [{processed_chunks_count}/{total_chunks}] YX: ({y_start},{x_start}) - Nuclei/Cell ratio ({num_nuclei_found}/{num_cells_found} = {num_nuclei_found/num_cells_found:.2f}) < trigger ({adaptive_nuclei_trigger_ratio:.2f}). Triggering adaptive nuclei (cellprob DECREASING) segmentation.")

                    if trigger_adaptive:
                        adaptive_nuclei_cellprob_current = nuclei_cellprob_threshold
                        for attempt in range(nuclei_max_adaptive_attempts):
                            adaptive_nuclei_cellprob_current -= nuclei_adaptive_cellprob_step_decrement
                            if adaptive_nuclei_cellprob_current < nuclei_adaptive_cellprob_lower_limit:
                                logger.info(f"  Adaptive attempt {attempt+1}: Cellprob threshold ({adaptive_nuclei_cellprob_current:.3f}) would be below lower limit ({nuclei_adaptive_cellprob_lower_limit:.3f}). Stopping adaptive attempts.")
                                break

                            logger.info(f"  Adaptive attempt {attempt+1}/{nuclei_max_adaptive_attempts}: Retrying nuclei segmentation with cellprob_threshold = {adaptive_nuclei_cellprob_current:.3f} (Flow: {nuclei_flow_threshold:.3f})")
                            _, nuclei_in_chunk_adaptive_global = segment_chunk(
                                chunk_data_computed, nuclei_model, (y_start, x_start),
                                nuclei_diameter, nuclei_flow_threshold,
                                adaptive_nuclei_cellprob_current,
                                nuclei_channels, f"nucleus (adaptive cellprob {attempt+1})"
                            )
                            if len(nuclei_in_chunk_adaptive_global) > num_nuclei_found:
                                logger.info(f"  Adaptive attempt {attempt+1} successful: Found {len(nuclei_in_chunk_adaptive_global)} nuclei (previously {num_nuclei_found}) with cellprob_threshold = {adaptive_nuclei_cellprob_current:.3f}")
                                nuclei_in_chunk_global = nuclei_in_chunk_adaptive_global
                                num_nuclei_found = len(nuclei_in_chunk_global)
                                if num_nuclei_found > 0 and (adaptive_nuclei_trigger_ratio == 0 or (num_nuclei_found / num_cells_found) >= adaptive_nuclei_trigger_ratio):
                                    logger.info(f"  Adaptive condition met with {num_nuclei_found} nuclei. Stopping further attempts.")
                                    break
                            else:
                                logger.info(f"  Adaptive attempt {attempt+1}: Found {len(nuclei_in_chunk_adaptive_global)} nuclei (previously {num_nuclei_found}). No improvement or still zero.")
                        else:
                            logger.info(f"  Adaptive nuclei (cellprob DECREASING) segmentation: All {nuclei_max_adaptive_attempts} attempts completed. Final nuclei count for chunk: {num_nuclei_found}")

                for cell_info in cells_in_chunk_global:
                    cell_info['global_cell_id'] = global_cell_id_counter; all_cells_outlines_info.append(cell_info); global_cell_id_counter += 1
                for nuc_info in nuclei_in_chunk_global:
                    nuc_info['global_cell_id'] = global_nuclei_id_counter; all_nuclei_outlines_info.append(nuc_info); global_nuclei_id_counter += 1
                if cells_in_chunk_global or nuclei_in_chunk_global:
                    processed_chunk_coords_for_vis.append((y_start, x_start, f"{y_start}_{x_start}"))

                if live_update_image_path:
                    try:
                        logger.debug(f"Generating live update image for chunk YX ({y_start},{x_start})")
                        live_chunk_display_stack_list = []
                        temp_chunk_for_vis = chunk_data_computed.copy()
                        if temp_chunk_for_vis.ndim == 2: temp_chunk_for_vis = temp_chunk_for_vis[np.newaxis, :, :]

                        target_channel_source = temp_chunk_for_vis
                        if temp_chunk_for_vis.ndim == 4:
                            target_channel_source = temp_chunk_for_vis[0]

                        if target_channel_source.ndim == 3:
                            for ch_idx_orig_tiff in vis_bg_channel_indices:
                                if 0 <= ch_idx_orig_tiff < target_channel_source.shape[0]:
                                    live_chunk_display_stack_list.append(target_channel_source[ch_idx_orig_tiff])
                                else:
                                    logger.warning(f"Live view: Original TIFF channel index {ch_idx_orig_tiff} out of bounds for chunk's {target_channel_source.shape[0]} channels. Using channel 0 of chunk for this slot.")
                                    live_chunk_display_stack_list.append(target_channel_source[0])
                        else:
                             raise ValueError(f"Unsupported chunk_data_computed ndim for live view after T-squeeze: {target_channel_source.ndim}")

                        if not live_chunk_display_stack_list:
                             logger.warning(f"Live view: Could not prepare any channels for display. Skipping live image.")
                        else:
                            live_chunk_display_data = np.stack(live_chunk_display_stack_list, axis=0)
                            temp_cells_relative_rows = [{'global_cell_id': ci['global_cell_id'], 'x': x_cg - x_start, 'y': y_cg - y_start} for ci in cells_in_chunk_global for x_cg, y_cg in zip(ci['x_coords'], ci['y_coords'])]
                            df_temp_cells_live = pd.DataFrame(temp_cells_relative_rows)
                            temp_nuclei_relative_rows = [{'global_cell_id': ni['global_cell_id'], 'x': x_ng - x_start, 'y': y_ng - y_start} for ni in nuclei_in_chunk_global for x_ng, y_ng in zip(ni['x_coords'], ni['y_coords'])]
                            df_temp_nuclei_live = pd.DataFrame(temp_nuclei_relative_rows)

                            live_fig = visualize_roi_combined(
                                image_data_stack=live_chunk_display_data,
                                roi_position=(0, 0),
                                roi_size=(current_chunk_height, current_chunk_width),
                                df_cells_outlines=df_temp_cells_live,
                                df_nuclei_outlines=df_temp_nuclei_live,
                                vis_ch_indices=list(range(live_chunk_display_data.shape[0])),
                                original_bg_channels_for_title=vis_bg_channel_indices,
                                figsize=(8,8)
                            )
                            if live_fig:
                                live_fig.savefig(live_update_image_path, dpi=100, bbox_inches='tight')
                                plt.close(live_fig); logger.debug(f"Live update image saved to {live_update_image_path}")
                    except Exception as live_e: logger.error(f"Error generating live update image for YX ({y_start},{x_start}): {live_e}", exc_info=False)

                chunk_duration = (datetime.now() - chunk_start_time).total_seconds()
                logger.info(f"Chunk [{processed_chunks_count}/{total_chunks}] done in {chunk_duration:.2f}s. Found {len(cells_in_chunk_global)} cells, {len(nuclei_in_chunk_global)} nuclei.")
            except Exception as e: logger.error(f"Error processing chunk YX ({y_start},{x_start}): {e}", exc_info=True)

    df_cells = pd.DataFrame()
    if all_cells_outlines_info:
        df_cells_rows = [{'global_cell_id': ci['global_cell_id'], 'chunk_id': ci['chunk_id'], 'x': x, 'y': y} for ci in all_cells_outlines_info for x, y in zip(ci['x_coords'], ci['y_coords'])]
        df_cells = pd.DataFrame(df_cells_rows)
        cells_csv_path = csv_base_path + "_cells.csv"
        try: df_cells.to_csv(cells_csv_path, index=False); logger.info(f"Cell outlines saved: {cells_csv_path} ({len(df_cells)} pts, {df_cells['global_cell_id'].nunique() if not df_cells.empty else 0} cells)")
        except Exception as e: logger.error(f"Failed to save cells CSV: {e}")
    else: logger.warning("No cells found. Cells CSV not created.")

    df_nuclei = pd.DataFrame()
    if all_nuclei_outlines_info:
        df_nuclei_rows = [{'global_cell_id': ni['global_cell_id'], 'chunk_id': ni['chunk_id'], 'x': x, 'y': y} for ni in all_nuclei_outlines_info for x, y in zip(ni['x_coords'], ni['y_coords'])]
        df_nuclei = pd.DataFrame(df_nuclei_rows)
        nuclei_csv_path = csv_base_path + "_nuclei.csv"
        try: df_nuclei.to_csv(nuclei_csv_path, index=False); logger.info(f"Nuclei outlines saved: {nuclei_csv_path} ({len(df_nuclei)} pts, {df_nuclei['global_cell_id'].nunique() if not df_nuclei.empty else 0} nuclei)")
        except Exception as e: logger.error(f"Failed to save nuclei CSV: {e}")
    else: logger.warning("No nuclei found. Nuclei CSV not created.")

    vis_img_stack_bg = None
    if visualize and visualize_output_dir and processed_chunk_coords_for_vis:
        logger.info("--- Starting End-of-Run Visualization Step ---")
        if not os.path.exists(visualize_output_dir):
            try: os.makedirs(visualize_output_dir); logger.info(f"Created vis dir: {visualize_output_dir}")
            except Exception as e: logger.error(f"Could not create vis dir {visualize_output_dir}: {e}"); visualize = False
        if visualize and (df_cells.empty and df_nuclei.empty and num_visualize_chunks > 0):
             logger.warning("End-of-run Vis: Outlines DataFrames empty, cannot generate visualizations with outlines.")

        if visualize:
            num_to_sel = min(num_visualize_chunks, len(processed_chunk_coords_for_vis))
            if num_to_sel > 0:
                selected_vis_chunks_indices = random.sample(range(len(processed_chunk_coords_for_vis)), num_to_sel)
                selected_vis_chunks = [processed_chunk_coords_for_vis[i] for i in selected_vis_chunks_indices]
                logger.info(f"Selected {num_to_sel} chunks for summary visualization: {[c[2] for c in selected_vis_chunks]}")
                logger.info(f"Reloading image channels (original 0-indexed: {vis_bg_channel_indices}) for summary visualization background...")
                try:
                    with tifffile.TiffFile(image_path) as tiff_vis:
                        series0 = tiff_vis.series[0]; s0_shape = series0.shape; s0_ndim = series0.ndim
                        if s0_ndim < 3 and not (s0_ndim == 2 and len(vis_bg_channel_indices)==1 and vis_bg_channel_indices[0]==0) :
                             logger.error(f"Summary Vis BG needs >= 3D (C,Y,X) or 2D with vis_bg_channel_indices=[0], got shape {s0_shape} and vis_bg_channel_indices={vis_bg_channel_indices}"); visualize=False
                        else:
                            channel_axis_for_vis = 0 if s0_ndim == 3 else (1 if s0_ndim == 4 else -1)
                            if channel_axis_for_vis == -1 and s0_ndim !=2 : logger.error(f"Cannot determine channel axis for summary vis from shape {s0_shape}"); visualize=False

                            if visualize:
                                loaded_bg_channels_list = []
                                for ch_idx_orig in vis_bg_channel_indices:
                                    if s0_ndim == 2:
                                        if ch_idx_orig != 0:
                                            logger.error(f"Summary Vis BG: For 2D image, only original channel index 0 is valid, got {ch_idx_orig}."); visualize=False; break
                                        page = series0.asarray()
                                    else:
                                        if not (0 <= ch_idx_orig < s0_shape[channel_axis_for_vis]):
                                            logger.error(f"Summary Vis BG channel index {ch_idx_orig} out of bounds for axis {channel_axis_for_vis} (size {s0_shape[channel_axis_for_vis]})"); visualize=False; break
                                        key_list = [slice(None)] * s0_ndim; key_list[channel_axis_for_vis] = ch_idx_orig
                                        if s0_ndim == 4: key_list[0] = 0
                                        page = np.squeeze(series0.asarray(key=tuple(key_list)))

                                    if page.ndim == 2: loaded_bg_channels_list.append(page)
                                    else: logger.error(f"Loaded summary vis page for ch {ch_idx_orig} not 2D: {page.shape}"); visualize=False; break

                                if visualize and len(loaded_bg_channels_list) == len(vis_bg_channel_indices):
                                    vis_img_stack_bg = np.stack(loaded_bg_channels_list, axis=0)
                                    logger.info(f"Loaded summary vis BG stack from original channels {vis_bg_channel_indices}: {vis_img_stack_bg.shape}")
                                elif visualize: logger.warning(f"Failed to load all {len(vis_bg_channel_indices)} specified summary vis BG channels."); visualize=False
                except Exception as e: logger.error(f"Failed to reload image for summary vis: {e}", exc_info=True); visualize = False
            else: logger.warning("No processed chunks with objects for summary visualization.")

            if visualize and vis_img_stack_bg is not None:
                for y_s_chunk_global, x_s_chunk_global, chunk_id_str in selected_vis_chunks:
                    logger.info(f"Generating summary visualization for ROI near chunk area: {chunk_id_str}")
                    c_center_y = y_s_chunk_global + chunk_size[0]//2; c_center_x = x_s_chunk_global + chunk_size[1]//2
                    vis_y_s_global = max(0, c_center_y - visualize_roi_size[0]//2); vis_x_s_global = max(0, c_center_x - visualize_roi_size[1]//2)
                    fig = visualize_roi_combined(
                        image_data_stack=vis_img_stack_bg,
                        roi_position=(vis_y_s_global, vis_x_s_global),
                        roi_size=visualize_roi_size,
                        df_cells_outlines=df_cells,
                        df_nuclei_outlines=df_nuclei,
                        vis_ch_indices=list(range(vis_img_stack_bg.shape[0])),
                        original_bg_channels_for_title=vis_bg_channel_indices
                    )
                    if fig:
                        try:
                            plot_fn = os.path.join(visualize_output_dir, f"summary_visualization_chunk_{chunk_id_str}_roi_{vis_y_s_global}_{vis_x_s_global}.png")
                            fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
                            logger.info(f"Saved summary visualization: {plot_fn}"); plt.close(fig)
                        except Exception as e: logger.error(f"Failed to save summary plot for {chunk_id_str}: {e}"); plt.close(fig)
                    else: logger.warning(f"Skipped saving summary plot for {chunk_id_str}, figure gen failed.")
            elif visualize: logger.warning("Summary Vis enabled, but BG stack not prepared or no chunks to show. Skipping summary plots.")

    total_duration = (datetime.now() - start_time).total_seconds()
    logger.info("="*50); logger.info(f"Processing finished at {datetime.now()}"); logger.info(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f}m)")
    logger.info(f"Total cells: {df_cells['global_cell_id'].nunique() if not df_cells.empty else 0}, Total nuclei: {df_nuclei['global_cell_id'].nunique() if not df_nuclei.empty else 0}")
    logger.info(f"Log file: {log_file}")
    if visualize and visualize_output_dir and num_visualize_chunks > 0 and (vis_img_stack_bg is not None or not processed_chunk_coords_for_vis):
        logger.info(f"Summary visualizations potentially saved in: {visualize_output_dir}")
    if live_update_image_path: logger.info(f"Last live update image saved to: {live_update_image_path}")
    if tile_info_file_for_viewer: logger.info(f"Last tile info for viewer written to: {tile_info_file_for_viewer}") # <-- Log new file
    logger.info("="*50)

# --- Argument Parsing Helper ---
def list_of_ints(string):
    try:
        values = list(map(int, string.split()))
        if not (1 <= len(values) <= 3): raise argparse.ArgumentTypeError("Must be 1-3 integers for vis_bg_channel_indices")
        return values
    except ValueError: raise argparse.ArgumentTypeError("Channel indices must be integers")

def list_of_two_ints(string):
    try:
        values = list(map(int, string.split()))
        if len(values) != 2: raise argparse.ArgumentTypeError("Model channels must be two integers")
        return values
    except ValueError: raise argparse.ArgumentTypeError("Model channels must be two integers")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DUAL Cellpose segmentation with adaptive nuclei (cellprob DECREASING) option and live view output.")
    parser.add_argument("image_path", help="Path to input OME-TIFF image.")
    parser.add_argument("csv_base_path", help="Base path for output CSV files.")

    parser.add_argument("--cell_model_path", default="cyto", help="Cell model path. Default: cyto")
    parser.add_argument("--nuclei_model_path", default="nuclei", help="Nuclei model path. Default: nuclei")
    parser.add_argument("--chunk_size", type=int, nargs=2, default=[2048, 2048], metavar=('Y', 'X'), help="Chunk size. Default: 2048 2048")

    parser.add_argument("--cells_diameter", type=float, default=120.0, help="Cell diameter. Default: 120.0")
    parser.add_argument("--cells_flow_threshold", type=float, default=0.4, help="Cell flow_threshold. Default: 0.4")
    parser.add_argument("--cells_cellprob_threshold", type=float, default=-1.5, help="Cell cellprob_threshold. Default: -1.5")
    parser.add_argument("--cells_channels", type=list_of_two_ints, default="1 2", help="Cell model channels. Default: '1 2'")

    parser.add_argument("--nuclei_diameter", type=float, default=60.0, help="Nuclei diameter. Default: 60.0")
    parser.add_argument("--nuclei_flow_threshold", type=float, default=0.4, help="Nuclei flow_threshold (constant for adaptive). Default: 0.4")
    parser.add_argument("--nuclei_cellprob_threshold", type=float, default=-1.5, help="Initial nuclei cellprob_threshold. Default: -1.5")
    parser.add_argument("--nuclei_channels", type=list_of_two_ints, default="2 1", help="Nuclei model channels. Default: '2 1'")

    parser.add_argument("--enable_adaptive_nuclei", action='store_true', help="Enable adaptive nuclei segmentation by DECREASING cellprob_threshold.")
    parser.add_argument("--nuclei_adaptive_cellprob_lower_limit", type=float, default=-6.0, help="Lower limit for adaptive nuclei cellprob_threshold (more negative). Default: -6.0")
    parser.add_argument("--nuclei_adaptive_cellprob_step_decrement", type=float, default=0.5, help="Step to DECREMENT cellprob_threshold in adaptive attempts. Default: 0.5")
    parser.add_argument("--nuclei_max_adaptive_attempts", type=int, default=3, help="Max adaptive attempts for nuclei. Default: 3")
    parser.add_argument("--adaptive_nuclei_trigger_ratio", type=float, default=0.05, help="Trigger adaptive nuclei if (nuclei/cell count) < ratio. 0 for nuclei_count=0. Default: 0.05")

    parser.add_argument("--visualize", action='store_true', help="Enable end-of-run summary visualization.")
    parser.add_argument("--visualize_output_dir", default=None, help="Dir for summary visualization. Default: 'visualizations' subdir.")
    parser.add_argument("--num_visualize_chunks", type=int, default=5, help="Number of chunks for summary visualization. Default: 5")
    parser.add_argument("--visualize_roi_size", type=int, nargs=2, default=[512, 512], metavar=('Y', 'X'), help="Summary visualization ROI size. Default: 512 512")
    parser.add_argument("--vis_bg_channel_indices", type=list_of_ints, default="0 1", help="Original TIFF 0-indexed channels for vis BG (1-3 ints). Default: '0 1'")

    parser.add_argument("--live_update_image_path", type=str, default=None, help="Path to save live update image (e.g., /path/to/live_view.png).")
    parser.add_argument("--tile_info_file_for_viewer", type=str, default=None, help="Path to file for writing current tile info for the live scan viewer (e.g., current_tile_info.txt).") # <-- NEW ARGUMENT

    args = parser.parse_args()

    if not (1 <= len(args.vis_bg_channel_indices) <= 3):
        parser.error("--vis_bg_channel_indices must provide 1, 2, or 3 integer channel indices.")

    process_large_image(
        image_path=args.image_path,
        csv_base_path=args.csv_base_path,
        chunk_size=tuple(args.chunk_size),
        cell_model_path=args.cell_model_path,
        cells_diameter=args.cells_diameter,
        cells_flow_threshold=args.cells_flow_threshold,
        cells_cellprob_threshold=args.cells_cellprob_threshold,
        cells_channels=args.cells_channels,
        nuclei_model_path=args.nuclei_model_path,
        nuclei_diameter=args.nuclei_diameter,
        nuclei_flow_threshold=args.nuclei_flow_threshold,
        nuclei_cellprob_threshold=args.nuclei_cellprob_threshold,
        nuclei_channels=args.nuclei_channels,
        enable_adaptive_nuclei=args.enable_adaptive_nuclei,
        nuclei_adaptive_cellprob_lower_limit=args.nuclei_adaptive_cellprob_lower_limit,
        nuclei_adaptive_cellprob_step_decrement=args.nuclei_adaptive_cellprob_step_decrement,
        nuclei_max_adaptive_attempts=args.nuclei_max_adaptive_attempts,
        adaptive_nuclei_trigger_ratio=args.adaptive_nuclei_trigger_ratio,
        visualize=args.visualize,
        visualize_output_dir=args.visualize_output_dir,
        num_visualize_chunks=args.num_visualize_chunks,
        visualize_roi_size=tuple(args.visualize_roi_size),
        vis_bg_channel_indices=args.vis_bg_channel_indices,
        live_update_image_path=args.live_update_image_path,
        tile_info_file_for_viewer=args.tile_info_file_for_viewer # <-- PASS NEW ARGUMENT
    )
    print("\n--- Python Script Execution Finished ---")