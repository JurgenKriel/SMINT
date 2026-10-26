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
    """
    Extract cell outlines and their coordinates from a mask
    Returns a list of dictionaries containing cell ID and outline coordinates
    """
    cells = []
    cell_ids = np.unique(masks)[1:] # Exclude background = 0

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
    """
    Segment a single chunk using a given cellpose model and extract outlines.
    """
    if model_instance is None:
        print(f"Error: Model not provided to segment_chunk for {object_type}.")
        return np.array([], dtype=np.uint16), []

    # Preprocessing (simplified, Cellpose handles most)
    if chunk_data.ndim > 2 and channels == [0,0]: # Grayscale model, multi-channel input
        print(f"Warning: {object_type} chunk at {chunk_position} has {chunk_data.ndim} dims, model expects grayscale. Using first channel.")
        if chunk_data.ndim == 3: chunk_data = chunk_data[0] # C,Y,X -> Y,X
        elif chunk_data.ndim == 4: chunk_data = chunk_data[0,0] # T,C,Y,X -> Y,X
        else:
            print(f"Error: Cannot handle {object_type} chunk dimension {chunk_data.ndim} for grayscale model.")
            return np.zeros(chunk_data.shape[-2:], dtype=np.uint16), []
    elif chunk_data.ndim < 2:
        print(f"{object_type} chunk has insufficient dimensions at {chunk_position}: shape {chunk_data.shape}")
        return np.array([], dtype=np.uint16), []
    if chunk_data.size == 0:
        print(f"Empty {object_type} chunk at position {chunk_position}")
        return np.array([], dtype=np.uint16), []
    if any(s < 10 for s in chunk_data.shape[-2:]):
        print(f"{object_type} chunk spatial dimensions too small at {chunk_position}: shape {chunk_data.shape}")
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
                item_info['x_coords'] += x_offset
                item_info['y_coords'] += y_offset
                item_info['chunk_id'] = f"{y_offset}_{x_offset}"
        return masks.astype(np.uint16), outlines_info
    except Exception as e:
        print(f"Error processing {object_type} chunk at {chunk_position} with shape {chunk_data.shape}: {e}")
        return np.zeros(chunk_data.shape[-2:] if chunk_data.ndim >=2 else (0,0), dtype=np.uint16), []

# --- Visualization Functions ---

def create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices=[0, 1]):
    if image_data_stack.ndim != 3 or image_data_stack.shape[0] < len(vis_ch_indices):
        raise ValueError(f"Image stack has shape {image_data_stack.shape}, need {len(vis_ch_indices)} channels at indices {vis_ch_indices}.")
    roi_ch_g = image_data_stack[vis_ch_indices[0], y_start:y_end, x_start:x_end]
    roi_ch_b = image_data_stack[vis_ch_indices[1], y_start:y_end, x_start:x_end]
    p_low, p_high = 1, 99
    g_min, g_max = np.percentile(roi_ch_g, (p_low, p_high))
    b_min, b_max = np.percentile(roi_ch_b, (p_low, p_high))
    roi_ch_g_norm = rescale_intensity(roi_ch_g, in_range=(g_min, g_max if g_max > g_min else g_max + 1e-6), out_range=(0.0, 1.0))
    roi_ch_b_norm = rescale_intensity(roi_ch_b, in_range=(b_min, b_max if b_max > b_min else b_max + 1e-6), out_range=(0.0, 1.0))
    rgb_image = np.zeros((roi_ch_g_norm.shape[0], roi_ch_g_norm.shape[1], 3), dtype=float)
    rgb_image[..., 1] = roi_ch_g_norm
    rgb_image[..., 2] = roi_ch_b_norm
    return np.clip(rgb_image, 0, 1)

def visualize_roi_combined(image_data_stack, roi_position, roi_size,
                           df_cells_outlines=None, df_nuclei_outlines=None,
                           vis_ch_indices=[0, 1], figsize=(10, 10),
                           original_bg_channels=[0,1]):
    y_start, x_start = roi_position
    height, width = roi_size
    if image_data_stack.ndim != 3:
        print(f"Error: visualize_roi_combined expects 3D (C,Y,X) stack, got {image_data_stack.shape}")
        return None
    img_c, img_h, img_w = image_data_stack.shape
    y_end = min(y_start + height, img_h)
    x_end = min(x_start + width, img_w)
    y_start = max(0, y_end - height)
    x_start = max(0, x_end - width)
    actual_height, actual_width = y_end - y_start, x_end - x_start

    if actual_height <= 0 or actual_width <= 0:
        print(f"Warning: ROI at {roi_position} size {roi_size} zero area after clipping.")
        return None
    try:
        rgb_roi = create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices)
    except ValueError as e:
        print(f"Error creating RGB ROI: {e}"); return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb_roi)
    title = (f'ROI ({y_start},{x_start}) Size ({actual_width}x{actual_height}) | '
             f'BG: G=OrigCh{original_bg_channels[0]}, B=OrigCh{original_bg_channels[1]}')

    outline_info = []
    if df_cells_outlines is not None and not df_cells_outlines.empty:
        roi_df_cells = df_cells_outlines[
            (df_cells_outlines['x'] >= x_start) & (df_cells_outlines['x'] < x_end) &
            (df_cells_outlines['y'] >= y_start) & (df_cells_outlines['y'] < y_end)
        ].copy()
        if not roi_df_cells.empty:
            roi_df_cells['x_rel'] = roi_df_cells['x'] - x_start
            roi_df_cells['y_rel'] = roi_df_cells['y'] - y_start
            for obj_id, group in roi_df_cells.groupby('global_cell_id'):
                ax.plot(group['x_rel'], group['y_rel'], 'r-', linewidth=1.0, alpha=0.7) # Cells in Red
            outline_info.append("Cells:Red")

    if df_nuclei_outlines is not None and not df_nuclei_outlines.empty:
        roi_df_nuclei = df_nuclei_outlines[
            (df_nuclei_outlines['x'] >= x_start) & (df_nuclei_outlines['x'] < x_end) &
            (df_nuclei_outlines['y'] >= y_start) & (df_nuclei_outlines['y'] < y_end)
        ].copy()
        if not roi_df_nuclei.empty:
            roi_df_nuclei['x_rel'] = roi_df_nuclei['x'] - x_start
            roi_df_nuclei['y_rel'] = roi_df_nuclei['y'] - y_start
            for obj_id, group in roi_df_nuclei.groupby('global_cell_id'): # Assuming 'global_cell_id' for nuclei too
                ax.plot(group['x_rel'], group['y_rel'], 'cyan-', linewidth=1.0, alpha=0.7) # Nuclei in Cyan
            outline_info.append("Nuclei:Cyan")

    if outline_info:
        title += " | Outlines: " + ", ".join(outline_info)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig

# --- Main Processing Function ---
def process_large_image(
    image_path,
    csv_base_path, # Base path for output CSVs
    chunk_size=(2048, 2048),
    # Cell Model parameters
    cell_model_path="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700",
    cells_diameter=120.0,
    cells_flow_threshold=0.4,
    cells_cellprob_threshold=-1.5,
    cells_channels=[1,2],
    # Nuclei Model parameters
    nuclei_model_path="nuclei",
    nuclei_diameter=60.0,
    nuclei_flow_threshold=0.4,
    nuclei_cellprob_threshold=-1.5,
    nuclei_channels=[2,1],
    # Visualization parameters
    visualize=True,
    visualize_output_dir=None,
    num_visualize_chunks=5,
    visualize_roi_size=(512, 512),
    vis_bg_channel_indices=[0,1]
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
    logger.info(f"Nuclei Diameter: {nuclei_diameter}, Flow: {nuclei_flow_threshold}, Prob: {nuclei_cellprob_threshold}, Channels: {nuclei_channels}")

    logger.info(f"Visualization Enabled: {visualize}")
    if visualize:
        if visualize_output_dir is None:
            visualize_output_dir = os.path.dirname(csv_base_path) # Default to base CSV dir
        logger.info(f"Visualization Output Dir: {visualize_output_dir}")
        logger.info(f"Num Chunks to Visualize: {num_visualize_chunks}, ROI Size: {visualize_roi_size}")
        logger.info(f"Visualization BG Channels (Original TIFF): {vis_bg_channel_indices}")
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
            try:
                zarr_store = tiff.series[0].aszarr()
                data = da.from_zarr(zarr_store)
                logger.info("Using Zarr store for reading.")
            except Exception as zarr_err:
                logger.warning(f"Could not open as Zarr store ({zarr_err}), falling back to asarray.")
                data = da.from_array(tiff.series[0].asarray(), chunks=(1,1) + chunk_size if tiff.series[0].ndim > 2 else chunk_size ) # Basic chunking for dask
    except FileNotFoundError:
        logger.error(f"Input image file not found: {image_path}"); return
    except Exception as e:
        logger.error(f"Failed to read TIFF file {image_path}: {e}", exc_info=True); return

    logger.info(f"Original data shape: {data.shape}")
    img_shape_yx = data.shape[-2:]
    logger.info(f"Processing based on YX shape: {img_shape_yx}")

    n_chunks_y = (img_shape_yx[0] + chunk_size[0] - 1) // chunk_size[0]
    n_chunks_x = (img_shape_yx[1] + chunk_size[1] - 1) // chunk_size[1]
    total_chunks = n_chunks_y * n_chunks_x
    logger.info(f"Total number of chunks to process: {total_chunks} ({n_chunks_y}x{n_chunks_x})")

    # Rechunking based on data dimensions
    if data.ndim == 2: final_chunking = chunk_size # Y, X
    elif data.ndim == 3: final_chunking = (data.shape[0],) + chunk_size # C, Y, X
    elif data.ndim == 4: final_chunking = (data.shape[0], data.shape[1]) + chunk_size # T, C, Y, X
    else: final_chunking = 'auto'; logger.warning(f"Data dim {data.ndim} not explicitly handled for rechunking.")

    try:
        if final_chunking != 'auto': # Only rechunk if we have a specific strategy
            logger.info(f"Rechunking data with YX chunks: {chunk_size} -> Dask chunks: {final_chunking}")
            data = data.rechunk(final_chunking)
    except Exception as e:
        logger.error(f"Failed to rechunk dask array: {e}", exc_info=True) # Continue if rechunk fails, might be less optimal

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

            if data.ndim == 2: chunk_slice = (slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 3: chunk_slice = (slice(None), slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 4: chunk_slice = (slice(None), slice(None), slice(y_start, y_end), slice(x_start, x_end))
            else: logger.error(f"Cannot slice for data dim {data.ndim}"); continue

            logger.info(f"Processing chunk [{processed_chunks_count}/{total_chunks}] YX: ({y_start},{x_start})")
            try:
                chunk_data_computed = data[chunk_slice].compute()
                if chunk_data_computed.size == 0:
                    logger.warning(f"Computed chunk data empty at YX ({y_start},{x_start}). Skipping."); continue

                # Segment Cells
                _, cells_in_chunk = segment_chunk(
                    chunk_data_computed, cell_model, (y_start, x_start),
                    cells_diameter, cells_flow_threshold, cells_cellprob_threshold, cells_channels, "cell"
                )
                for cell_info in cells_in_chunk:
                    cell_info['global_cell_id'] = global_cell_id_counter
                    all_cells_outlines_info.append(cell_info)
                    global_cell_id_counter += 1

                # Segment Nuclei
                _, nuclei_in_chunk = segment_chunk(
                    chunk_data_computed, nuclei_model, (y_start, x_start),
                    nuclei_diameter, nuclei_flow_threshold, nuclei_cellprob_threshold, nuclei_channels, "nucleus"
                )
                for nuc_info in nuclei_in_chunk:
                    nuc_info['global_cell_id'] = global_nuclei_id_counter # Use same key for consistency in DF
                    all_nuclei_outlines_info.append(nuc_info)
                    global_nuclei_id_counter += 1

                if cells_in_chunk or nuclei_in_chunk: # If anything was found
                    processed_chunk_coords_for_vis.append((y_start, x_start, f"{y_start}_{x_start}"))

                chunk_duration = (datetime.now() - chunk_start_time).total_seconds()
                logger.info(f"Chunk [{processed_chunks_count}/{total_chunks}] done in {chunk_duration:.2f}s. Found {len(cells_in_chunk)} cells, {len(nuclei_in_chunk)} nuclei.")
            except Exception as e:
                logger.error(f"Error processing chunk YX ({y_start},{x_start}): {e}", exc_info=True)

    # --- Create and Save DataFrames ---
    df_cells = pd.DataFrame()
    if all_cells_outlines_info:
        df_cells_rows = []
        for cell_info in all_cells_outlines_info:
            for x, y in zip(cell_info['x_coords'], cell_info['y_coords']):
                df_cells_rows.append({'global_cell_id': cell_info['global_cell_id'], 'chunk_id': cell_info['chunk_id'], 'x': x, 'y': y})
        df_cells = pd.DataFrame(df_cells_rows)
        cells_csv_path = csv_base_path + "_cells.csv"
        try:
            df_cells.to_csv(cells_csv_path, index=False)
            logger.info(f"Cell outlines saved to: {cells_csv_path} ({len(df_cells)} points, {global_cell_id_counter-1} cells)")
        except Exception as e: logger.error(f"Failed to save cells CSV: {e}")
    else: logger.warning("No cells found. Cells CSV not created.")

    df_nuclei = pd.DataFrame()
    if all_nuclei_outlines_info:
        df_nuclei_rows = []
        for nuc_info in all_nuclei_outlines_info:
            for x, y in zip(nuc_info['x_coords'], nuc_info['y_coords']):
                df_nuclei_rows.append({'global_cell_id': nuc_info['global_cell_id'], 'chunk_id': nuc_info['chunk_id'], 'x': x, 'y': y})
        df_nuclei = pd.DataFrame(df_nuclei_rows)
        nuclei_csv_path = csv_base_path + "_nuclei.csv"
        try:
            df_nuclei.to_csv(nuclei_csv_path, index=False)
            logger.info(f"Nuclei outlines saved to: {nuclei_csv_path} ({len(df_nuclei)} points, {global_nuclei_id_counter-1} nuclei)")
        except Exception as e: logger.error(f"Failed to save nuclei CSV: {e}")
    else: logger.warning("No nuclei found. Nuclei CSV not created.")

    # --- Optional Visualization ---
    vis_img_stack_bg = None
    if visualize and visualize_output_dir and processed_chunk_coords_for_vis:
        logger.info("--- Starting Visualization Step ---")
        if not os.path.exists(visualize_output_dir):
            try: os.makedirs(visualize_output_dir); logger.info(f"Created vis dir: {visualize_output_dir}")
            except Exception as e: logger.error(f"Could not create vis dir {visualize_output_dir}: {e}"); visualize = False

        if visualize and (df_cells.empty and df_nuclei.empty and num_visualize_chunks > 0):
             logger.warning("Outlines DataFrames are empty, cannot generate visualizations with outlines.")
             # visualize = False # Or proceed to show background only

        if visualize:
            num_to_sel = min(num_visualize_chunks, len(processed_chunk_coords_for_vis))
            if num_to_sel > 0:
                selected_vis_chunks = random.sample(processed_chunk_coords_for_vis, num_to_sel)
                logger.info(f"Selected {num_to_sel} chunks for visualization: {[c[2] for c in selected_vis_chunks]}")
                logger.info(f"Reloading image channels (original indices: {vis_bg_channel_indices}) for visualization background...")
                try: # Simplified loading from previous version, adapt if complex channel logic needed
                    with tifffile.TiffFile(image_path) as tiff_vis:
                        series0 = tiff_vis.series[0]
                        if series0.ndim < 3:
                            logger.error(f"Vis BG needs >=3D, got {series0.shape}"); visualize=False
                        else:
                            # Assuming vis_bg_channel_indices are for the first dimension if C,Y,X or second if T,C,Y,X
                            # This part might need more robust channel axis detection like in previous script if TIFFs vary a lot
                            channel_axis_for_vis = 0 if series0.ndim == 3 else 1 # Heuristic

                            loaded_bg_channels = []
                            for ch_idx_orig in vis_bg_channel_indices:
                                if not (0 <= ch_idx_orig < series0.shape[channel_axis_for_vis]):
                                    logger.error(f"Vis BG channel index {ch_idx_orig} out of bounds for axis {channel_axis_for_vis} (size {series0.shape[channel_axis_for_vis]})");
                                    visualize=False; break
                                key_list = [slice(None)] * series0.ndim
                                key_list[channel_axis_for_vis] = ch_idx_orig
                                if series0.ndim == 4 and channel_axis_for_vis != 0: key_list[0] = 0 # Take T=0

                                page = series0.asarray(key=tuple(key_list))
                                squeezed_page = np.squeeze(page)
                                if squeezed_page.ndim == 2:
                                    loaded_bg_channels.append(squeezed_page)
                                else:
                                    logger.error(f"Loaded vis page for ch {ch_idx_orig} not 2D after squeeze: {squeezed_page.shape}");
                                    visualize=False; break
                            if visualize and len(loaded_bg_channels) == len(vis_bg_channel_indices):
                                vis_img_stack_bg = np.stack(loaded_bg_channels, axis=0)
                                logger.info(f"Loaded vis background stack: {vis_img_stack_bg.shape}")
                            elif visualize:
                                logger.warning(f"Failed to load all {len(vis_bg_channel_indices)} vis BG channels."); visualize=False
                except Exception as e:
                    logger.error(f"Failed to reload image data for visualization: {e}", exc_info=True); visualize = False
            else: logger.warning("No chunks to visualize."); visualize = False

            if visualize and vis_img_stack_bg is not None:
                for y_s_chunk, x_s_chunk, chunk_id_str in selected_vis_chunks:
                    logger.info(f"Generating visualization for chunk area: {chunk_id_str}")
                    c_center_y, c_center_x = y_s_chunk + chunk_size[0]//2, x_s_chunk + chunk_size[1]//2
                    vis_y_s = max(0, c_center_y - visualize_roi_size[0]//2)
                    vis_x_s = max(0, c_center_x - visualize_roi_size[1]//2)

                    fig = visualize_roi_combined(
                        vis_img_stack_bg, (vis_y_s, vis_x_s), visualize_roi_size,
                        df_cells_outlines=df_cells, df_nuclei_outlines=df_nuclei,
                        vis_ch_indices=list(range(len(vis_bg_channel_indices))), # Use 0,1 for the 2-channel stack
                        original_bg_channels=vis_bg_channel_indices
                    )
                    if fig:
                        try:
                            plot_fn = os.path.join(visualize_output_dir, f"visualization_chunk_{chunk_id_str}.png")
                            fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
                            logger.info(f"Saved visualization: {plot_fn}"); plt.close(fig)
                        except Exception as e: logger.error(f"Failed to save plot for {chunk_id_str}: {e}"); plt.close(fig)
                    else: logger.warning(f"Skipped saving plot for {chunk_id_str}, figure gen failed.")
            elif visualize: logger.warning("Vis enabled, but BG stack not prepared. Skipping plots.")

    total_duration = (datetime.now() - start_time).total_seconds()
    logger.info("="*50)
    logger.info(f"Processing finished at {datetime.now()}")
    logger.info(f"Total time: {total_duration:.2f}s ({total_duration/60:.2f}m)")
    logger.info(f"Total cells: {global_cell_id_counter-1}, Total nuclei: {global_nuclei_id_counter-1}")
    logger.info(f"Log file: {log_file}")
    if visualize and visualize_output_dir and num_visualize_chunks > 0 and vis_img_stack_bg is not None:
        logger.info(f"Visualizations saved in: {visualize_output_dir}")
    logger.info("="*50)

# --- Argument Parsing Helper ---
def list_of_two_ints(string):
    try:
        values = list(map(int, string.split()))
        if len(values) != 2:
            raise argparse.ArgumentTypeError("Channels must be two integers separated by space (e.g., '0 0' or '1 2')")
        return values
    except ValueError:
        raise argparse.ArgumentTypeError("Channels must be two integers separated by space")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DUAL Cellpose segmentation (cells and nuclei) on a large image.")
    parser.add_argument("image_path", help="Path to input OME-TIFF image.")
    parser.add_argument("csv_base_path", help="Base path for output CSV files (e.g., /path/to/output/basename -> basename_cells.csv, basename_nuclei.csv).")

    # Model paths
    parser.add_argument("--cell_model_path", default="cyto", help="Path to Cellpose model for cells (or 'cyto', 'cyto2'). Default: cyto")
    parser.add_argument("--nuclei_model_path", default="nuclei", help="Path to Cellpose model for nuclei (or 'nuclei'). Default: nuclei")

    # Processing
    parser.add_argument("--chunk_size", type=int, nargs=2, default=[2048, 2048], metavar=('Y', 'X'), help="Chunk size (Y X). Default: 2048 2048")

    # Cell model parameters
    parser.add_argument("--cells_diameter", type=float, default=120.0, help="Cell diameter for cell model. Default: 120.0")
    parser.add_argument("--cells_flow_threshold", type=float, default=0.4, help="Flow threshold for cell model. Default: 0.4")
    parser.add_argument("--cells_cellprob_threshold", type=float, default=-1.5, help="Cell probability threshold for cell model. Default: -1.5")
    parser.add_argument("--cells_channels", type=list_of_two_ints, default="0 0", help="Channels for cell model (e.g., '0 0' for grayscale, '1 2' for cyto/nuc). Default: '0 0'")

    # Nuclei model parameters
    parser.add_argument("--nuclei_diameter", type=float, default=60.0, help="Cell diameter for nuclei model. Default: 60.0")
    parser.add_argument("--nuclei_flow_threshold", type=float, default=0.4, help="Flow threshold for nuclei model. Default: 0.4")
    parser.add_argument("--nuclei_cellprob_threshold", type=float, default=-1.5, help="Cell probability threshold for nuclei model. Default: -1.5")
    parser.add_argument("--nuclei_channels", type=list_of_two_ints, default="0 0", help="Channels for nuclei model (e.g., '0 0' for grayscale from channel 0, '2 0' for grayscale from channel 2). Default: '0 0'")

    # Visualization
    parser.add_argument("--visualize", action='store_true', help="Enable visualization.")
    parser.add_argument("--visualize_output_dir", default=None, help="Directory for visualization plots. Defaults to CSV base path directory.")
    parser.add_argument("--num_visualize_chunks", type=int, default=5, help="Number of random chunks to visualize. Default: 5")
    parser.add_argument("--visualize_roi_size", type=int, nargs=2, default=[512, 512], metavar=('Y', 'X'), help="Size of visualization ROI (Y X). Default: 512 512")
    parser.add_argument("--vis_bg_channel_indices", type=list_of_two_ints, default="0 1", help="Original TIFF channel indices for Green and Blue in visualization BG (e.g., '0 1'). Default: '0 1'")

    args = parser.parse_args()

    process_large_image(
        image_path=args.image_path,
        csv_base_path=args.csv_base_path,
        chunk_size=tuple(args.chunk_size),
        cell_model_path=args.cell_model_path,
        cells_diameter=args.cells_diameter,
        cells_flow_threshold=args.cells_flow_threshold,
        cells_cellprob_threshold=args.cells_cellprob_threshold,
        cells_channels=args.cells_channels, # Already a list from list_of_two_ints
        nuclei_model_path=args.nuclei_model_path,
        nuclei_diameter=args.nuclei_diameter,
        nuclei_flow_threshold=args.nuclei_flow_threshold,
        nuclei_cellprob_threshold=args.nuclei_cellprob_threshold,
        nuclei_channels=args.nuclei_channels, # Already a list
        visualize=args.visualize,
        visualize_output_dir=args.visualize_output_dir,
        num_visualize_chunks=args.num_visualize_chunks,
        visualize_roi_size=tuple(args.visualize_roi_size),
        vis_bg_channel_indices=args.vis_bg_channel_indices # Already a list
    )
    print("\n--- Python Script Execution Finished ---")