import dask.array as da
import tifffile
import numpy as np
from cellpose import models, utils
# import zarr # Not strictly needed if only saving CSV and visualizing
# from distributed import Client # Not needed for this sequential script
import os
import pandas as pd
from skimage import measure
from skimage.exposure import rescale_intensity # Needed for visualization
import logging
from datetime import datetime
import matplotlib.pyplot as plt # Needed for visualization
# from matplotlib.colors import ListedColormap # Not used in the adapted visualization
import random # Needed for random chunk selection
import argparse # Needed for command-line arguments

# --- Helper Functions ---

def get_cell_outlines(masks):
    """
    Extract cell outlines and their coordinates from a mask
    Returns a list of dictionaries containing cell ID and outline coordinates
    """
    cells = []
    # Find unique cell IDs (excluding background = 0)
    cell_ids = np.unique(masks)[1:]

    for cell_id in cell_ids:
        # Get binary mask for this cell
        cell_mask = masks == cell_id
        # Find contours using skimage
        contours = measure.find_contours(cell_mask, 0.5)

        if len(contours) > 0:
            # Take the longest contour if there are multiple
            # contour = max(contours, key=len) # Use longest
            contour = contours[0] # Or just the first one found

            # Add cell information to list
            cells.append({
                'cell_id': cell_id, # Local ID within the chunk mask
                'x_coords': contour[:, 1],  # x coordinates
                'y_coords': contour[:, 0],  # y coordinates
            })
    return cells

def segment_chunk(chunk, model=None, chunk_position=None, diameter=80, flow_threshold=0.8, cellprob_threshold=-3.5, channels=[0,0]):
    """
    Segment a single chunk using cellpose and extract outlines.
    Uses parameters passed from the main function.
    """
    if model is None:
        # Fallback if model not passed, though it should be
        print("Error: Model not provided to segment_chunk.")
        # Handle this case appropriately, maybe raise an error or return empty
        # For now, returning empty results.
        return np.array([], dtype=np.uint16), []


    # --- Preprocessing specific to Cellpose ---
    # Ensure chunk is 2D (or handle specific channels if needed)
    if chunk.ndim > 2 and len(channels) != 2:
         # If model expects grayscale ([0,0]) but chunk is multi-channel,
         # we might need to select a channel or convert.
         # This depends heavily on the model's training data.
         # Assuming the model handles channel selection internally if channels != [0,0]
         # If channels == [0,0], Cellpose might expect a 2D array.
         # Let's add a check/warning here.
         if channels == [0,0]:
              print(f"Warning: Chunk at {chunk_position} has {chunk.ndim} dimensions but model channels are [0,0]. Using first channel (index 0).")
              # Example: take the first channel if grayscale is expected
              if chunk.ndim == 3: # C, Y, X
                  chunk = chunk[0]
              elif chunk.ndim == 4: # T, C, Y, X
                  chunk = chunk[0, 0]
              else:
                  print(f"Error: Cannot handle chunk dimension {chunk.ndim} for grayscale model.")
                  return np.zeros(chunk.shape[-2:], dtype=np.uint16), []
         # If channels are like [1,2], Cellpose expects multi-channel input, so pass as is.

    elif chunk.ndim < 2:
         print(f"Chunk has insufficient dimensions at position {chunk_position}: shape {chunk.shape}")
         return np.array([], dtype=np.uint16), []


    # Ensure chunk is not empty and has proper dimensions
    if chunk.size == 0:
        print(f"Empty chunk at position {chunk_position}")
        return np.array([], dtype=np.uint16), []

    # Check minimum size for processing (Cellpose might have internal checks)
    if any(s < 10 for s in chunk.shape[-2:]): # Check spatial dims
        print(f"Chunk spatial dimensions too small at position {chunk_position}: shape {chunk.shape}")
        # Return empty mask of appropriate shape if possible
        return np.zeros(chunk.shape[-2:], dtype=np.uint16), []

    # --- Normalization ---
    # Cellpose often performs internal normalization. Manual normalization is usually not needed.
    # chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-7) * 255
    # chunk = chunk.astype(np.uint8)

    # --- Run Cellpose ---
    try:
        # Run cellpose
        masks, flows, styles = model.eval(chunk,
                                          diameter=diameter,
                                          flow_threshold=flow_threshold,
                                          cellprob_threshold=cellprob_threshold,
                                          channels=channels,
                                          # do_3D=False # Ensure 2D processing if needed
                                          )

        # Get cell outlines
        cells = get_cell_outlines(masks)

        # Adjust coordinates based on chunk position
        if chunk_position is not None:
            y_offset, x_offset = chunk_position # Expecting (y, x)

            for cell in cells:
                # Adjust outline coordinates relative to the whole slide
                cell['x_coords'] += x_offset
                cell['y_coords'] += y_offset
                # Add chunk identifier
                cell['chunk_id'] = f"{y_offset}_{x_offset}"

        # Return the masks (local to chunk) and the adjusted outline info
        return masks.astype(np.uint16), cells

    except Exception as e:
        print(f"Error processing chunk at {chunk_position} with shape {chunk.shape}: {e}")
        # Return empty mask of appropriate shape if possible
        return np.zeros(chunk.shape[-2:], dtype=np.uint16), []


# --- Visualization Functions (Adapted from script2) ---

def create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices=[0, 1]):
    """
    Creates a normalized RGB image from specified channels for an ROI.
    Maps channel 1 (index 0 in vis_ch_indices) to Green,
    channel 2 (index 1 in vis_ch_indices) to Blue.
    """
    # Ensure we have at least two channels in the stack
    if image_data_stack.ndim < 3 or image_data_stack.shape[0] < max(vis_ch_indices) + 1:
        raise ValueError(f"Image stack has shape {image_data_stack.shape}, but need at least {max(vis_ch_indices) + 1} channels at indices {vis_ch_indices}")

    # Extract channels based on provided indices
    roi_ch_g = image_data_stack[vis_ch_indices[0], y_start:y_end, x_start:x_end]
    roi_ch_b = image_data_stack[vis_ch_indices[1], y_start:y_end, x_start:x_end]

    # Normalize each channel to 0-1 range for display
    p_low, p_high = 1, 99
    g_min, g_max = np.percentile(roi_ch_g, (p_low, p_high))
    b_min, b_max = np.percentile(roi_ch_b, (p_low, p_high))

    # Avoid division by zero if max == min
    roi_ch_g_norm = rescale_intensity(roi_ch_g, in_range=(g_min, g_max if g_max > g_min else g_max + 1e-6), out_range=(0.0, 1.0))
    roi_ch_b_norm = rescale_intensity(roi_ch_b, in_range=(b_min, b_max if b_max > b_min else b_max + 1e-6), out_range=(0.0, 1.0))

    # Create RGB image: Map Channel 1 (index 0) to Green, Channel 2 (index 1) to Blue
    rgb_image = np.zeros((roi_ch_g_norm.shape[0], roi_ch_g_norm.shape[1], 3), dtype=float)
    rgb_image[..., 1] = roi_ch_g_norm   # Green
    rgb_image[..., 2] = roi_ch_b_norm   # Blue

    # Clip values just in case normalization produced slightly out-of-range values
    rgb_image = np.clip(rgb_image, 0, 1)

    return rgb_image


def visualize_roi_combined(image_data_stack, roi_position, roi_size, df_outlines=None, vis_ch_indices=[0, 1], figsize=(10, 10)):
    """
    Visualize a specific ROI with overlaid outlines on a composite color image.
    Assumes df_outlines contains columns 'global_cell_id', 'x', 'y'.

    Parameters:
    -----------
    image_data_stack : numpy.ndarray
        A (C, Y, X) stack containing at least the channels specified in vis_ch_indices.
    roi_position : tuple
        (y, x) coordinates of the top-left corner of the ROI.
    roi_size : tuple
        (height, width) size of the ROI.
    df_outlines : pandas.DataFrame, optional
        DataFrame with segmentation outlines ('global_cell_id', 'x', 'y').
    vis_ch_indices : list, optional
        Indices of the channels in image_data_stack to use for Green and Blue background.
    figsize : tuple
        Figure size.
    """
    # Extract ROI coordinates and ensure they are within image bounds
    y_start, x_start = roi_position
    height, width = roi_size
    # Get image shape from stack (assuming C, Y, X)
    if image_data_stack.ndim != 3:
         print(f"Error: visualize_roi_combined expects a 3D (C, Y, X) image stack, got shape {image_data_stack.shape}")
         return None
    img_c, img_h, img_w = image_data_stack.shape

    # Calculate end coordinates, clipping to image boundaries
    y_end = min(y_start + height, img_h)
    x_end = min(x_start + width, img_w)

    # Adjust start coordinates if clipping occurred, to maintain size if possible
    y_start = max(0, y_end - height)
    x_start = max(0, x_end - width)

    # Recalculate actual height/width after clipping/adjustment
    actual_height = y_end - y_start
    actual_width = x_end - x_start

    if actual_height <= 0 or actual_width <= 0:
        print(f"Warning: ROI at {roi_position} with size {roi_size} resulted in zero area after clipping.")
        return None # Cannot visualize zero-area ROI

    # Create RGB image for the ROI using the specified channels
    try:
        rgb_roi = create_rgb_roi(image_data_stack, y_start, y_end, x_start, x_end, vis_ch_indices)
    except ValueError as e:
        print(f"Error creating RGB ROI: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating RGB ROI: {e}")
        return None


    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the composite color ROI
    ax.imshow(rgb_roi)
    ax.set_title(f'ROI at ({y_start}, {x_start}) Size ({actual_width}x{actual_height}) | BG: G=Ch{vis_ch_indices[0]}, B=Ch{vis_ch_indices[1]} | Outlines: Red')
    ax.axis('off')

    # Plot Outlines (Red)
    if df_outlines is not None and not df_outlines.empty:
        # Filter outlines to only those within the *actual* ROI bounds
        roi_df = df_outlines[(df_outlines['x'] >= x_start) & (df_outlines['x'] < x_end) &
                             (df_outlines['y'] >= y_start) & (df_outlines['y'] < y_end)].copy()

        if not roi_df.empty:
            # Adjust coordinates to be relative to the ROI's top-left corner (0,0)
            roi_df['x_rel'] = roi_df['x'] - x_start
            roi_df['y_rel'] = roi_df['y'] - y_start

            # Group by the global cell ID and plot each outline
            for obj_id, group in roi_df.groupby('global_cell_id'):
                # Plot outlines in red for contrast
                ax.plot(group['x_rel'], group['y_rel'], 'r-', linewidth=1.0, alpha=0.7)

    plt.tight_layout()
    return fig


# --- Main Processing Function ---

def process_large_image(
    image_path,
    csv_path,
    # Chunking
    chunk_size=(1024, 1024),
    # Cellpose model parameters
    model_path="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700",
    diameter=80.0,
    flow_threshold=0.8,
    cellprob_threshold=-3.5,
    channels=[1,2], # Defaulting to grayscale [0,0] - adjust if model needs color
    # Visualization parameters
    visualize=True,
    visualize_output_dir=None,
    num_visualize_chunks=5,
    visualize_roi_size=(1024, 1024),
    vis_bg_channel_indices=[0, 1]
    ):
    """
    Process a large OME-TIFF image chunk by chunk, save outlines to CSV,
    and optionally visualize segmentation on selected chunks.
    """
    # Set up logging
    log_file = csv_path.replace('.csv', '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Log start time and parameters
    start_time = datetime.now()
    logger.info("="*50)
    logger.info(f"Starting image processing at {start_time}")
    logger.info(f"Input image: {image_path}")
    logger.info(f"Output CSV: {csv_path}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Cellpose Model Path: {model_path}")
    logger.info(f"Cellpose Diameter: {diameter}")
    logger.info(f"Cellpose Flow Threshold: {flow_threshold}")
    logger.info(f"Cellpose Cellprob Threshold: {cellprob_threshold}")
    logger.info(f"Cellpose Channels: {channels}")
    logger.info(f"Visualization Enabled: {visualize}")
    if visualize:
        # Ensure vis output dir is set, default to csv dir if not
        if visualize_output_dir is None:
            visualize_output_dir = os.path.dirname(csv_path)
            logger.info(f"Visualization Output Dir not set, defaulting to: {visualize_output_dir}")
        else:
             logger.info(f"Visualization Output Dir: {visualize_output_dir}")
        logger.info(f"Number of Chunks to Visualize: {num_visualize_chunks}")
        logger.info(f"Visualization ROI Size: {visualize_roi_size}")
        logger.info(f"Visualization Background Channels (G, B): {vis_bg_channel_indices}")
    logger.info("="*50)


    # --- Initialize Model ---
    logger.info("Initializing Cellpose model...")
    try:
        # Use CellposeModel for direct model loading/evaluation
        model = models.CellposeModel(pretrained_model=model_path, gpu=True)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Cellpose model from {model_path}: {e}")
        return # Cannot proceed without a model

    # --- Read OME-TIFF ---
    logger.info("Reading OME-TIFF file structure...")
    try:
        with tifffile.TiffFile(image_path) as tiff:
            # Use dask for efficient chunked reading via Zarr store if possible
            try:
                zarr_store = tiff.series[0].aszarr()
                data = da.from_zarr(zarr_store)
                logger.info("Using Zarr store for reading.")
            except Exception as zarr_err:
                logger.warning(f"Could not open as Zarr store ({zarr_err}), falling back to asarray.")
                # Fallback: Read the whole array into dask (might be memory intensive)
                # Chunking here is mainly for dask internal representation before rechunking
                data = da.from_array(tiff.series[0].asarray(), chunks=chunk_size)

    except FileNotFoundError:
        logger.error(f"Input image file not found: {image_path}")
        return
    except Exception as e:
        logger.error(f"Failed to read TIFF file {image_path}: {e}")
        return

    logger.info(f"Original data shape: {data.shape}")

    # --- Data Shape Handling & Rechunking ---
    # Determine the shape relevant for 2D processing (Y, X)
    if data.ndim >= 2:
        img_shape_yx = data.shape[-2:]
    else:
        logger.error(f"Data has insufficient dimensions for 2D processing: {data.shape}")
        return

    logger.info(f"Processing based on YX shape: {img_shape_yx}")

    # Calculate chunks based on the determined YX shape
    n_chunks_y = (img_shape_yx[0] + chunk_size[0] - 1) // chunk_size[0]
    n_chunks_x = (img_shape_yx[1] + chunk_size[1] - 1) // chunk_size[1]
    total_chunks = n_chunks_y * n_chunks_x
    logger.info(f"Total number of chunks to process: {total_chunks} ({n_chunks_y} x {n_chunks_x})")

    # Ensure dask array is chunked according to processing chunk_size for efficiency
    if data.ndim == 2: # Y, X
        final_chunking = chunk_size
    elif data.ndim == 3: # Assume C, Y, X
        final_chunking = (data.shape[0], chunk_size[0], chunk_size[1])
    elif data.ndim == 4: # Assume T, C, Y, X
        final_chunking = (data.shape[0], data.shape[1], chunk_size[0], chunk_size[1])
    else:
        logger.warning(f"Data dimension {data.ndim} not explicitly handled for rechunking, using default.")
        final_chunking = 'auto'

    logger.info(f"Rechunking data with chunks: {final_chunking}")
    try:
        data = data.rechunk(final_chunking)
    except Exception as e:
        logger.error(f"Failed to rechunk dask array: {e}")
        # Decide whether to proceed or stop
        # return


    # --- Process Chunks ---
    all_cells_info = [] # Store dicts from get_cell_outlines (adjusted coords)
    processed_chunk_coords = [] # Store (y_start, x_start, chunk_id) for visualization selection
    global_cell_id_counter = 1 # Start global IDs from 1
    processed_chunks_count = 0

    logger.info("Starting chunk processing loop...")
    for y_idx in range(n_chunks_y):
        for x_idx in range(n_chunks_x):
            chunk_start_time = datetime.now()
            processed_chunks_count += 1

            # Calculate chunk boundaries in YX plane
            y_start = y_idx * chunk_size[0]
            x_start = x_idx * chunk_size[1]
            y_end = min(y_start + chunk_size[0], img_shape_yx[0])
            x_end = min(x_start + chunk_size[1], img_shape_yx[1])

            # Define the slice for the current chunk
            if data.ndim == 2: # Y, X
                chunk_slice = (slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 3: # Assume C, Y, X
                chunk_slice = (slice(None), slice(y_start, y_end), slice(x_start, x_end))
            elif data.ndim == 4: # Assume T, C, Y, X
                chunk_slice = (slice(None), slice(None), slice(y_start, y_end), slice(x_start, x_end))
            else:
                logger.error(f"Cannot create slice for data dimension {data.ndim}")
                continue # Skip this chunk

            logger.info(f"Processing chunk [{processed_chunks_count}/{total_chunks}] at YX: ({y_start}, {x_start}) -> Slice: {chunk_slice}")

            try:
                # Compute the chunk data (bring it into memory)
                chunk_data = data[chunk_slice].compute()

                if chunk_data.size == 0:
                    logger.warning(f"Computed chunk data is empty at YX ({y_start}, {x_start}). Skipping.")
                    continue

                # Process the computed chunk
                masks_chunk, cells_in_chunk_info = segment_chunk(
                    chunk_data,
                    model=model,
                    chunk_position=(y_start, x_start), # Pass YX position
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    channels=channels
                )

                # Assign global IDs and store outline info
                cells_found_count = 0
                if cells_in_chunk_info:
                    cells_found_count = len(cells_in_chunk_info)
                    for cell_info in cells_in_chunk_info:
                        cell_info['global_cell_id'] = global_cell_id_counter
                        all_cells_info.append(cell_info)
                        global_cell_id_counter += 1

                    # Store chunk info for potential visualization
                    chunk_id = f"{y_start}_{x_start}"
                    processed_chunk_coords.append((y_start, x_start, chunk_id))

                chunk_end_time = datetime.now()
                chunk_duration = (chunk_end_time - chunk_start_time).total_seconds()

                logger.info(f"Chunk [{processed_chunks_count}/{total_chunks}] completed in {chunk_duration:.2f}s. Found {cells_found_count} cells.")

            except Exception as e:
                logger.error(f"Error processing chunk at YX ({y_start}, {x_start}): {e}", exc_info=True)
                continue

    logger.info("Chunk processing loop finished.")

    # --- Create and Save DataFrame ---
    if all_cells_info:
        logger.info("Creating final DataFrame from collected cell outlines...")
        df_rows = []
        for cell_info in all_cells_info:
            cell_id = cell_info['global_cell_id']
            chunk_id = cell_info['chunk_id']
            for x, y in zip(cell_info['x_coords'], cell_info['y_coords']):
                df_rows.append({
                    'global_cell_id': cell_id,
                    'chunk_id': chunk_id,
                    'x': x,
                    'y': y
                })

        df = pd.DataFrame(df_rows)
        logger.info(f"Created DataFrame with {len(df)} points from {global_cell_id_counter - 1} cells.")

        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved successfully to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to CSV {csv_path}: {e}")

    else:
        logger.warning("No cells were found in the image. No CSV file created.")
        df = pd.DataFrame() # Create empty dataframe for visualization step consistency

    # --- Optional Visualization ---
    if visualize and visualize_output_dir and processed_chunk_coords:
        logger.info("--- Starting Visualization Step ---")

        if not os.path.exists(visualize_output_dir):
            try:
                os.makedirs(visualize_output_dir)
                logger.info(f"Created visualization output directory: {visualize_output_dir}")
            except Exception as e:
                logger.error(f"Could not create visualization directory {visualize_output_dir}: {e}")
                visualize = False # Disable visualization if dir creation fails

        if visualize and df.empty:
             logger.warning("Outline DataFrame is empty, cannot generate visualizations.")
             visualize = False

        if visualize:
            num_to_select = min(num_visualize_chunks, len(processed_chunk_coords))
            if num_to_select > 0:
                selected_chunks = random.sample(processed_chunk_coords, num_to_select)
                logger.info(f"Selected {num_to_select} chunks for visualization: {[c[2] for c in selected_chunks]}")

                logger.info(f"Reloading image channels {vis_bg_channel_indices} for visualization background...")
                try:
                    # Use tifffile directly to load only the required channels
                    with tifffile.TiffFile(image_path) as tiff_vis:
                        data_shape_vis = tiff_vis.series[0].shape
                        if tiff_vis.series[0].ndim < 3:
                             logger.error("Cannot create composite visualization background from 2D image data.")
                             visualize = False
                        else:
                             # Determine channel axis (heuristic)
                             channel_axis = -3 # Default assumption for C, Y, X
                             if len(data_shape_vis) == 3 and data_shape_vis[0] > 1 and data_shape_vis[0] < min(data_shape_vis[1], data_shape_vis[2]):
                                 channel_axis = 0 # Likely (C, Y, X)
                             elif len(data_shape_vis) == 4 and data_shape_vis[1] > 1: # Assume T, C, Y, X
                                 channel_axis = 1

                             # Build slice object
                             vis_slice = [slice(None)] * len(data_shape_vis)
                             vis_slice[channel_axis] = vis_bg_channel_indices
                             # If 4D (T,C,Y,X), take first time point T=0
                             if len(data_shape_vis) == 4:
                                 vis_slice[0] = 0

                             vis_img_stack = tiff_vis.series[0].asarray(key=tuple(vis_slice))

                             # Ensure stack is (C, Y, X) where C=len(vis_bg_channel_indices)
                             if vis_img_stack.ndim != 3 or vis_img_stack.shape[0] != len(vis_bg_channel_indices):
                                 raise ValueError(f"Loaded stack has shape {vis_img_stack.shape}, expected ({len(vis_bg_channel_indices)}, Y, X). Slice: {tuple(vis_slice)}")

                             logger.info(f"Loaded visualization background stack with shape: {vis_img_stack.shape}")

                except Exception as e:
                    logger.error(f"Failed to reload image data for visualization: {e}", exc_info=True)
                    visualize = False # Disable visualization

            else:
                logger.warning("No chunks were processed successfully, skipping visualization.")
                visualize = False

            # Generate plots for selected chunks
            if visualize:
                for y_start_chunk, x_start_chunk, chunk_id in selected_chunks:
                    logger.info(f"Generating visualization for chunk area: {chunk_id}")

                    # Center the visualization ROI on the chunk's center
                    chunk_center_y = y_start_chunk + chunk_size[0] // 2
                    chunk_center_x = x_start_chunk + chunk_size[1] // 2
                    vis_y_start = max(0, chunk_center_y - visualize_roi_size[0] // 2)
                    vis_x_start = max(0, chunk_center_x - visualize_roi_size[1] // 2)
                    vis_roi_pos = (vis_y_start, vis_x_start)

                    # Generate the plot
                    fig = visualize_roi_combined(
                        vis_img_stack,
                        vis_roi_pos, # Position for visualization ROI
                        visualize_roi_size, # Size of visualization ROI
                        df_outlines=df, # Pass the full dataframe
                        vis_ch_indices=vis_bg_channel_indices
                    )

                    # Save the plot
                    if fig:
                        try:
                            plot_filename = os.path.join(visualize_output_dir, f"visualization_chunk_{chunk_id}.png")
                            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                            logger.info(f"Saved visualization: {plot_filename}")
                            plt.close(fig) # Close figure to free memory
                        except Exception as e:
                            logger.error(f"Failed to save plot for chunk {chunk_id}: {e}")
                            plt.close(fig) # Still close it
                    else:
                         logger.warning(f"Skipped saving plot for chunk {chunk_id} as figure generation failed.")

    # --- Final Summary ---
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    logger.info("="*50)
    logger.info(f"Processing finished at {end_time}")
    logger.info(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info(f"Total cells found: {global_cell_id_counter - 1}")
    logger.info(f"Results CSV saved to: {csv_path}")
    logger.info(f"Log file saved to: {log_file}")
    if visualize and visualize_output_dir:
        logger.info(f"Visualizations saved in: {visualize_output_dir}")
    logger.info("="*50)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cellpose segmentation on a large image, chunk by chunk, with optional visualization.")

    # Required arguments
    parser.add_argument("image_path", help="Path to the input OME-TIFF image.")
    parser.add_argument("csv_path", help="Path to save the output CSV file with cell outlines.")
    parser.add_argument("model_path", help="Path to the pre-trained Cellpose model file.")

    # Optional arguments - Processing
    parser.add_argument("--chunk_size", type=int, nargs=2, default=[2048, 2048], metavar=('Y', 'X'), help="Chunk size for processing (Y X). Default: 2048 2048")
    parser.add_argument("--diameter", type=float, default=120.0, help="Cell diameter for Cellpose. Default: 80.0")
    parser.add_argument("--flow_threshold", type=float, default=0.6, help="Flow threshold for Cellpose. Default: 0.8")
    parser.add_argument("--cellprob_threshold", type=float, default=-1.5, help="Cell probability threshold for Cellpose. Default: -3.5")
    parser.add_argument("--channels", type=int, nargs=2, default=[1, 2], metavar=('C1', 'C2'), help="Channels for Cellpose model (e.g., [0,0] grayscale, [1,2] cyto/nuc). Default: [0, 0]")

    # Optional arguments - Visualization
    parser.add_argument("--visualize", action='store_true', help="Enable visualization of random chunks.")
    parser.add_argument("--visualize_output_dir", default=None, help="Directory to save visualization plots. Defaults to CSV directory if --visualize is set.")
    parser.add_argument("--num_visualize_chunks", type=int, default=5, help="Number of random chunks to visualize. Default: 5")
    parser.add_argument("--visualize_roi_size", type=int, nargs=2, default=[512, 512], metavar=('Y', 'X'), help="Size of the visualization ROI (Y X). Default: 512 512")
    parser.add_argument("--vis_bg_channel_indices", type=int, nargs=2, default=[0, 1], metavar=('G_CH', 'B_CH'), help="Channel indices from original TIFF for Green and Blue background visualization. Default: [0, 1]")

    args = parser.parse_args()

    # Convert lists to tuples where needed
    chunk_size_tuple = tuple(args.chunk_size)
    visualize_roi_size_tuple = tuple(args.visualize_roi_size)
    # Keep channels as lists as expected by Cellpose and visualization functions
    channels_list = list(args.channels)
    vis_bg_channel_indices_list = list(args.vis_bg_channel_indices)

    # Call the main processing function
    process_large_image(
        image_path=args.image_path,
        csv_path=args.csv_path,
        chunk_size=chunk_size_tuple,
        model_path=args.model_path,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        channels=channels_list,
        visualize=args.visualize,
        visualize_output_dir=args.visualize_output_dir,
        num_visualize_chunks=args.num_visualize_chunks,
        visualize_roi_size=visualize_roi_size_tuple,
        vis_bg_channel_indices=vis_bg_channel_indices_list
    )

    print("\n--- Script Execution Finished ---")
    # Final messages printed by the logger within process_large_image