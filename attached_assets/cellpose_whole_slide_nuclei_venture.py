import dask.array as da
import tifffile
import numpy as np
from cellpose import models, utils
import zarr
from distributed import Client
import os
import pandas as pd
from skimage import measure
import logging
from datetime import datetime


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
            contour = contours[0]
            
            # Add cell information to list
            cells.append({
                'cell_id': cell_id,
                'x_coords': contour[:, 1],  # x coordinates
                'y_coords': contour[:, 0],  # y coordinates
            })
    
    return cells

def segment_chunk(chunk, model=None, chunk_position=None):
    """
    Segment a single chunk using cellpose and extract outlines
    """
    if model is None:
        model = models.CellposeModel(model_type='nuclei', gpu=True)
        #model = models.CellposeModel(model_type='CP', gpu=True) #use for DAPI-only images
        #model = models.CellposeModel(pretrained_model="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700", gpu=True)

    # Ensure chunk is 2D and properly formatted
    if chunk.ndim > 2:
        # If 3D/4D, take the first channel/z-slice
        chunk = chunk[0] if chunk.ndim == 3 else chunk[0,0]

    # Ensure chunk is not empty and has proper dimensions
    if chunk.size == 0:
        print(f"Empty chunk at position {chunk_position}")
        return np.array([]), []

    # Check minimum size for processing
    if any(s < 3 for s in chunk.shape):
        print(f"Chunk too small at position {chunk_position}: shape {chunk.shape}")
        return np.zeros(chunk.shape, dtype=np.uint16), []

    # Normalize and convert to uint8
    chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-7) * 255
    chunk = chunk.astype(np.uint8)

    try:
        # Run cellpose with error handling for different versions
        result = model.eval(chunk,
                            diameter=100,
                            flow_threshold=0.8,
                            cellprob_threshold=-0.5,
                            channels=[0,0])


        # Handle different return formats
        if isinstance(result, tuple):
            if len(result) == 3:
                masks, flows, styles = result
            elif len(result) == 2:
                masks, flows = result
            else:
                masks = result[0]
        else:
            masks = result

        # Get cell outlines
        cells = get_cell_outlines(masks)

        # Adjust coordinates based on chunk position
        if chunk_position is not None:
            # Ensure chunk_position has correct dimensions
            if len(chunk_position) > 2:
                chunk_position = chunk_position[-2:]  # Take last two dimensions
            y_offset, x_offset = chunk_position

            for cell in cells:
                cell['x_coords'] += x_offset
                cell['y_coords'] += y_offset
                cell['chunk_id'] = f"{y_offset}_{x_offset}"

        return masks, cells

    except Exception as e:
        print(f"Error processing chunk: {e}")
        print(f"Chunk shape: {chunk.shape}")
        print(f"Chunk position: {chunk_position}")
        return np.zeros_like(chunk, dtype=np.uint16), []


def process_large_image(image_path, output_path, csv_path, chunk_size=(1024, 1024)):
    """
    Process a large OME-TIFF image and save outlines to CSV
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting image processing at {start_time}")
    logger.info(f"Input image: {image_path}")
    logger.info(f"Chunk size: {chunk_size}")

    # Initialize model with specific parameters
    logger.info("Initializing Cellpose model...")
    model = models.Cellpose(
        model_type='nuclei',
        gpu=True,
    )
    logger.info("Model initialized successfully")

    # Read the OME-TIFF
    logger.info("Reading OME-TIFF file...")
    with tifffile.TiffFile(image_path) as tiff:
        data = da.from_array(tiff.asarray())

    logger.info(f"Original data shape: {data.shape}")

    # Handle multi-dimensional data
    if data.ndim > 2:
        data = data[0] if data.ndim == 3 else data[0,0]
    logger.info(f"Processing data shape: {data.shape}")

    # Calculate chunks
    n_chunks_y = (data.shape[0] + chunk_size[0] - 1) // chunk_size[0]
    n_chunks_x = (data.shape[1] + chunk_size[1] - 1) // chunk_size[1]
    total_chunks = n_chunks_y * n_chunks_x
    logger.info(f"Total number of chunks to process: {total_chunks} ({n_chunks_y} x {n_chunks_x})")

    # Rechunk if needed
    data = data.rechunk(chunk_size)

    # Lists to store all cell information
    all_cells = []
    global_cell_id = 0
    processed_chunks = 0

    # Process chunks
    for y in range(n_chunks_y):
        for x in range(n_chunks_x):
            chunk_start_time = datetime.now()
            try:
                # Calculate chunk boundaries
                y_start = y * chunk_size[0]
                x_start = x * chunk_size[1]
                y_end = min(y_start + chunk_size[0], data.shape[0])
                x_end = min(x_start + chunk_size[1], data.shape[1])

                # Skip if we're outside the image bounds
                if y_start >= data.shape[0] or x_start >= data.shape[1]:
                    logger.info(f"Skipping out-of-bounds chunk ({y}, {x})")
                    continue

                logger.info(f"Processing chunk [{processed_chunks + 1}/{total_chunks}] at position ({y_start}, {x_start})")

                # Get chunk
                chunk_data = data[y_start:y_end, x_start:x_end].compute()

                # Skip empty chunks
                if chunk_data.size == 0:
                    logger.warning(f"Empty chunk at position ({y_start}, {x_start})")
                    continue

                # Process chunk
                masks, cells = segment_chunk(chunk_data, model, (y_start, x_start))

                # Add global cell IDs and append to master list
                cells_in_chunk = len(cells)
                for cell in cells:
                    cell['global_cell_id'] = global_cell_id
                    global_cell_id += 1
                    all_cells.append(cell)

                chunk_end_time = datetime.now()
                chunk_duration = (chunk_end_time - chunk_start_time).total_seconds()

                logger.info(f"Chunk [{processed_chunks + 1}/{total_chunks}] completed in {chunk_duration:.2f} seconds")
                logger.info(f"Found {cells_in_chunk} cells in this chunk")

                processed_chunks += 1

            except Exception as e:
                logger.error(f"Error processing chunk at position ({y_start}, {x_start}): {e}")
                continue

    # Create DataFrame from all cells
    if all_cells:
        logger.info("Creating final DataFrame...")
        df_rows = []
        for cell in all_cells:
            for x, y in zip(cell['x_coords'], cell['y_coords']):
                df_rows.append({
                    'global_cell_id': cell['global_cell_id'],
                    'chunk_id': cell['chunk_id'],
                    'x': x,
                    'y': y
                })

        # Create and save DataFrame
        df = pd.DataFrame(df_rows)
        df.to_csv(csv_path, index=False)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        logger.info(f"Processing completed in {total_duration:.2f} seconds")
        logger.info(f"Total cells found: {global_cell_id}")
        logger.info(f"Results saved to: {csv_path}")
    else:
        logger.warning("No cells were found in the image")
if __name__ == "__main__":
    # Set up dask client
    client = Client()

    # Process image
    image_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_processed.tif"
    output_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/Venture5_cell_outlines.zarr"
    csv_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/Venture5_nuclei_outlines.csv"

    chunk_size = (3024, 3024)
    # Process the image
    process_large_image(image_path, output_path, csv_path, chunk_size)

    # Print created files
    print("Created/Modified files during execution:")
    print(output_path)
    print(csv_path)
