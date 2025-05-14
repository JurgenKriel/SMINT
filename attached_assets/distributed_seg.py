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
import numcodecs
from multiprocessing import freeze_support

import os
import logging
from multiprocessing import freeze_support
from distributed import Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_tiff_to_zarr(tiff_path, zarr_path, chunks):
    """
    Read a large TIFF file and store it as a zarr array
    Handles multi-dimensional TIFFs correctly without using slice objects
    """
    logger.info(f"Reading TIFF file: {tiff_path}")

    # Get TIFF dimensions without loading the entire file
    with tifffile.TiffFile(tiff_path) as tiff:
        # Get image shape and dtype
        shape = tiff.series[0].shape
        dtype = tiff.series[0].dtype
        logger.info(f"TIFF shape: {shape}, dtype: {dtype}")

        # Create zarr array
        z_array = zarr.open(
            zarr_path,
            mode='w',
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=None  # No compression for speed
        )

        # For 3D+ TIFFs, we need to handle the dimensions correctly
        if len(shape) == 3:  # 3D array (channels, height, width)
            logger.info("Processing 3D TIFF (channels, height, width)")

            # Process each channel separately
            for c in range(shape[0]):
                logger.info(f"Processing channel {c+1}/{shape[0]}")

                # Process in chunks to avoid memory issues
                for i in range(0, shape[1], chunks[0]):
                    i_end = min(i + chunks[0], shape[1])
                    for j in range(0, shape[2], chunks[1]):
                        j_end = min(j + chunks[1], shape[2])

                        # Read this chunk - avoid using slice objects directly
                        # Instead, read the whole page and then extract the region
                        chunk_data = np.zeros((i_end-i, j_end-j), dtype=dtype)
                        
                        # Read the data page by page
                        page = tiff.pages[c]  # Get the page for this channel
                        page_data = page.asarray()
                        chunk_data = page_data[i:i_end, j:j_end]

                        # Write to zarr
                        z_array[c, i:i_end, j:j_end] = chunk_data

        elif len(shape) == 2:  # 2D array (height, width)
            logger.info("Processing 2D TIFF (height, width)")

            # Process in chunks to avoid memory issues
            for i in range(0, shape[0], chunks[0]):
                i_end = min(i + chunks[0], shape[0])
                for j in range(0, shape[1], chunks[1]):
                    j_end = min(j + chunks[1], shape[1])

                    # Read this chunk - avoid using slice objects directly
                    page = tiff.pages[0]  # Get the first page
                    page_data = page.asarray()
                    chunk_data = page_data[i:i_end, j:j_end]

                    # Write to zarr
                    z_array[i:i_end, j:j_end] = chunk_data

        else:
            logger.error(f"Unsupported TIFF dimensions: {len(shape)}")
            raise ValueError(f"Unsupported TIFF dimensions: {len(shape)}")

    logger.info(f"TIFF file successfully converted to zarr: {zarr_path}")
    return z_array

def extract_channel_from_zarr(input_zarr, channel_zarr_path, channel, chunks):
    """
    Extract a single channel from a multi-channel zarr array
    """
    logger.info(f"Extracting channel {channel} from multi-channel zarr")
    
    # Get the shape of the selected channel
    channel_shape = input_zarr.shape[1:]  # Remove the channel dimension
    
    # Create zarr array for the channel
    channel_zarr = zarr.open(
        channel_zarr_path,
        mode='w',
        shape=channel_shape,
        chunks=chunks,
        dtype=input_zarr.dtype,
        compressor=None
    )
    
    # Copy the selected channel data
    for i in range(0, channel_shape[0], chunks[0]):
        i_end = min(i + chunks[0], channel_shape[0])
        for j in range(0, channel_shape[1], chunks[1]):
            j_end = min(j + chunks[1], channel_shape[1])
            
            # Read from the input zarr and write to the channel zarr
            channel_zarr[i:i_end, j:j_end] = input_zarr[channel, i:i_end, j:j_end]
    
    return channel_zarr

def get_cell_outlines_from_zarr(zarr_array, output_csv):
    """
    Extract cell outlines from a zarr array containing segmentation masks
    """
    logger.info("Extracting cell outlines from segmentation masks...")

    cells = []
    global_cell_id = 0

    # Process the zarr array in chunks to avoid memory issues
    chunk_size = (1024, 1024)
    for i in range(0, zarr_array.shape[0], chunk_size[0]):
        i_end = min(i + chunk_size[0], zarr_array.shape[0])
        for j in range(0, zarr_array.shape[1], chunk_size[1]):
            j_end = min(j + chunk_size[1], zarr_array.shape[1])

            # Get this chunk
            chunk_masks = zarr_array[i:i_end, j:j_end]

            # Find unique cell IDs in this chunk (excluding background = 0)
            chunk_cell_ids = np.unique(chunk_masks)[1:]

            for cell_id in chunk_cell_ids:
                # Get binary mask for this cell
                cell_mask = chunk_masks == cell_id

                # Find contours using skimage
                contours = measure.find_contours(cell_mask, 0.5)

                if len(contours) > 0:
                    # Take the longest contour if there are multiple
                    contour = contours[0]

                    # Adjust coordinates to global image coordinates
                    adjusted_contour = [(y + i, x + j) for y, x in contour]

                    # Create cell entry
                    for y, x in adjusted_contour:
                        cells.append({
                            'global_cell_id': global_cell_id,
                            'original_cell_id': cell_id,
                            'x': x,
                            'y': y
                        })

                    global_cell_id += 1

    # Create and save DataFrame
    df = pd.DataFrame(cells)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {global_cell_id} cell outlines to {output_csv}")

    return df
def process_large_image_distributed(image_path, output_zarr_path, csv_path, blocksize=(1024, 1024), channel=0):
    """
    Process a large image using distributed cellpose
    """
    # Set up logging
    logger = logging.getLogger(__name__)

    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting distributed image processing at {start_time}")

    try:
        # First, convert the input image to zarr format
        logger.info("Converting input image to zarr format...")

        # Get image info without loading the entire file
        with tifffile.TiffFile(image_path) as tiff:
            # Get image shape
            data_shape = tiff.series[0].shape
            logger.info(f"Image shape: {data_shape}")

        # Check if the image is multi-dimensional
        is_multichannel = len(data_shape) > 2
        logger.info(f"Multi-channel image: {is_multichannel}")

        # Read the image and store in zarr
        logger.info("Reading image to zarr...")
        input_zarr_path = output_zarr_path + '_input.zarr'
        input_zarr = read_tiff_to_zarr(image_path, input_zarr_path, blocksize)

        # For multi-channel images, we need to extract the channel we want to process
        if is_multichannel:
            logger.info(f"Extracting channel {channel} for processing")
            channel_zarr_path = output_zarr_path + f'_channel{channel}.zarr'
            processing_zarr = extract_channel_from_zarr(input_zarr, channel_zarr_path, channel, blocksize)
        else:
            # For single-channel images, use the input zarr directly
            processing_zarr = input_zarr

        # Define cellpose parameters
        model_kwargs = {
            'gpu': True,
            'pretrained_model': "/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700"
        }

        eval_kwargs = {
            'diameter': 120,
            'flow_threshold': 0.8,
            'cellprob_threshold': -1.5,
            'channels': [0, 0]  # Single channel mode
        }

        # Define cluster parameters for local workstation
        cluster_kwargs = {
            'n_workers': 1,  # Adjust based on number of GPUs
            'ncpus': 48,     # Adjust based on your system
            'memory_limit': '256GB',
            'threads_per_worker': 1,
            'multiprocessing': False  # Try setting this to False
        }

        # Optional preprocessing step for normalization
        def normalize_block(image, crop):
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-7) * 255
            return normalized.astype(np.uint8)

        preprocessing_steps = [(normalize_block, {})]

        # Run distributed segmentation
        logger.info("Running distributed segmentation...")

        # Import here to avoid issues if not available
        from cellpose.contrib.distributed_segmentation import distributed_eval

        segments, boxes = distributed_eval(
            input_zarr=processing_zarr,
            blocksize=blocksize,
            write_path=output_zarr_path,
            preprocessing_steps=preprocessing_steps,
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            cluster_kwargs=cluster_kwargs,
            overlap=60  # Add overlap for better stitching
        )

        # Extract cell outlines from the segmentation results
        logger.info("Extracting cell outlines...")
        get_cell_outlines_from_zarr(segments, csv_path)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        logger.info(f"Processing completed in {total_duration:.2f} seconds")

        # Return the paths of created files
        return {
            'segmentation_zarr': output_zarr_path,
            'cell_outlines_csv': csv_path,
            'bounding_boxes': boxes
        }

    except Exception as e:
        logger.error(f"Error during distributed processing: {e}", exc_info=True)
        # Return a minimal result dictionary to avoid NameError
        return {
            'error': str(e),
            'segmentation_zarr': output_zarr_path if 'output_zarr_path' in locals() else None,
            'cell_outlines_csv': csv_path if 'csv_path' in locals() else None
        }
def process_image_direct(image_path, output_zarr_path, csv_path, blocksize=(1024, 1024), channel=0):
    """
    Process image directly with Cellpose without distributed framework
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting direct image processing (non-distributed)")

    try:
        # First, convert the input image to zarr format
        logger.info("Converting input image to zarr format...")

        # Get image info without loading the entire file
        with tifffile.TiffFile(image_path) as tiff:
            # Get image shape
            data_shape = tiff.series[0].shape
            logger.info(f"Image shape: {data_shape}")

        # Check if the image is multi-dimensional
        is_multichannel = len(data_shape) > 2
        logger.info(f"Multi-channel image: {is_multichannel}")

        # Read the image and store in zarr
        logger.info("Reading image to zarr...")
        input_zarr_path = output_zarr_path + '_input.zarr'
        input_zarr = read_tiff_to_zarr(image_path, input_zarr_path, blocksize)

        # For multi-channel images, we need to extract the channel we want to process
        if is_multichannel:
            logger.info(f"Extracting channel {channel} for processing")
            channel_zarr_path = output_zarr_path + f'_channel{channel}.zarr'
            processing_zarr = extract_channel_from_zarr(input_zarr, channel_zarr_path, channel, blocksize)
        else:
            # For single-channel images, use the input zarr directly
            processing_zarr = input_zarr

        # Process the image in chunks
        logger.info("Processing image in chunks...")

        # Create output zarr array
        segments = zarr.open(
            output_zarr_path,
            mode='w',
            shape=processing_zarr.shape,
            chunks=blocksize,
            dtype=np.uint16,
            compressor=None
        )

        # Load Cellpose model
        from cellpose import models
        model = models.Cellpose(pretrained_model="/home/users/allstaff/kriel.j/.cellpose/models/CP_20240119_101700", gpu=True)
        # Process in chunks
        max_cell_id = 0
        for i in range(0, processing_zarr.shape[0], blocksize[0]):
            i_end = min(i + blocksize[0], processing_zarr.shape[0])
            for j in range(0, processing_zarr.shape[1], blocksize[1]):
                j_end = min(j + blocksize[1], processing_zarr.shape[1])

                # Get chunk
                chunk = processing_zarr[i:i_end, j:j_end]

                # Normalize chunk
                chunk_norm = (chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-7) * 255
                chunk_norm = chunk_norm.astype(np.uint8)

                # Run Cellpose on chunk
                masks, flows, styles, diams = model.eval(
                    chunk_norm,
                    diameter=120,
                    flow_threshold=0.8,
                    cellprob_threshold=-1.5,
                    channels=[1, 2]
                )

                # Offset cell IDs to avoid duplicates
                masks[masks > 0] += max_cell_id
                max_cell_id = masks.max() if masks.max() > max_cell_id else max_cell_id

                # Save to output zarr
                segments[i:i_end, j:j_end] = masks

                logger.info(f"Processed chunk ({i}:{i_end}, {j}:{j_end}), found {len(np.unique(masks)) - 1} cells")

        # Extract cell outlines
        logger.info("Extracting cell outlines...")
        get_cell_outlines_from_zarr(segments, csv_path)

        return {
            'segmentation_zarr': output_zarr_path,
            'cell_outlines_csv': csv_path
        }

    except Exception as e:
        logger.error(f"Error in direct processing: {e}", exc_info=True)
        return {
            'error': str(e),
            'segmentation_zarr': output_zarr_path,
            'cell_outlines_csv': csv_path
        }
def create_interactive_plots_html(image_path, zarr_path, csv_path, output_dir, region=None, channel=0):
    """
    Create interactive HTML plots of the segmentation results and cell statistics

    Parameters:
    -----------
    image_path : str
        Path to the input image
    zarr_path : str
        Path to the zarr file containing segmentation masks
    csv_path : str
        Path to the CSV file containing cell outlines
    output_dir : str
        Directory where HTML files will be saved
    region : tuple, optional
        Region to plot as ((y_start, y_end), (x_start, x_end)). If None, plots entire image
    channel : int
        Channel to display for multi-channel images
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read the data
    print("Reading input data...")

    # Get image info without loading the entire file
    with tifffile.TiffFile(image_path) as tiff:
        # Get image shape
        img_shape = tiff.series[0].shape
        is_multichannel = len(img_shape) > 2

        # For multi-channel images, adjust the shape
        if is_multichannel:
            img_shape = img_shape[1:]  # Remove the channel dimension

    # For large images, read only the region of interest or a downsampled version
    if region is not None:
        (y_start, y_end), (x_start, x_end) = region

        # Read the specific region from the TIFF file
        with tifffile.TiffFile(image_path) as tiff:
            if is_multichannel:
                # For multi-channel TIFFs, read the specified channel
                page = tiff.pages[channel]  # Get the page for this channel
                page_data = page.asarray()
                original_image = page_data[y_start:y_end, x_start:x_end]
            else:
                # For single-channel TIFFs
                page = tiff.pages[0]
                page_data = page.asarray()
                original_image = page_data[y_start:y_end, x_start:x_end]

        # Read the same region from the zarr segmentation
        segmentation_masks = zarr.open(zarr_path, mode='r')[y_start:y_end, x_start:x_end]

        # Filter cell outlines to the region
        cell_outlines = pd.read_csv(csv_path)
        cell_outlines = cell_outlines[
            (cell_outlines['y'] >= y_start) & (cell_outlines['y'] < y_end) &
            (cell_outlines['x'] >= x_start) & (cell_outlines['x'] < x_end)
        ].copy()

        # Adjust coordinates
        cell_outlines['x'] = cell_outlines['x'] - x_start
        cell_outlines['y'] = cell_outlines['y'] - y_start

        region_str = f"_region_{y_start}_{y_end}_{x_start}_{x_end}"
    else:
        # For viewing the entire image, use a downsampled version to avoid memory issues
        # Calculate downsampling factor based on image size
        total_pixels = np.prod(img_shape)
        downsample = max(1, int(np.sqrt(total_pixels / 1e6)))  # Target ~1 million pixels
        print(f"Downsampling image by factor of {downsample} for visualization")

        # Read downsampled image
        with tifffile.TiffFile(image_path) as tiff:
            if is_multichannel:
                # For multi-channel TIFFs, read the specified channel
                page = tiff.pages[channel]  # Get the page for this channel
                page_data = page.asarray()
                # Downsample by taking every Nth pixel
                original_image = page_data[::downsample, ::downsample]
            else:
                # For single-channel TIFFs
                page = tiff.pages[0]
                page_data = page.asarray()
                original_image = page_data[::downsample, ::downsample]

        # Also downsample segmentation masks
        segmentation_masks = zarr.open(zarr_path, mode='r')[::downsample, ::downsample]

        # Read cell outlines
        cell_outlines = pd.read_csv(csv_path)

        # Adjust coordinates for downsampling
        cell_outlines['x'] = cell_outlines['x'] / downsample
        cell_outlines['y'] = cell_outlines['y'] / downsample

        region_str = "_full"

    # Create main visualization
    print("Creating main visualization...")
    fig_main = make_subplots(rows=1, cols=2,
                            subplot_titles=('Original Image with Cell Outlines',
                                          'Segmentation Masks'),
                            horizontal_spacing=0.05)

    # Add original image
    fig_main.add_trace(
        go.Heatmap(
            z=original_image,
            colorscale='gray',
            showscale=False,
            name='Original Image'
        ),
        row=1, col=1
    )

    # Add segmentation masks
    fig_main.add_trace(
        go.Heatmap(
            z=segmentation_masks,
            colorscale='Viridis',
            showscale=True,
            name='Segmentation Masks'
        ),
        row=1, col=2
    )

    # Add cell outlines
    print("Adding cell outlines...")
    for cell_id in cell_outlines['global_cell_id'].unique():
        cell_data = cell_outlines[cell_outlines['global_cell_id'] == cell_id]

        fig_main.add_trace(
            go.Scatter(
                x=cell_data['x'],
                y=cell_data['y'],
                mode='lines',
                line=dict(color='red', width=1),
                name=f'Cell {cell_id}',
                showlegend=False,
                hoverinfo='text',
                text=f'Cell ID: {cell_id}'
            ),
            row=1, col=1
        )

    # Update main visualization layout
    fig_main.update_layout(
        title='Cell Segmentation Results',
        height=800,
        width=1600,
        showlegend=False,
        template='plotly_white'
    )

    # Update axes
    fig_main.update_xaxes(scaleanchor="y", scaleratio=1)
    fig_main.update_yaxes(scaleanchor="x", scaleratio=1)

    # Calculate cell statistics
    print("Calculating cell statistics...")
    cell_stats = []
    for cell_id in cell_outlines['global_cell_id'].unique():
        cell_data = cell_outlines[cell_outlines['global_cell_id'] == cell_id]

        x_coords = cell_data['x'].values
        y_coords = cell_data['y'].values

        # Calculate cell properties
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        area = np.abs(np.trapz(y_coords, x_coords))
        perimeter = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0

        cell_stats.append({
            'cell_id': cell_id,
            'center_x': center_x,
            'center_y': center_y,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        })

    cell_stats_df = pd.DataFrame(cell_stats)

    # Create statistics visualization
    print("Creating statistics visualization...")
    fig_stats = make_subplots(rows=2, cols=2,
                             subplot_titles=('Cell Distribution',
                                           'Area Distribution',
                                           'Perimeter vs Area',
                                           'Circularity Distribution'))

    # Cell spatial distribution
    fig_stats.add_trace(
        go.Scatter(
            x=cell_stats_df['center_x'],
            y=cell_stats_df['center_y'],
            mode='markers',
            marker=dict(
                size=5,
                color=cell_stats_df['area'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Area')
            ),
            name='Cell Centers'
        ),
        row=1, col=1
    )

    # Area histogram
    fig_stats.add_trace(
        go.Histogram(
            x=cell_stats_df['area'],
            name='Area Distribution',
            nbinsx=50
        ),
        row=1, col=2
    )

    # Perimeter vs Area scatter plot
    fig_stats.add_trace(
        go.Scatter(
            x=cell_stats_df['area'],
            y=cell_stats_df['perimeter'],
            mode='markers',
            marker=dict(
                size=5,
                color=cell_stats_df['circularity'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Circularity')
            ),
            name='Perimeter vs Area'
        ),
        row=2, col=1
    )

    # Circularity histogram
    fig_stats.add_trace(
        go.Histogram(
            x=cell_stats_df['circularity'],
            name='Circularity Distribution',
            nbinsx=50
        ),
        row=2, col=2
    )

    # Update statistics layout
    fig_stats.update_layout(
        height=1000,
        width=1600,
        showlegend=False,
        template='plotly_white',
        title='Cell Population Statistics'
    )

    # Save summary statistics to CSV
    stats_summary = cell_stats_df.describe()
    csv_summary_path = os.path.join(output_dir, f'cell_statistics_summary_{timestamp}{region_str}.csv')
    stats_summary.to_csv(csv_summary_path)

    # Save HTML files
    print("Saving HTML files...")
    main_html_path = os.path.join(output_dir, f'segmentation_visualization_{timestamp}{region_str}.html')
    stats_html_path = os.path.join(output_dir, f'cell_statistics_{timestamp}{region_str}.html')

    fig_main.write_html(main_html_path)
    fig_stats.write_html(stats_html_path)

    # Create an index HTML file that links to both visualizations
    index_html_path = os.path.join(output_dir, f'index_{timestamp}{region_str}.html')
    with open(index_html_path, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>Cell Segmentation Analysis - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .link-container {{ margin: 20px 0; }}
                a {{ color: #0066cc; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Cell Segmentation Analysis</h1>
            <div class="link-container">
                <h2>Visualizations:</h2>
                <p><a href="{os.path.basename(main_html_path)}" target="_blank">Segmentation Results</a></p>
                <p><a href="{os.path.basename(stats_html_path)}" target="_blank">Cell Statistics</a></p>
            </div>
            <div class="link-container">
                <h2>Data:</h2>
                <p><a href="{os.path.basename(csv_summary_path)}" target="_blank">Statistical Summary (CSV)</a></p>
            </div>
            <div class="link-container">
                <h2>Analysis Information:</h2>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Image: {os.path.basename(image_path)}</p>
                <p>Channel: {channel}</p>
                {f'<p>Region: {region}</p>' if region else '<p>Region: Full image</p>'}
            </div>
        </body>
        </html>
        """)

    print("HTML files generated successfully!")
    return {
        'index': index_html_path,
        'main_visualization': main_html_path,
        'statistics': stats_html_path,
        'summary_csv': csv_summary_path
    }

# Test the functions with a small region to verify they work
# This is just to demonstrate the code works, not to process the full image
def test_tiff_reading():
    """Test reading a small region of the TIFF file"""
    image_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_combined_af_merged_cropped_small_roi.ome.tif"
    
    # Get image info
    with tifffile.TiffFile(image_path) as tiff:
        shape = tiff.series[0].shape
        dtype = tiff.series[0].dtype
        print(f"TIFF shape: {shape}, dtype: {dtype}")
        
        # Read a small region from the first channel
        if len(shape) == 3:  # 3D array (channels, height, width)
            channel = 0
            y_start, y_end = 0, min(1024, shape[1])
            x_start, x_end = 0, min(1024, shape[2])
            
            # Read the page for this channel
            page = tiff.pages[channel]
            page_data = page.asarray()
            region = page_data[y_start:y_end, x_start:x_end]
            
            print(f"Successfully read region from channel {channel}: shape={region.shape}")
            return True
        else:
            print("Not a multi-channel TIFF")
            return False



if __name__ == "__main__":
    freeze_support()  # This is needed for Windows

    try:
        # Set up dask client
        client = Client()
        logger.info(f"Dask client started: {client}")

        # Define paths
        image_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/dapi_combined_af_merged_cropped_small_roi.ome.tif"
        output_zarr_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/Venture5_cell_test_outlines.zarr"
        csv_path = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/Venture5_cell_test_outlines.csv"
        output_dir = "/vast/scratch/users/kriel.j/venture_pt5/Jurgen/ome/segmentation_visualization"

        logger.info("Starting distributed image processing...")

        # Process the image with distributed cellpose
        try:
            results = process_large_image_distributed(
                image_path,
                output_zarr_path,
                csv_path,
                blocksize=(1024, 1024),
                channel=0  # Use the first channel (index 0)
            )
        except Exception as e:
            logger.error(f"Error in process_large_image_distributed: {e}", exc_info=True)
            # Create a default results dictionary to avoid NameError
            results = {
                'error': str(e),
                'segmentation_zarr': output_zarr_path,
                'cell_outlines_csv': csv_path
            }

        # Check if results is None or not defined
        if results is None:
            logger.error("Processing completed but returned no results")
            # Create a default results dictionary
            results = {
                'error': "No results returned",
                'segmentation_zarr': output_zarr_path,
                'cell_outlines_csv': csv_path
            }

        # Print created files
        logger.info("\nCreated/Modified files during execution:")
        for key, path in results.items():
            logger.info(f"{key}: {path}")

        # Check if segmentation was successful before creating visualizations
        if 'error' in results and not os.path.exists(output_zarr_path):
            logger.error("Skipping visualization as segmentation failed")
        else:
            # Create HTML visualizations
            try:
                logger.info("\nGenerating HTML visualizations...")
                html_results = create_interactive_plots_html(
                    image_path,
                    output_zarr_path,
                    csv_path,
                    output_dir,
                    channel=0  # Use the first channel for visualization
                )

                if html_results:
                    logger.info("\nHTML files created:")
                    for key, path in html_results.items():
                        logger.info(f"{key}: {path}")
                else:
                    logger.warning("No HTML files were created")
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}", exc_info=True)
    finally:
        # Close the client
        try:
            client.close()
            logger.info("Dask client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")
