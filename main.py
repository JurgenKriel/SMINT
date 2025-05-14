import os
import sys
import importlib
import json
import logging
import base64
from io import BytesIO
from flask import Flask, render_template, redirect, url_for, request, jsonify, abort

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Check if SMINT is available
SMINT_AVAILABLE = False
SMINT_VERSION = None
SMINT_MODULES = []

try:
    import smint
    SMINT_AVAILABLE = True
    SMINT_VERSION = smint.__version__
    SMINT_MODULES = smint.__all__
except ImportError:
    logger.warning("SMINT package not available")

# Check optional dependencies
DEPENDENCIES = {
    "OpenCV": False,
    "Cellpose": False,
    "Dask": False,
    "Distributed": False,
    "Dask-CUDA": False,
    "NumPy": False,
    "Pandas": False,
    "Matplotlib": False,
    "scikit-image": False,
    "tifffile": False
}

def check_dependency(package_name, import_name=None):
    if import_name is None:
        import_name = package_name.lower()
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

# Update dependency status
DEPENDENCIES["OpenCV"] = check_dependency("OpenCV", "cv2")
DEPENDENCIES["Cellpose"] = check_dependency("Cellpose", "cellpose")
DEPENDENCIES["Dask"] = check_dependency("Dask", "dask")
DEPENDENCIES["Distributed"] = check_dependency("Distributed", "distributed")
DEPENDENCIES["Dask-CUDA"] = check_dependency("Dask-CUDA", "dask_cuda")
DEPENDENCIES["NumPy"] = check_dependency("NumPy", "numpy")
DEPENDENCIES["Pandas"] = check_dependency("Pandas", "pandas")
DEPENDENCIES["Matplotlib"] = check_dependency("Matplotlib", "matplotlib")
DEPENDENCIES["scikit-image"] = check_dependency("scikit-image", "skimage")
DEPENDENCIES["tifffile"] = check_dependency("tifffile")

# Documentation structure
DOCUMENTATION = {
    "segmentation": {
        "title": "Segmentation",
        "description": "Cell segmentation tools for microscopy images",
        "functions": [
            {
                "name": "process_large_image",
                "description": "Segment cells in a large microscopy image",
                "parameters": [
                    {"name": "image_path", "type": "str", "description": "Path to the input image"},
                    {"name": "csv_path", "type": "str", "description": "Path to save segmentation results"},
                    {"name": "chunk_size", "type": "tuple", "description": "Size of chunks for processing"},
                    {"name": "model_path", "type": "str", "description": "Path to cellpose model"}
                ]
            },
            {
                "name": "segment_chunk",
                "description": "Segment a single chunk of an image",
                "parameters": [
                    {"name": "chunk", "type": "ndarray", "description": "Image chunk to segment"},
                    {"name": "model", "type": "CellposeModel", "description": "Cellpose model instance"}
                ]
            }
        ]
    },
    "alignment": {
        "title": "Alignment",
        "description": "Tools for aligning spatial omics data",
        "functions": [
            {
                "name": "run_alignment",
                "description": "Align spatial transcriptomics data",
                "parameters": [
                    {"name": "source_data", "type": "DataFrame", "description": "Source data"},
                    {"name": "target_data", "type": "DataFrame", "description": "Target data"},
                    {"name": "method", "type": "str", "description": "Alignment method"}
                ]
            },
            {
                "name": "transform_coordinates",
                "description": "Transform coordinates using alignment matrix",
                "parameters": [
                    {"name": "coordinates", "type": "DataFrame", "description": "Coordinates to transform"},
                    {"name": "transformation_matrix", "type": "ndarray", "description": "Transformation matrix"}
                ]
            }
        ]
    },
    "visualization": {
        "title": "Visualization",
        "description": "Tools for visualizing segmentation results and spatial data",
        "functions": [
            {
                "name": "create_rgb_composite",
                "description": "Create RGB composite from multiple channels",
                "parameters": [
                    {"name": "image_data", "type": "ndarray", "description": "Multi-channel image data"},
                    {"name": "channel_indices", "type": "tuple", "description": "Channel indices for RGB"}
                ]
            },
            {
                "name": "visualize_cell_outlines",
                "description": "Visualize cell outlines on an image",
                "parameters": [
                    {"name": "image", "type": "ndarray", "description": "Background image"},
                    {"name": "cell_outlines", "type": "list", "description": "List of cell outline coordinates"}
                ]
            }
        ]
    }
}

# Simple examples
EXAMPLES = [
    {
        "title": "Basic Cell Segmentation",
        "description": "Segment cells in a microscopy image using Cellpose",
        "code": """
import smint
from smint.segmentation import process_large_image

# Run segmentation on a whole-slide image
process_large_image(
    image_path="sample.ome.tiff",
    csv_path="segmentation_results.csv",
    chunk_size=(2048, 2048),
    model_path="cyto"
)
        """
    },
    {
        "title": "Spatial Transcriptomics Alignment",
        "description": "Align spatial transcriptomics data from different modalities",
        "code": """
import smint
import pandas as pd
from smint.alignment import run_alignment, transform_coordinates

# Load spatial data
source_data = pd.read_csv("source_spots.csv")
target_data = pd.read_csv("target_spots.csv")

# Run alignment
alignment_result = run_alignment(
    source_data=source_data,
    target_data=target_data,
    method="similarity"
)

# Transform coordinates
transformed_coords = transform_coordinates(
    coordinates=source_data,
    transformation_matrix=alignment_result["transformation_matrix"]
)
        """
    },
    {
        "title": "Multi-GPU Segmentation",
        "description": "Distribute segmentation across multiple GPUs",
        "code": """
import smint
from smint.segmentation.distributed_seg import process_large_image_distributed

# Run distributed segmentation
process_large_image_distributed(
    image_path="large_sample.ome.tiff",
    output_zarr_path="segmentation_results.zarr",
    csv_path="segmentation_results.csv",
    blocksize=(1024, 1024),
    channel=0
)
        """
    },
    {
        "title": "Xenium-Metabolomics Alignment",
        "description": "Align 10X Xenium spatial transcriptomics with spatial metabolomics data",
        "code": """
import smint
from smint.alignment import align_xenium_to_metabolomics

# Basic alignment
aligned_data = align_xenium_to_metabolomics(
    xenium_file="xenium_cell_data.csv",
    metabolomics_file="metabolomics_data.csv",
    output_dir="alignment_results",
    pixel_size=30,
    visualize=True
)

# Advanced alignment with custom parameters
aligned_data = align_xenium_to_metabolomics(
    xenium_file="xenium_cell_data.csv",
    metabolomics_file="metabolomics_data.csv",
    output_dir="alignment_results",
    pixel_size=30,
    xenium_x_col="x_centroid",
    xenium_y_col="y_centroid",
    met_x_col="x",
    met_y_col="y",
    lddmm_params={
        'niter': 1500,
        'sigmaM': 0.3,
        'epV': 600
    },
    visualize=True
)
        """
    }
]

def get_placeholder_image():
    """Generate a simple placeholder image when matplotlib is not available"""
    # ASCII art of a cell
    return """
    +---------+
    |         |
    |   Cell  |
    |         |
    +---------+
    """

def generate_sample_cell_visualization():
    """Generate a sample cell visualization"""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Background
        background = np.zeros((100, 100, 3))
        background[:, :, 0] = 0.1  # Red channel
        background[:, :, 1] = 0.2  # Green channel
        background[:, :, 2] = 0.3  # Blue channel
        
        # Display background
        ax.imshow(background)
        
        # Add cell outlines
        for i in range(10):
            x = np.random.randint(20, 80)
            y = np.random.randint(20, 80)
            r = np.random.randint(5, 15)
            circle = Circle((x, y), r, fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"{i+1}", color='white', ha='center', va='center')
        
        # Set title and labels
        ax.set_title('Sample Cell Segmentation')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        
        # Save the figure to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        # Convert the buffer to a base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    except ImportError:
        logger.warning("Cannot generate visualization: matplotlib or numpy not available")
        return None

@app.route('/')
def index():
    return render_template('index.html', 
                          smint_available=SMINT_AVAILABLE,
                          smint_version=SMINT_VERSION,
                          smint_modules=SMINT_MODULES,
                          dependencies=DEPENDENCIES)

@app.route('/documentation')
def documentation():
    return render_template('documentation.html',
                          documentation=DOCUMENTATION,
                          smint_available=SMINT_AVAILABLE)

@app.route('/examples')
def examples():
    return render_template('examples.html',
                          examples=EXAMPLES,
                          smint_available=SMINT_AVAILABLE)

@app.route('/demo')
def demo():
    cell_viz = generate_sample_cell_visualization()
    return render_template('demo.html',
                          cell_visualization=cell_viz,
                          has_visualization=(cell_viz is not None),
                          dependencies=DEPENDENCIES,
                          smint_available=SMINT_AVAILABLE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
