
# SMINT - Spatial Multi-Omics Integration Toolkit

SMINT is a Python package for Spatial Multi-Omics Integration with enhanced segmentation capabilities and streamlined workflow. It provides a comprehensive toolkit for processing, analyzing, and visualizing spatial multi-omics data.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jurgenkriel.github.io/SMINT/)

## Features

- Multi-GPU cell segmentation for whole-slide images
- Distributed segmentation using Dask for improved performance
- Live segmentation monitoring with intuitive visualization tools
- Streamlined alignment workflow using ST Align
- Integration with R analysis scripts via rpy2
- Graceful dependency handling for all optional components
- Comprehensive documentation with step-by-step guides
- HPC deployment scripts for SLURM-based clusters

## Documentation

Comprehensive documentation is available at [https://jurgenkriel.github.io/SMINT/](https://jurgenkriel.github.io/SMINT/)

## Installation

```bash
pip install smint
```

## Optional Dependencies

SMINT is designed with graceful dependency handling to ensure robust operation across different environments.

```bash
# Full installation with all dependencies
pip install "smint[all]"

# Install specific feature sets
pip install "smint[segmentation]"  # For segmentation functionality
pip install "smint[visualization]"  # For visualization functionality
pip install "smint[r_integration]"  # For R integration
```

For more information, see the [Dependency Handling Documentation](https://jurgenkriel.github.io/SMINT/dependency_handling/).

## Linking to GitHub Repository

This project is linked to the GitHub repository at [https://github.com/JurgenKriel/SMINT](https://github.com/JurgenKriel/SMINT).

### Pushing to GitHub

To contribute changes to the GitHub repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/JurgenKriel/SMINT.git
   cd SMINT
   ```

2. Make your changes and test them locally.

3. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin main
   ```

### Documentation Deployment

Documentation is automatically deployed to GitHub Pages using GitHub Actions. The workflow file is located at `.github/workflows/deploy-docs.yml`.

When you push changes to the `main` branch, the documentation will be automatically built and deployed to [https://jurgenkriel.github.io/SMINT/](https://jurgenkriel.github.io/SMINT/).

To build documentation locally for testing:

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings
mkdocs serve
```

## Usage Examples

### Cell Segmentation

```python
import smint
from smint.segmentation import process_large_image

# Run cell segmentation on a whole-slide image
process_large_image(
    image_path="path/to/whole_slide_image.ome.tiff",
    csv_base_path="path/to/output_directory",
    cell_model_path="cyto",  # Use pre-trained Cellpose model
    cells_diameter=120.0,
    cells_channels=[1, 2],  # Green and Blue channels for cells
    nuclei_model_path="nuclei",  # Use pre-trained nuclei model
    nuclei_diameter=60.0,
    nuclei_channels=[2, 1],  # Blue and Green channels for nuclei
    visualize=True  # Generate visualization images
)
```

### Visualization

```python
from smint.visualization import visualize_segmentation_results

# Create interactive visualization of segmentation results
visualize_segmentation_results(
    image_path="path/to/whole_slide_image.ome.tiff",
    segmentation_csv="path/to/segmentation_results.csv",
    output_html="path/to/output/visualization.html",
    thumbnail_size=(800, 800)
)
```

### R Integration

```python
from smint.r_integration import run_spatialomics_analysis

# Run spatial statistics analysis using R
run_spatialomics_analysis(
    segmentation_data="path/to/segmentation_results.csv",
    expression_data="path/to/gene_expression.csv",
    output_directory="path/to/output",
    spatial_methods=["Ripley", "Morans"]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
=======
# SMINT: Spatial Multi-Omics Integration

![Workflow Image](https://github.com/JurgenKriel/SMINT/raw/main/SpatialSegPaper_v2.png)

# Workflows

## Cell Segmentation workflow

### 1. Setup conda environment

```
   conda env create -f conda_env_setup.yml
```
### 2. Export images for training

If using a Zeiss Confocal Instrument, tiles can be exported direclty from the Zen Blue software using 'export selected tiles' option. 
If you already have a stiched tile scan, you can chunk it up with specified tile sizes using the following: 

```python

# Read the TIFF image file
image = aicsimageio.AICSImage(image_path)

# Get the dimensions of the image
width, height = image.shape[-2:]

# Define the tile size
tile_size = 1024

# Calculate the number of tiles in the horizontal and vertical directions
num_tiles_horizontal = (width + tile_size - 1) // tile_size
num_tiles_vertical = (height + tile_size - 1) // tile_size

# Create a directory to save the new images
output_dir = "/path/to/output/directory/"
os.makedirs(output_dir, exist_ok=True)

# Chunk up the image into tiles
for y in range(0, height, tile_size):
    for x in range(0, width, tile_size):
        # Calculate the coordinates of the current tile
        left = x
        upper = y
        right = min(x + tile_size, width)
        lower = min(y + tile_size, height)

        # Crop the image to extract the current tile
        tile = image.data[0, 0, left:right, upper:lower, :]

        # Save the tile as a new image
        tile_path = os.path.join(output_dir, f"tile_{x}_{y}.tif")
        OmeTiffWriter.save(tile,tile_path,dim_order="CYX")

        print(f"Saved tile {x}_{y}: {tile_path}")
```


### 3. Train Cellpose Model 

For coherent guidelines on training a Cellpose Model from scratch, please see: https://cellpose.readthedocs.io/en/latest/train.html

For label creation, we started with the pre-trained CP model, mannually corrected segmentations, then exported the seg.npy files for training.
Full ipython notebook available under segmentation/Cellpose_Train_and_Seg.ipynb

```python
#train models on image tiles 

#start logger (to see training across epochs)
logger = io.logger_setup()

# DEFINE CELLPOSE MODEL (without size model)
CP2='path/to/pretrained/model/CP'
model=models.CellposeModel(gpu=use_GPU,pretrained_model=CP2)
train_dir='/path/to/images/and/masks/'
test_dir='/path/to/test/images'
#model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

# set channels
channels = [1,2]

# get files
output = io.load_train_test_data(train_dir, test_dir, mask_filter='_seg.npy')
train_files, train_seg, _, test_data, test_labels, _ = output

new_model_path = model.train(train_files, train_seg, 
                              test_data=test_data,
                              test_labels=test_labels,
                              channels=channels, 
                              save_path=train_dir, 
                              n_epochs=1000,
                              learning_rate=0.01, 
                              weight_decay=0.00001, 
                              nimg_per_epoch=1,
                              model_name=CP2)

# diameter of labels in training images
diam_labels = model.diam_labels.copy()

```

### 4. Run segmentation on all tiles

```python
image_files = glob('/path/to/images/*.tif')

#sort images as list
imgs=[]
for file in image_files:
    img=io.imread(file)
    imgs.append(img)

nimg=len(imgs)
nimg

#Run segmentation with CP2 model

channels = [1,2]
masks, flows, diams = model.eval(imgs, diameter=100.53, flow_threshold=3,cellprob_threshold=-3, channels=channels, min_size=99, normalize=True)
io.masks_flows_to_seg(imgs, masks, flows, diams, train_files, channels=channels)

io.save_masks(imgs, masks, flows, train_files, save_txt=True)
```

### 5. Export tile coordinates
The aicsimageio module provides a handy get_mosaic_tile_positions function to export tile positions:
```python
z8="/path/to/czi/image.czi/"
z8_img=aicsimageio.AICSImage(z8)
mosaic=z8_img.get_mosaic_tile_positions(C=0)
np.savetxt('/path/to/tiles.csv',mosaic, delimiter=',', fmt='%s')
```


## Transcriptomics workflow  

### 1. Stitch Polygons
See stitching script under stitching/stitching.Rmd

### 2. Align and Assign Transcripts to polygons
See alignment of polygons and transcripts under
See transcript assignment script under assignment/transcript_assignment.R

## Alignment Workflow

### 1. Setup STalign

```
pip install --upgrade "git+https://github.com/JEFworks-Lab/STalign.git"
```

### 2. Align ST to SM
See alignment script under alignment/ST_align_xenium.py


## Integration 

### 1. Metabolomics Cluster Analysis 
See metabolite assignment script under integration/metabolomics_integration.R
>>>>>>> e9fadf4d662d283dafd59ae756781bbe1e40863c
