# SMINT: Spatial Multi-Omics Integration

![Workflow Image](https://github.com/JurgenKriel/SMINT/raw/main/SpatialSegPaper_v2.png)

# Workflows

## Cell Segmentation workflow

1. Setup conda environment

```
   conda env create -f environment.yml
```
3. Export images for training

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
output_dir = "/stornext/Bioinf/data/lab_brain_cancer/users/j_kriel/Confocal/Venture_2_1/output_tiles/"
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


5. Train Cell Pose Model 

6. Run segmentation on all tiles

7. Export tile coordinates

## Transcriptomics workflow  

1. Stitch Polygons

2. Align Transcripts to polygons

## Alignment Workflow

1. Setup STalign

2. Align ST to SM

## Integration 

1. Metabolomics Cluster Analysis 
