# SMINT: Spatial Multi-Omics Integration

![Workflow Image](https://github.com/JurgenKriel/SMINT/raw/main/SpatialSegPaper_v2.png)

# Workflows

## Cell Segmentation workflow

1. Setup conda environment

```
   conda env create -f environment.yml
```
2. Export images for training

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


3. Train Cellpose Model 

For coherent guidelines on training a Cellpose Model from scratch, please see: https://cellpose.readthedocs.io/en/latest/train.html

For label creation, we started with the pre-trained CP model, and mannually corrected segmentations, then used the seg.npy files for training:

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

4. Run segmentation on all tiles

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

5. Export tile coordinates



## Transcriptomics workflow  

1. Stitch Polygons

2. Align Transcripts to polygons

## Alignment Workflow

1. Setup STalign

2. Align ST to SM

## Integration 

1. Metabolomics Cluster Analysis 
