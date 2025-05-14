# Spatial Alignment

SMINT provides a streamlined workflow for aligning different types of spatial omics data using the ST Align algorithm. This guide covers the complete alignment process, from data preparation to result validation.

## Overview of Alignment Workflow

The SMINT alignment pipeline consists of these key stages:

1. **Data Preparation** - Converting and preprocessing spatial data
2. **Reference Selection** - Choosing appropriate reference points
3. **Alignment Computation** - Calculating the optimal transformation
4. **Transform Application** - Applying the transformation to target data
5. **Validation & Quality Control** - Assessing alignment accuracy

## Input Data Requirements

### Supported Data Formats

SMINT's alignment module supports the following data formats:
- **CSV files** (preferred) - Simple tabular format with spatial coordinates
- **AnnData objects** - Python objects with spatial omics data
- **Pandas DataFrames** - In-memory tabular data
- **10X Xenium data** - Spatial transcriptomics from 10X Genomics

### Required Data Columns

For optimal alignment, input data files should contain:
- **Spatial coordinates**: Columns named 'x' and 'y' or similar ('X_position', 'Y_position')
- **Feature values**: Gene expression or other features (optional, for feature-based alignment)
- **Cell/spot IDs**: Unique identifiers for each point (optional, but recommended)

### Example Input Data

Reference data (`reference_data.csv`):
```
spot_id,x,y,feature1,feature2
1,100.5,200.3,0.8,0.2
2,150.2,220.1,0.6,0.4
...
```

Target data (`target_data.csv`):
```
cell_id,x_position,y_position,marker1,marker2
cell_1,1050.5,2200.3,0.9,0.1
cell_2,1150.2,2220.1,0.7,0.3
...
```

## Quick Start

### Basic Alignment

```bash
python -m scripts.run_alignment \
    --reference reference_data.csv \
    --target target_data.csv \
    --output-dir results/alignment \
    --method affine \
    --ref-x-col x \
    --ref-y-col y \
    --target-x-col x_position \
    --target-y-col y_position
```

### Feature-Based Alignment

For alignment based on matching gene/protein expression patterns:

```bash
python -m scripts.run_alignment \
    --reference reference_data.csv \
    --target target_data.csv \
    --output-dir results/alignment \
    --method affine \
    --use-features \
    --ref-feature-cols feature1,feature2 \
    --target-feature-cols marker1,marker2 \
    --feature-weight 0.7
```

## Detailed Parameter Guide

### Common Parameters

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `--reference` | Path to reference data | Required | - |
| `--target` | Path to target data | Required | - |
| `--output-dir` | Directory to save results | Required | - |
| `--method` | Transformation method | "affine" | "rigid", "similarity", "affine", "projective" |
| `--ref-x-col` | X coordinate column in reference | "x" | Any column name |
| `--ref-y-col` | Y coordinate column in reference | "y" | Any column name |
| `--target-x-col` | X coordinate column in target | "x" | Any column name |
| `--target-y-col` | Y coordinate column in target | "y" | Any column name |
| `--visualize` | Generate visualizations | False | - |

### Advanced Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--use-features` | Use features for alignment | False | Enables feature-based alignment |
| `--ref-feature-cols` | Feature columns in reference | None | Comma-separated column names |
| `--target-feature-cols` | Feature columns in target | None | Comma-separated column names |
| `--feature-weight` | Weight of features vs. spatial | 0.5 | 0.0-1.0 (higher = more feature influence) |
| `--max-points` | Maximum points to use | 10000 | Lower for faster processing |
| `--ransac-threshold` | RANSAC inlier threshold | 10.0 | Lower for stricter matching |
| `--ransac-iterations` | RANSAC iterations | 1000 | Higher for better robustness |
| `--ransac-min-samples` | Min. samples for RANSAC | Method-dependent | 2 (rigid), 3 (affine), 4 (projective) |
| `--scale-factor` | Scale factor for coordinates | 1.0 | Adjusts for different coordinate systems |
| `--pre-align` | Use simple pre-alignment | False | Helps with very different starting positions |

### Quality Control Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--validate` | Validate alignment quality | False | Enables quality assessment |
| `--holdout-fraction` | Fraction of points to hold out | 0.1 | 0.05-0.2 recommended |
| `--min-confidence` | Minimum alignment confidence | 0.5 | 0-1 range, higher = stricter |
| `--distance-threshold` | Max allowed point distance | 50.0 | Units same as coordinates |
| `--save-validation-plots` | Save validation plots | False | Requires matplotlib |

## Alignment API

For programmatic usage within Python scripts:

```python
from smint.alignment import run_alignment, transform_coordinates

# Basic usage
alignment_result = run_alignment(
    source_data="path/to/target_data.csv",
    target_data="path/to/reference_data.csv",
    method="affine",
    config={
        "source_x_column": "x_position",
        "source_y_column": "y_position",
        "target_x_column": "x",
        "target_y_column": "y",
        "ransac_threshold": 10.0,
        "ransac_max_iterations": 1000
    }
)

# Access the transformation matrix
transform_matrix = alignment_result["transformation_matrix"]
quality_metrics = alignment_result["quality_metrics"]

# Apply transformation to coordinates
import pandas as pd
data_to_transform = pd.read_csv("path/to/additional_data.csv")
transformed_coords = transform_coordinates(
    coordinates=data_to_transform[["x_position", "y_position"]].values,
    transformation_matrix=transform_matrix
)

# Save transformed coordinates
data_to_transform["x_transformed"] = transformed_coords[:, 0]
data_to_transform["y_transformed"] = transformed_coords[:, 1]
data_to_transform.to_csv("path/to/transformed_data.csv", index=False)
```

## Advanced Feature-Based Alignment

SMINT supports alignment based on matching gene/protein expression patterns:

```python
from smint.alignment import run_alignment
import pandas as pd

# Load data
reference_data = pd.read_csv("reference_data.csv")
target_data = pd.read_csv("target_data.csv")

# Run feature-based alignment
alignment_result = run_alignment(
    source_data=target_data,
    target_data=reference_data,
    method="affine",
    config={
        "source_x_column": "x_position",
        "source_y_column": "y_position",
        "target_x_column": "x",
        "target_y_column": "y",
        "use_features": True,
        "source_feature_columns": ["marker1", "marker2", "marker3"],
        "target_feature_columns": ["feature1", "feature2", "feature3"],
        "feature_weight": 0.7,  # 70% features, 30% spatial
        "normalize_features": True
    }
)
```

## Transformation Methods Explained

SMINT provides several transformation methods with different properties:

| Method | Degrees of Freedom | Preserves | Use Case |
|--------|-------------------|-----------|----------|
| **Rigid** | 3 | Distances, Angles | Same-scale data with rotation/translation |
| **Similarity** | 4 | Angles, Relative distances | Similar data with uniform scaling |
| **Affine** | 6 | Parallel lines | Different imaging modalities, tissue deformation |
| **Projective** | 8 | Straight lines | Significant perspective changes, severe distortion |

## Output Files and Formats

SMINT generates the following output files:

| File | Description | Format |
|------|-------------|--------|
| `transformation_matrix.csv` | Transformation matrix | CSV (3×3 matrix) |
| `transformed_coordinates.csv` | Transformed target coordinates | CSV with original + transformed coordinates |
| `alignment_metrics.json` | Quality metrics | JSON |
| `alignment_report.html` | Interactive visualization | HTML (if `--visualize` is used) |
| `validation_plots/*.png` | Validation visualizations | PNG (if validation enabled) |

## Visualizing Alignment Results

SMINT provides built-in visualization tools for alignment results:

```python
from smint.visualization import visualize_alignment

# Generate interactive visualization
visualize_alignment(
    reference_data="path/to/reference_data.csv",
    target_data="path/to/target_data.csv",
    transformed_data="path/to/transformed_coordinates.csv",
    output_html="alignment_visualization.html",
    reference_name="Spatial Transcriptomics",
    target_name="IF Imaging",
    point_size=5,
    opacity=0.7,
    colormap="viridis"
)
```

## Common Issues and Troubleshooting

### Poor Alignment Quality

- **Problem**: Points not properly aligned
- **Solution**: Try different transformation methods (start with affine), adjust RANSAC parameters, use feature-based alignment if possible

### Flipped or Rotated Alignment

- **Problem**: Alignment appears mirror-flipped or severely rotated
- **Solution**: Use `--pre-align` option, or manually flip one dataset before alignment

### Slow Processing

- **Problem**: Alignment taking too long
- **Solution**: Reduce number of points with `--max-points`, decrease RANSAC iterations, use simpler transformation method

### Feature Mismatch

- **Problem**: Feature-based alignment fails to converge
- **Solution**: Verify matching features between datasets, adjust feature weights, normalize features

## Performance Considerations

Alignment performance depends on several factors:

- **Data size**: Larger datasets (>10,000 points) require more processing time
- **Transformation complexity**: Projective > Affine > Similarity > Rigid (in terms of computation)
- **Feature-based alignment**: Using features increases computation time but may improve accuracy
- **RANSAC parameters**: Higher iterations and lower thresholds increase computation time

## Tips for Best Results

1. **Start Simple**: Begin with simpler transformations (rigid, similarity) before trying affine or projective
2. **Preprocessing**: Remove outliers and normalize coordinates before alignment
3. **Use Features**: When available, feature-based alignment often provides better results
4. **Validation**: Always validate alignment quality with holdout points
5. **Visualization**: Visually inspect alignment results to catch issues metrics might miss
6. **Iterative Approach**: For difficult cases, try iterative alignment with progressively more complex transformations
7. **Common Markers**: For multi-modal data, focus on features/markers present in both datasets
