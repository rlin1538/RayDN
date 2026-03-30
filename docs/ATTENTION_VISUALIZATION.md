# Attention Visualization for RayDN

This guide explains how to visualize the attention weights from the DETR3D Transformer in RayDN.

## Overview

The attention visualization system captures and visualizes:
- **Image Cross-Attention**: Shows which image regions the model focuses on for each query
- **Point Cloud Cross-Attention**: Shows which point cloud regions the model focuses on for each query
- **Multi-View Visualization**: Displays attention across all 6 camera views
- **3D Visualization**: Shows attention weights mapped onto the point cloud

## Requirements

Make sure you have these additional dependencies installed:

```bash
conda activate mv  # Your environment
pip install open3d matplotlib opencv-python
```

**Note**: Point cloud rendering uses matplotlib for PNG output and saves PLY files for detailed viewing. Open3D window rendering is disabled to support headless server environments.

## Usage

### Basic Usage

To visualize attention for a few samples:

```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 5 \
    --output-dir attention_vis
```

### Advanced Options

```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 10 \
    --output-dir attention_vis \
    --query-ids 0 50 100 150 200 \
    --layer-ids 0 2 5 \
    --save-data
```

### Command Line Arguments

- `config`: Path to the config file (required)
- `--checkpoint`: Path to the model checkpoint (required)
- `--samples`: Number of samples to visualize (default: 10)
- `--output-dir`: Directory to save visualizations (default: 'attention_vis')
- `--query-ids`: Which queries to visualize (default: [0, 50, 100])
- `--layer-ids`: Which decoder layers to visualize (default: [5], range: 0-5)
- `--save-data`: Save raw attention data as .npz files for later analysis

## Output Files

The script generates several types of visualizations:

### Image Attention Visualizations

```
attention_vis/
├── sample0_image_query0_layer5.png       # Multi-view image attention heatmaps
├── sample0_image_query50_layer5.png
└── ...
```

These show:
- 6 camera views in a 2×3 grid
- Attention heatmaps overlaid on images
- Sampled key points colored by attention weight
- Red = high attention, Blue = low attention

### Point Cloud Attention Visualizations

```
attention_vis/
├── sample0_pointcloud_query0_layer5.ply        # Full point cloud with attention colors
├── sample0_pointcloud_keypoints_query0_layer5.ply  # Just the sampled key points
├── sample0_pointcloud_query0_layer5.png        # Rendered view
└── ...
```

Point cloud files (.ply) can be opened with:
- **CloudCompare**: Free point cloud viewer (Recommended for headless servers)
  - Download PLY files and open locally
- **MeshLab**: Free 3D mesh processing software
- **Open3D** (if you have a display):
  ```python
  import open3d as o3d
  pcd = o3d.io.read_point_cloud('file.ply')
  o3d.visualization.draw_geometries([pcd])
  ```
- **PNG previews**: Matplotlib-generated 2D projections are also saved

### Raw Attention Data (if --save-data is used)

```
attention_vis/
├── sample0_layer5_attention_data.npz
└── ...
```

These contain raw numpy arrays that can be loaded for custom analysis:

```python
import numpy as np
data = np.load('attention_vis/sample0_layer5_attention_data.npz')
print(data.files)  # List all arrays in the file
weights = data['module.pts_bbox_head.transformer.decoder.layers.5.attentions.1_weights_img']
```

## Understanding the Visualizations

### Image Attention

Each multi-view image shows:
1. **Heatmap**: Blurred regions showing overall attention distribution
2. **Key Points**: Small circles showing exact sampled locations
3. **Color Coding**:
   - Red regions/points = High attention
   - Blue regions/points = Low attention

### Point Cloud Attention

The colored point cloud shows:
1. **Background Points**: Colored based on distance to key points
   - Red = Near a high-attention key point
   - Blue = Near a low-attention key point
   - Gray = Far from any key points
2. **Key Points File**: Shows only the sampled key points colored by attention weight

## Example Workflow

1. **Quick Visualization** - Visualize a few samples to get an overview:
```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 3
```

2. **Detailed Analysis** - Visualize multiple queries and layers:
```bash
python tools/visualize_attention.py \
    work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py \
    --checkpoint work_dirs/0519_grid0075_r50_1600_for_test/latest.pth \
    --samples 10 \
    --query-ids 0 10 20 30 40 50 \
    --layer-ids 0 2 4 5 \
    --save-data
```

3. **View Results**:
```bash
# View generated images
eog attention_vis/sample0_image_query0_layer5.png

# View point cloud PNG preview (matplotlib rendering)
eog attention_vis/sample0_pointcloud_query0_layer5.png

# For detailed 3D view, download PLY files and open with CloudCompare or MeshLab
# Or if you have a display, use Open3D:
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('attention_vis/sample0_pointcloud_query0_layer5.ply')])"
```

## Technical Details

### Modified Files

The following files have been modified to support attention visualization:

1. **projects/mmdet3d_plugin/models/utils/detr3d_transformer.py**
   - Added `save_attention` flag to `DeformableFeatureAggregationCuda`
   - Added `save_attention` flag to `MixedCrossAttention`
   - Added storage for attention weights and sampled points
   - Modified forward methods to save attention data when enabled

2. **tools/attention_visualizer.py** (New)
   - AttentionVisualizer class for creating visualizations
   - Methods for image and point cloud visualization

3. **tools/visualize_attention.py** (New)
   - Main inference script with attention visualization

### Attention Weights Shape

- **Image Attention**: `[B*num_cams, num_query, num_groups, num_levels*num_pts]`
  - B: Batch size (usually 1)
  - num_cams: 6 camera views
  - num_query: 300 queries
  - num_groups: 8 attention groups
  - num_levels: 4 feature pyramid levels
  - num_pts: 13 sampled key points per query

- **Point Cloud Attention**: `[B, num_query, num_pts, num_groups]`
  - Similar structure but for BEV point cloud features

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
- Reduce `--samples`
- Visualize fewer queries with `--query-ids`
- Visualize only specific layers with `--layer-ids`

### No Attention Data Collected

If you see "Warning: No attention data collected":
- Make sure the checkpoint is loaded correctly
- Check that the model architecture matches the config file
- Verify that the layer indices are valid (0-5 for 6-layer decoder)

### Visualization Errors

If visualization fails:
- Check that images and point clouds are being loaded correctly
- Ensure Open3D is installed for point cloud visualization
- Try with `--save-data` to check if raw data is being captured

## Citation

If you use this visualization tool in your research, please cite the RayDN paper.
