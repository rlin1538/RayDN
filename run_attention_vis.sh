#!/bin/bash
# Quick start script for attention visualization

echo "========================================="
echo "RayDN Attention Visualization"
echo "========================================="
echo ""

# Configuration
CONFIG="work_dirs/0519_grid0075_r50_1600_for_test/height_map_grid0075_with_lightweight_denoise_1600_for_test.py"
CHECKPOINT="work_dirs/0519_grid0075_r50_1600_for_test/latest.pth"
OUTPUT_DIR="attention_vis"
SAMPLES=20
# Sample every 50th sample to get diverse scenes (instead of consecutive frames)
SAMPLE_STRIDE=10

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please update the CHECKPOINT variable in this script"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found at $CONFIG"
    echo "Please update the CONFIG variable in this script"
    exit 1
fi

echo "Using config: $CONFIG"
echo "Using checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Number of samples: $SAMPLES"
echo "Sample stride: $SAMPLE_STRIDE (sample every ${SAMPLE_STRIDE}th frame for diverse scenes)"
echo ""

# Make sure conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Activating conda environment 'mv'..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mv
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Run visualization with stride for diverse scenes
# Visualize queries 0, 100, 200, 300, 400 to see different detection patterns
echo "Running attention visualization..."
echo ""

python tools/visualize_attention.py \
    "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --samples $SAMPLES \
    --sample-stride $SAMPLE_STRIDE \
    --output-dir "$OUTPUT_DIR" \
    --query-ids 0 50 100 150 200 250 300 350 400 \
    --layer-ids 5 \
    --add-merged

echo ""
echo "========================================="
echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view the results:"
echo "  Image attention: eog $OUTPUT_DIR/*_image_*.png"
echo "  Point cloud previews: eog $OUTPUT_DIR/*_pointcloud_*.png"
echo "  Point cloud PLY files: Download and open with CloudCompare/MeshLab"
echo "========================================="
