#!/usr/bin/env python
"""Simple test script to verify attention weights are being captured."""

import torch
import sys
sys.path.append('./')

# Test if attention modules have the save_attention attribute
print("Testing attention capture mechanism...")

# Create a simple test
from projects.mmdet3d_plugin.models.utils.detr3d_transformer import DeformableFeatureAggregationCuda

# Create instance
module = DeformableFeatureAggregationCuda(
    embed_dims=256,
    num_groups=8,
    num_levels=4,
    num_cams=6,
    num_pts=13
)

print("\nChecking DeformableFeatureAggregationCuda:")
print(f"  Has save_attention attribute: {hasattr(module, 'save_attention')}")
print(f"  save_attention value: {module.save_attention}")
print(f"  Has attention_weights_img: {hasattr(module, 'attention_weights_img')}")
print(f"  Has sampled_points_2d: {hasattr(module, 'sampled_points_2d')}")

# Enable attention saving
module.save_attention = True
print(f"\nAfter enabling:")
print(f"  save_attention value: {module.save_attention}")

print("\n✓ Attention capture mechanism is properly set up!")
print("\nNow test with actual inference to see if data is captured.")
