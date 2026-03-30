#!/usr/bin/env python
"""Show which query detects which object to help select queries for visualization."""

import argparse
import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
import os
import sys
sys.path.append('./')


def parse_args():
    parser = argparse.ArgumentParser(description='Show query predictions')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', required=True)
    parser.add_argument('--sample-idx', type=int, default=0, help='which sample to analyze')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                       help='only show predictions above this score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Import plugin
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # Build dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # Build model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location=device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Get sample
    for i, data in enumerate(data_loader):
        if i == args.sample_idx:
            break

    print(f"\n{'='*80}")
    print(f"Sample {args.sample_idx} - Query Predictions")
    print(f"{'='*80}\n")

    # Run inference
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    # Parse results
    if isinstance(result, list):
        result = result[0]

    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]

    # Get predictions
    boxes_3d = result['pts_bbox']['boxes_3d']  # 3D boxes
    scores_3d = result['pts_bbox']['scores_3d']  # Confidence scores
    labels_3d = result['pts_bbox']['labels_3d']  # Class labels

    print(f"Total predictions: {len(scores_3d)}")
    print(f"Predictions above {args.score_threshold}: ", end='')

    # Filter by score
    valid_mask = scores_3d > args.score_threshold
    num_valid = valid_mask.sum()
    print(f"{num_valid}\n")

    if num_valid == 0:
        print(f"No predictions above threshold {args.score_threshold}")
        print("Try lowering --score-threshold")
        return

    # Show predictions sorted by score
    print(f"{'Query':<8} {'Score':<8} {'Class':<20} {'Location (x, y, z)':<30}")
    print(f"{'-'*80}")

    # Get indices of valid predictions sorted by score
    valid_indices = torch.where(valid_mask)[0]
    valid_scores = scores_3d[valid_mask]
    sorted_indices = valid_indices[torch.argsort(valid_scores, descending=True)]

    for idx in sorted_indices:
        idx = idx.item()
        score = scores_3d[idx].item()
        label = labels_3d[idx].item()
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"

        # Get 3D box center
        box = boxes_3d[idx]
        if hasattr(box, 'center'):
            center = box.center.cpu().numpy()
        else:
            # For tensor format
            center = box[:3].cpu().numpy() if len(box.shape) == 1 else box[0, :3].cpu().numpy()

        location = f"({center[0]:>6.1f}, {center[1]:>6.1f}, {center[2]:>6.1f})"

        print(f"{idx:<8} {score:<8.3f} {class_name:<20} {location:<30}")

    print(f"\n{'-'*80}")
    print("\nRecommended queries for visualization:")

    # Get top queries
    top_k = min(10, len(sorted_indices))
    top_queries = sorted_indices[:top_k].cpu().numpy()

    print(f"  --query-ids {' '.join(map(str, top_queries))}")

    print(f"\nTo visualize these queries:")
    print(f"python tools/visualize_attention.py \\")
    print(f"    {args.config} \\")
    print(f"    --checkpoint {args.checkpoint} \\")
    print(f"    --samples 1 \\")
    print(f"    --query-ids {' '.join(map(str, top_queries[:5]))} \\")
    print(f"    --layer-ids 5")


if __name__ == '__main__':
    main()
