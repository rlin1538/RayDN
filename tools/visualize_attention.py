# ------------------------------------------------------------------------
# Modified from benchmark.py for attention visualization
# ------------------------------------------------------------------------
import argparse
import time
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

from tools.attention_visualizer import AttentionVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file', required=True)
    parser.add_argument('--samples', default=10, type=int, help='number of samples to visualize (used with --sample-stride)')
    parser.add_argument('--sample-indices', nargs='+', type=int, default=None,
                       help='specific sample indices to visualize (e.g., 0 100 200 300). Overrides --samples and --sample-stride')
    parser.add_argument('--sample-stride', default=1, type=int,
                       help='stride for sampling (e.g., 50 means sample every 50th sample for diverse scenes)')
    parser.add_argument('--output-dir', default='attention_vis', help='output directory for visualizations')
    parser.add_argument('--query-ids', nargs='+', type=int, default=[0, 50, 100],
                       help='query indices to visualize')
    parser.add_argument('--add-merged', action='store_true',
                       help='also create a merged visualization of all queries')
    parser.add_argument('--layer-ids', nargs='+', type=int, default=[5],
                       help='decoder layer indices to visualize (0-5 for 6 layers)')
    parser.add_argument('--save-data', action='store_true',
                       help='save raw attention data as npz files')
    args = parser.parse_args()
    return args


def enable_attention_saving(model):
    """Enable attention saving for all attention modules in the model."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'save_attention'):
            module.save_attention = True
            count += 1
            print(f"Enabled attention saving for: {name}")
    print(f"Total attention modules enabled: {count}")


def collect_attention_weights(model, layer_idx=None):
    """Collect attention weights from all attention modules."""
    attention_data = {}

    for name, module in model.named_modules():
        if hasattr(module, 'save_attention') and module.save_attention:
            # Check if this is the target layer
            if layer_idx is not None and f'layers.{layer_idx}' not in name:
                continue

            module_data = {}
            if hasattr(module, 'attention_weights_img') and module.attention_weights_img is not None:
                module_data['weights_img'] = module.attention_weights_img
            if hasattr(module, 'attention_weights_pts') and module.attention_weights_pts is not None:
                module_data['weights_pts'] = module.attention_weights_pts
            if hasattr(module, 'sampled_points_2d') and module.sampled_points_2d is not None:
                module_data['points_2d'] = module.sampled_points_2d
            if hasattr(module, 'sampled_points_3d') and module.sampled_points_3d is not None:
                module_data['points_3d'] = module.sampled_points_3d

            if module_data:
                attention_data[name] = module_data

    return attention_data


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

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
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location=device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # Enable attention saving
    enable_attention_saving(model)

    # Initialize visualizer
    visualizer = AttentionVisualizer(output_dir=args.output_dir)

    # Determine which samples to visualize
    if args.sample_indices is not None:
        # Use specific indices provided by user
        sample_indices = args.sample_indices
        print(f"\nWill visualize specific samples: {sample_indices}")
    else:
        # Use stride-based sampling
        sample_indices = list(range(0, args.samples * args.sample_stride, args.sample_stride))
        print(f"\nWill visualize {args.samples} samples with stride {args.sample_stride}")
        print(f"Sample indices: {sample_indices}")

    print(f"\nStarting attention visualization...")
    print(f"Total samples to process: {len(sample_indices)}")
    print(f"Query IDs: {args.query_ids}")
    print(f"Layer IDs: {args.layer_ids}")

    # Process samples
    processed_count = 0
    for data_idx, data in enumerate(data_loader):
        if data_idx not in sample_indices:
            continue

        # Find the position of this sample in our list
        sample_idx = sample_indices.index(data_idx)

        print(f"\n{'='*60}")
        print(f"Processing sample {processed_count + 1}/{len(sample_indices)} (dataset index: {data_idx})")
        print(f"{'='*60}")

        with torch.no_grad():
            # Run inference
            result = model(return_loss=False, rescale=True, **data)

            # Get images and points from data
            img_data = data['img'][0].data[0]
            print(f"Image data type: {type(img_data)}")
            print(f"Image data shape: {img_data.shape if hasattr(img_data, 'shape') else 'N/A'}")

            images = img_data.cpu().numpy()

            # Handle different possible image shapes
            if len(images.shape) == 4:  # [num_cams, 3, H, W]
                images = images.transpose(0, 2, 3, 1)  # [num_cams, H, W, 3]
            elif len(images.shape) == 5:  # [B, num_cams, 3, H, W]
                images = images[0].transpose(0, 2, 3, 1)  # [num_cams, H, W, 3]
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")

            print(f"Processed images shape: {images.shape}")

            # Denormalize images
            if hasattr(cfg, 'img_norm_cfg'):
                mean = np.array(cfg.img_norm_cfg['mean']).reshape(1, 1, 1, 3)
                std = np.array(cfg.img_norm_cfg['std']).reshape(1, 1, 1, 3)
            else:
                # Default normalization values
                mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, 1, 3)
                std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, 1, 3)

            images = images * std + mean
            images = np.clip(images, 0, 255).astype(np.uint8)

            # Get point cloud
            points = data['points'][0].data[0][0].cpu().numpy()  # [N, 5]
            print(f"Points shape: {points.shape}")

            # Visualize for each specified layer
            for layer_idx in args.layer_ids:
                print(f"\nVisualizing layer {layer_idx}...")

                # Collect attention weights for this layer
                attention_data = collect_attention_weights(model, layer_idx=layer_idx)

                if not attention_data:
                    print(f"Warning: No attention data collected for layer {layer_idx}")
                    continue

                # Find the cross-attention module for this layer
                cross_attn_data = None
                print(f"Available attention modules:")
                for name in attention_data.keys():
                    print(f"  - {name}")

                for name, data_dict in attention_data.items():
                    if f'layers.{layer_idx}' in name and 'weights_img' in data_dict:
                        cross_attn_data = data_dict
                        print(f"Using attention data from: {name}")
                        # Print shapes for debugging
                        for key, value in data_dict.items():
                            if hasattr(value, 'shape'):
                                print(f"  {key} shape: {value.shape}")
                        break

                if cross_attn_data is None:
                    print(f"Warning: No cross-attention data found for layer {layer_idx}")
                    continue

                # First, create merged visualization if requested
                if args.add_merged and 'weights_img' in cross_attn_data and 'points_2d' in cross_attn_data:
                    print(f"\n  Creating merged visualization...")
                    try:
                        visualizer.visualize_image_attention(
                            images=images,
                            attention_weights=cross_attn_data['weights_img'],
                            sampled_points_2d=cross_attn_data['points_2d'],
                            query_idx='merged',
                            layer_idx=layer_idx,
                            save_name=f'sample{sample_idx}_image'
                        )
                    except Exception as e:
                        print(f"    Error creating merged visualization: {e}")
                        import traceback
                        traceback.print_exc()

                # Visualize for each specified query
                for query_idx in args.query_ids:
                    # Check if query_idx is valid
                    num_queries = cross_attn_data['weights_img'].shape[1] if 'weights_img' in cross_attn_data else 0
                    if query_idx >= num_queries:
                        print(f"  Skipping query {query_idx} (only {num_queries} queries available)")
                        continue

                    print(f"  Query {query_idx}...")

                    # Visualize image attention
                    if 'weights_img' in cross_attn_data and 'points_2d' in cross_attn_data:
                        try:
                            visualizer.visualize_image_attention(
                                images=images,
                                attention_weights=cross_attn_data['weights_img'],
                                sampled_points_2d=cross_attn_data['points_2d'],
                                query_idx=query_idx,
                                layer_idx=layer_idx,
                                save_name=f'sample{sample_idx}_image'
                            )
                        except Exception as e:
                            print(f"    Error visualizing image attention: {e}")
                            import traceback
                            traceback.print_exc()

                    # Visualize point cloud attention
                    if 'weights_pts' in cross_attn_data and 'points_3d' in cross_attn_data:
                        try:
                            visualizer.visualize_point_cloud_attention(
                                points=points,
                                attention_weights=cross_attn_data['weights_pts'],
                                sampled_points_3d=cross_attn_data['points_3d'],
                                query_idx=query_idx,
                                layer_idx=layer_idx,
                                save_name=f'sample{sample_idx}_pointcloud'
                            )
                        except Exception as e:
                            print(f"    Error visualizing point cloud attention: {e}")

                # Create merged point cloud visualization if requested
                if args.add_merged and 'weights_pts' in cross_attn_data and 'points_3d' in cross_attn_data:
                    print(f"\n  Creating merged point cloud visualization...")
                    try:
                        visualizer.visualize_point_cloud_attention(
                            points=points,
                            attention_weights=cross_attn_data['weights_pts'],
                            sampled_points_3d=cross_attn_data['points_3d'],
                            query_idx='merged',
                            layer_idx=layer_idx,
                            save_name=f'sample{sample_idx}_pointcloud'
                        )
                    except Exception as e:
                        print(f"    Error creating merged point cloud visualization: {e}")
                        import traceback
                        traceback.print_exc()

                # Save raw attention data if requested
                if args.save_data:
                    save_data = {}
                    for name, data_dict in attention_data.items():
                        for key, value in data_dict.items():
                            if isinstance(value, torch.Tensor):
                                value = value.cpu().numpy()
                            save_data[f"{name}_{key}"] = value

                    visualizer.save_attention_data(
                        save_data,
                        save_name=f'sample{sample_idx}_layer{layer_idx}_attention_data'
                    )

        processed_count += 1

        # Check if we've processed all requested samples
        if processed_count >= len(sample_indices):
            break

    print(f"\n{'='*60}")
    print(f"Attention visualization completed!")
    print(f"Processed {processed_count} samples")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
