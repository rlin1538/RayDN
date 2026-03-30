# ------------------------------------------------------------------------
# Attention Visualization Tool
# ------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import os
from pathlib import Path
import open3d as o3d
from matplotlib import cm


class AttentionVisualizer:
    """Visualize attention weights on images and point clouds."""

    def __init__(self, output_dir='attention_vis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_image_attention(self, images, attention_weights, sampled_points_2d,
                                  query_idx=0, layer_idx=0, save_name='image_attention'):
        """
        Visualize attention weights on multi-view images.

        Args:
            images: List of images [num_cams, H, W, 3]
            attention_weights: Attention weights [B*num_cams, num_query, num_groups, num_levels*num_pts]
            sampled_points_2d: Sampled 2D points [B, num_cams, num_query, num_pts, 2]
            query_idx: Which query to visualize, or 'all' to merge all queries
            layer_idx: Which decoder layer to visualize
            save_name: Base name for saved images
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        if isinstance(sampled_points_2d, torch.Tensor):
            sampled_points_2d = sampled_points_2d.cpu().numpy()

        print(f"\n[DEBUG] visualize_image_attention:")
        print(f"  images shape: {images.shape}")
        print(f"  attention_weights shape: {attention_weights.shape}")
        print(f"  sampled_points_2d shape: {sampled_points_2d.shape}")
        print(f"  query_idx: {query_idx}, layer_idx: {layer_idx}")

        num_cams = len(images)
        bs = sampled_points_2d.shape[0]

        # Reshape attention weights: [B*num_cams, num_query, num_groups, num_levels*num_pts]
        # -> [B, num_cams, num_query, num_groups, num_levels*num_pts]
        attention_weights = attention_weights.reshape(bs, num_cams, -1,
                                                      attention_weights.shape[-2],
                                                      attention_weights.shape[-1])
        print(f"  Reshaped attention_weights: {attention_weights.shape}")

        if query_idx == 'all' or query_idx == 'merged':
            # Merge all queries
            return self._visualize_merged_attention(images, attention_weights, sampled_points_2d,
                                                   layer_idx, save_name)

        # Single query visualization
        # Average over groups and points for visualization
        # [B, num_cams, num_query, num_groups, num_levels*num_pts] -> [B, num_cams, num_query, num_points]
        attn = attention_weights[0, :, query_idx].mean(axis=1)  # [num_cams, num_points]
        points_2d = sampled_points_2d[0, :, query_idx]  # [num_cams, num_pts, 2]

        print(f"  Selected attention shape: {attn.shape}")
        print(f"  Selected points shape: {points_2d.shape}")
        print(f"  Attention stats - min: {attn.min():.4f}, max: {attn.max():.4f}, mean: {attn.mean():.4f}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for cam_idx in range(num_cams):
            overlay = self._render_single_query_attention(
                images[cam_idx], attn[cam_idx], points_2d[cam_idx], cam_idx
            )
            axes[cam_idx].imshow(overlay)
            axes[cam_idx].set_title(f'Camera {cam_idx} - Query {query_idx}\nLayer {layer_idx}',
                                   fontsize=10)
            axes[cam_idx].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}_query{query_idx}_layer{layer_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved image attention visualization to {save_path}")

    def _render_single_query_attention(self, img, cam_attn, cam_points, cam_idx):
        """Render attention for a single query on a single camera."""
        img = img.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]
        print(f"\n  Camera {cam_idx}: Image size {w}x{h}")

        print(f"    Points range - X: [{cam_points[:, 0].min():.1f}, {cam_points[:, 0].max():.1f}], Y: [{cam_points[:, 1].min():.1f}, {cam_points[:, 1].max():.1f}]")
        print(f"    Attention range: [{cam_attn.min():.4f}, {cam_attn.max():.4f}]")

        # Normalize attention weights to [0, 1]
        if cam_attn.max() > cam_attn.min():
            cam_attn_norm = (cam_attn - cam_attn.min()) / (cam_attn.max() - cam_attn.min())
        else:
            cam_attn_norm = np.ones_like(cam_attn) * 0.5

        # Apply power transformation to enhance contrast
        cam_attn_norm = np.power(cam_attn_norm, 0.5)  # Square root for better visualization

        # Create heatmap with higher resolution
        heatmap = np.zeros((h, w), dtype=np.float32)
        valid_points = 0

        # Draw attention on heatmap with larger radius
        for pt_idx, (point, weight) in enumerate(zip(cam_points, cam_attn_norm)):
            x, y = int(point[0]), int(point[1])
            # Check if point is within image bounds
            if 0 <= x < w and 0 <= y < h:
                # Draw a larger circle - use fixed radius but vary intensity
                radius = 40  # Fixed larger radius
                cv2.circle(heatmap, (x, y), radius, float(weight), -1)
                valid_points += 1

        print(f"    Valid points in image: {valid_points}/{len(cam_points)}")

        if valid_points == 0:
            print(f"    WARNING: No valid points for camera {cam_idx}!")

        # Apply Gaussian blur for smoother visualization
        if valid_points > 0:
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            # Apply gamma correction to enhance visibility
            heatmap = np.power(heatmap, 0.7)

        print(f"    Heatmap range after blur: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

        # Convert heatmap to color using jet colormap
        heatmap_colored = cm.jet(heatmap)[:, :, :3] * 255
        heatmap_colored = heatmap_colored.astype(np.uint8)

        # Create overlay - make image more visible (70% image, 30% heatmap)
        if valid_points > 0:
            overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
        else:
            overlay = img.copy()

        # Draw sampled points on top with better visibility
        for pt_idx, (point, weight) in enumerate(zip(cam_points, cam_attn_norm)):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                # Color based on attention weight (blue=low, red=high)
                color = (int(255 * weight), 0, int(255 * (1 - weight)))
                # Draw filled circle
                cv2.circle(overlay, (x, y), 6, color, -1)
                # Draw white outline for better visibility
                cv2.circle(overlay, (x, y), 7, (255, 255, 255), 2)
                # Draw black outline for contrast
                cv2.circle(overlay, (x, y), 8, (0, 0, 0), 1)

        return overlay

    def _visualize_merged_attention(self, images, attention_weights, sampled_points_2d,
                                   layer_idx, save_name):
        """Merge attention from all queries into a single visualization."""
        print(f"\n  Merging attention from all queries...")

        bs = sampled_points_2d.shape[0]
        num_cams = len(images)
        num_queries = attention_weights.shape[2]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for cam_idx in range(num_cams):
            img = images[cam_idx].copy()
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            h, w = img.shape[:2]

            # Accumulate heatmap from all queries
            heatmap_accum = np.zeros((h, w), dtype=np.float32)
            total_valid_points = 0

            # Process each query
            for q_idx in range(num_queries):
                # Get attention and points for this query
                attn = attention_weights[0, cam_idx, q_idx].mean()  # Average attention weight for this query
                points = sampled_points_2d[0, cam_idx, q_idx]  # [num_pts, 2]

                # Only consider queries with significant attention
                if attn > 0.001:  # Threshold to filter out background queries
                    # Draw points with attention weight
                    for point in points:
                        x, y = int(point[0]), int(point[1])
                        if 0 <= x < w and 0 <= y < h:
                            radius = 30
                            cv2.circle(heatmap_accum, (x, y), radius, float(attn), -1)
                            total_valid_points += 1

            print(f"  Camera {cam_idx}: Merged {total_valid_points} points from {num_queries} queries")

            # Apply Gaussian blur
            if total_valid_points > 0:
                heatmap_accum = cv2.GaussianBlur(heatmap_accum, (51, 51), 0)

                # Normalize
                if heatmap_accum.max() > 0:
                    heatmap_accum = heatmap_accum / heatmap_accum.max()
                    heatmap_accum = np.power(heatmap_accum, 0.7)

            # Convert to color
            heatmap_colored = cm.jet(heatmap_accum)[:, :, :3] * 255
            heatmap_colored = heatmap_colored.astype(np.uint8)

            # Overlay
            if total_valid_points > 0:
                overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
            else:
                overlay = img.copy()

            axes[cam_idx].imshow(overlay)
            axes[cam_idx].set_title(f'Camera {cam_idx} - All Queries Merged\nLayer {layer_idx}',
                                   fontsize=10)
            axes[cam_idx].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / f'{save_name}_query_merged_layer{layer_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved merged attention visualization to {save_path}")

    def visualize_point_cloud_attention(self, points, attention_weights, sampled_points_3d,
                                       query_idx=0, layer_idx=0, save_name='pointcloud_attention'):
        """
        Visualize attention weights on point cloud.

        Args:
            points: Point cloud [N, 3] or [N, 4] (with intensity)
            attention_weights: Attention weights for point cloud [B, num_query, num_pts, num_groups]
            sampled_points_3d: Sampled 3D key points [B, num_query, num_pts, 3]
            query_idx: Which query to visualize, or 'merged' to merge all queries
            layer_idx: Which decoder layer to visualize
            save_name: Base name for saved visualization
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        if isinstance(sampled_points_3d, torch.Tensor):
            sampled_points_3d = sampled_points_3d.cpu().numpy()

        if query_idx == 'merged':
            return self._visualize_merged_pointcloud_attention(
                points, attention_weights, sampled_points_3d, layer_idx, save_name
            )

        # Single query visualization
        # Get points for specific query
        attn = attention_weights[0, query_idx].mean(axis=-1)  # [num_pts]
        key_points = sampled_points_3d[0, query_idx]  # [num_pts, 3]

        # Normalize attention weights
        if attn.max() > attn.min():
            attn_norm = (attn - attn.min()) / (attn.max() - attn.min())
        else:
            attn_norm = np.ones_like(attn) * 0.5

        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Color based on distance to nearest key point with attention
        pcd_colors = np.ones((len(points), 3)) * 0.5  # Default gray

        # For each key point, color nearby points based on attention weight
        for key_pt, weight in zip(key_points, attn_norm):
            # Calculate distances from all points to this key point
            dists = np.linalg.norm(points[:, :3] - key_pt, axis=1)
            # Find points within certain radius
            radius = 2.0  # meters
            mask = dists < radius
            # Color these points based on attention weight and distance
            # Closer points get stronger color
            distance_weights = 1.0 - (dists[mask] / radius)
            color_intensity = weight * distance_weights
            # Red for high attention, blue for low
            pcd_colors[mask, 0] = np.maximum(pcd_colors[mask, 0], color_intensity)
            pcd_colors[mask, 2] = np.maximum(pcd_colors[mask, 2], 1 - color_intensity)

        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        # Create key points visualization
        key_pcd = o3d.geometry.PointCloud()
        key_pcd.points = o3d.utility.Vector3dVector(key_points)
        key_colors = cm.jet(attn_norm)[:, :3]
        key_pcd.colors = o3d.utility.Vector3dVector(key_colors)

        # Save visualization
        save_path = self.output_dir / f'{save_name}_query{query_idx}_layer{layer_idx}.ply'
        o3d.io.write_point_cloud(str(save_path), pcd)
        key_save_path = self.output_dir / f'{save_name}_keypoints_query{query_idx}_layer{layer_idx}.ply'
        o3d.io.write_point_cloud(str(key_save_path), key_pcd)

        print(f"Saved point cloud attention visualization to {save_path}")
        print(f"Saved key points to {key_save_path}")

        # Create a matplotlib-based 2D projection instead of Open3D window rendering
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            # Plot main point cloud
            pts = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            # Subsample for faster rendering
            if len(pts) > 10000:
                indices = np.random.choice(len(pts), 10000, replace=False)
                pts = pts[indices]
                colors = colors[indices]

            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1, alpha=0.3)

            # Plot key points
            key_pts = np.asarray(key_pcd.points)
            key_colors = np.asarray(key_pcd.colors)
            ax.scatter(key_pts[:, 0], key_pts[:, 1], key_pts[:, 2],
                      c=key_colors, s=50, marker='o', edgecolors='white', linewidths=1)

            # Set bird's eye view (top-down view)
            ax.view_init(elev=90, azim=-90)  # elev=90 for top-down view

            # Turn off axis display
            ax.set_axis_off()

            # Set title
            ax.set_title(f'Point Cloud Attention (Bird\'s Eye View) - Query {query_idx}, Layer {layer_idx}',
                        pad=20, fontsize=12)

            # Set equal aspect ratio
            max_range = np.array([pts[:, 0].max()-pts[:, 0].min(),
                                 pts[:, 1].max()-pts[:, 1].min(),
                                 pts[:, 2].max()-pts[:, 2].min()]).max() / 2.0
            mid_x = (pts[:, 0].max()+pts[:, 0].min()) * 0.5
            mid_y = (pts[:, 1].max()+pts[:, 1].min()) * 0.5
            mid_z = (pts[:, 2].max()+pts[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            img_path = self.output_dir / f'{save_name}_query{query_idx}_layer{layer_idx}.png'
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved rendered image to {img_path}")
        except Exception as e:
            print(f"Warning: Could not create matplotlib rendering: {e}")
            print(f"Point cloud files saved. You can view them with: open3d.visualization.draw_geometries([o3d.io.read_point_cloud('{save_path}')])")

    def _visualize_merged_pointcloud_attention(self, points, attention_weights, sampled_points_3d,
                                              layer_idx, save_name):
        """Merge attention from all queries for point cloud visualization."""
        print(f"\n  Merging point cloud attention from all queries...")

        num_queries = attention_weights.shape[1]

        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        # Accumulate colors from all queries
        pcd_colors = np.ones((len(points), 3)) * 0.5  # Default gray
        all_key_points = []
        all_key_colors = []

        # Process each query
        for q_idx in range(num_queries):
            attn = attention_weights[0, q_idx].mean(axis=-1)  # [num_pts]
            key_points = sampled_points_3d[0, q_idx]  # [num_pts, 3]

            # Only consider queries with significant attention
            if attn.mean() > 0.001:
                # Normalize attention weights
                if attn.max() > attn.min():
                    attn_norm = (attn - attn.min()) / (attn.max() - attn.min())
                else:
                    attn_norm = np.ones_like(attn) * 0.5

                # Color nearby points based on attention
                for key_pt, weight in zip(key_points, attn_norm):
                    dists = np.linalg.norm(points[:, :3] - key_pt, axis=1)
                    radius = 2.0
                    mask = dists < radius
                    distance_weights = 1.0 - (dists[mask] / radius)
                    color_intensity = weight * distance_weights * 0.5  # Scale down for merged view
                    pcd_colors[mask, 0] = np.maximum(pcd_colors[mask, 0], color_intensity)
                    pcd_colors[mask, 2] = np.maximum(pcd_colors[mask, 2], 0.5 - color_intensity * 0.5)

                # Collect key points
                all_key_points.append(key_points)
                key_colors = cm.jet(attn_norm)[:, :3]
                all_key_colors.append(key_colors)

        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

        # Merge all key points
        if all_key_points:
            all_key_points = np.vstack(all_key_points)
            all_key_colors = np.vstack(all_key_colors)

            key_pcd = o3d.geometry.PointCloud()
            key_pcd.points = o3d.utility.Vector3dVector(all_key_points)
            key_pcd.colors = o3d.utility.Vector3dVector(all_key_colors)

            # Save PLY files
            save_path = self.output_dir / f'{save_name}_query_merged_layer{layer_idx}.ply'
            o3d.io.write_point_cloud(str(save_path), pcd)
            key_save_path = self.output_dir / f'{save_name}_keypoints_merged_layer{layer_idx}.ply'
            o3d.io.write_point_cloud(str(key_save_path), key_pcd)

            print(f"Saved merged point cloud attention to {save_path}")
            print(f"Saved merged key points ({len(all_key_points)} points) to {key_save_path}")

            # Create matplotlib rendering
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')

                # Plot main point cloud
                pts = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                # Subsample for faster rendering
                if len(pts) > 10000:
                    indices = np.random.choice(len(pts), 10000, replace=False)
                    pts = pts[indices]
                    colors = colors[indices]

                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1, alpha=0.3)

                # Plot key points (subsample if too many)
                key_pts = all_key_points
                key_colors_plot = all_key_colors
                if len(key_pts) > 500:
                    indices = np.random.choice(len(key_pts), 500, replace=False)
                    key_pts = key_pts[indices]
                    key_colors_plot = key_colors_plot[indices]

                ax.scatter(key_pts[:, 0], key_pts[:, 1], key_pts[:, 2],
                          c=key_colors_plot, s=30, marker='o', edgecolors='white', linewidths=0.5, alpha=0.8)

                # Set bird's eye view (top-down view)
                ax.view_init(elev=90, azim=-90)

                # Turn off axis display
                ax.set_axis_off()

                # Set title
                ax.set_title(f'Point Cloud Attention (Bird\'s Eye View) - All Queries Merged, Layer {layer_idx}',
                            pad=20, fontsize=12)

                # Set equal aspect ratio
                max_range = np.array([pts[:, 0].max()-pts[:, 0].min(),
                                     pts[:, 1].max()-pts[:, 1].min(),
                                     pts[:, 2].max()-pts[:, 2].min()]).max() / 2.0
                mid_x = (pts[:, 0].max()+pts[:, 0].min()) * 0.5
                mid_y = (pts[:, 1].max()+pts[:, 1].min()) * 0.5
                mid_z = (pts[:, 2].max()+pts[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                img_path = self.output_dir / f'{save_name}_query_merged_layer{layer_idx}.png'
                plt.savefig(img_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved merged rendered image to {img_path}")
            except Exception as e:
                print(f"Warning: Could not create matplotlib rendering: {e}")

    def save_attention_data(self, attention_data, save_name='attention_data'):
        """Save raw attention data for later analysis."""
        save_path = self.output_dir / f'{save_name}.npz'
        np.savez(save_path, **attention_data)
        print(f"Saved attention data to {save_path}")
