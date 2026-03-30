#!/usr/bin/env python
"""Simple script to view point cloud PLY files with Open3D."""

import argparse
import open3d as o3d
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='View point cloud PLY files')
    parser.add_argument('ply_files', nargs='+', help='Path to PLY file(s) to view')
    parser.add_argument('--combined', action='store_true',
                       help='View main point cloud and keypoints together')
    args = parser.parse_args()

    geometries = []

    for ply_file in args.ply_files:
        ply_path = Path(ply_file)
        if not ply_path.exists():
            print(f"Error: File not found: {ply_file}")
            continue

        print(f"Loading: {ply_file}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"  Points: {len(pcd.points)}")
        geometries.append(pcd)

    if not geometries:
        print("No valid point cloud files to display")
        return

    print(f"\nDisplaying {len(geometries)} point cloud(s)")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Ctrl+Mouse: Pan")
    print("  - H: Print help")
    print("  - Q/ESC: Quit")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Visualization",
        width=1280,
        height=720,
        left=50,
        top=50,
        point_show_normal=False
    )


if __name__ == '__main__':
    main()
