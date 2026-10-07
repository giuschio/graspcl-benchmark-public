"""
Brief description: Visualize a stored recall entry point cloud with its saved grasp annotations.

Last modified on: 2026-03-12
Commit message: Small changes
"""

import argparse
from pathlib import Path

import open3d as o3d

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler
from grasp_learning_cl.visualization import visualize_scene_pcd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a single recall-module entry (pointcloud + grasps)."
    )
    parser.add_argument(
        "--entry_path",
        type=str,
        default="data2/prova3/recall/c0d5a93fdccc4424b99d3b3039f49efa",
        help="Path to a recall entry folder, e.g. data2/prova2/recall/<entry_id>",
    )
    parser.add_argument(
        "--cloud-file",
        type=str,
        default="pointcloud_downsampled.pcd",
        help="Point cloud filename inside the entry folder",
    )
    parser.add_argument(
        "--grasps-file",
        type=str,
        default="grasps.csv",
        help="Grasp file filename inside the entry folder",
    )
    parser.add_argument(
        "--color-key",
        type=str,
        default="score",
        help="Grasp key used for coloring (e.g. score, success, type)",
    )

    args = parser.parse_args()

    entry_path = Path(args.entry_path)
    cloud_path = entry_path / args.cloud_file
    grasps_path = entry_path / args.grasps_file

    if not entry_path.is_dir():
        raise FileNotFoundError(f"Entry folder not found: {entry_path}")
    if not cloud_path.exists():
        raise FileNotFoundError(f"Pointcloud file not found: {cloud_path}")
    if not grasps_path.exists():
        raise FileNotFoundError(f"Grasps file not found: {grasps_path}")

    cloud = o3d.io.read_point_cloud(str(cloud_path))
    if cloud.is_empty():
        raise ValueError(f"Loaded pointcloud is empty: {cloud_path}")

    grasps = SceneGraspDataHandler.from_file(str(grasps_path)).data
    if len(grasps) == 0:
        print("No grasps in file. Visualizing pointcloud only.")

    print(f"Loaded pointcloud: {cloud_path}")
    print(f"Loaded grasps: {len(grasps)} from {grasps_path}")

    visualize_scene_pcd(cloud, grasps, color_key=args.color_key)


if __name__ == "__main__":
    main()
