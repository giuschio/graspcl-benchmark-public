"""
Brief description: Visualize heuristic grasp sampler outputs on generated scenes.

Last modified on: 2026-04-20
Commit message: Some reorganization
"""

import argparse
import os

import open3d as o3d

from grasp_learning_cl.visualization import visualize_scene_pcd
from grasp_learning_cl.grasp_heuristics import get_sampler
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker

from pathlib import Path


SAMPLER_KEYS = [
    "antipodal",
    "antipodal_simple",
    "edge",
    "lowquality",
    "side",
    "topdown",
    "vgn",
]


def main():
    parser = argparse.ArgumentParser(description="Visualize grasps sampled by heuristic samplers.")
    parser.add_argument(
        "--scene_dir", 
        type=Path,
        default="data_local/2026_03_17_simulation_training_data/scene_0", 
        help="Scene folder containing the point cloud file")
    parser.add_argument("--num_grasps", type=int, default=64, help="Number of grasps to sample per sampler")
    args = parser.parse_args()

    cloud_no_table = o3d.io.read_point_cloud(args.scene_dir / "cloud_no_table.ply")
    cloud_w_table = o3d.io.read_point_cloud(args.scene_dir / "cloud_w_table.ply")

    collision_checker = CollisionChecker()

    for sampler_key in SAMPLER_KEYS:
        sampler = get_sampler(sampler_key)
        grasps = sampler(cloud_no_table, args.num_grasps)
        grasps = collision_checker(cloud_w_table, grasps)
        print(f"{sampler_key}: visualizing {len(grasps)} grasps")
        visualize_scene_pcd(cloud_w_table, grasps, window_name=sampler_key)


if __name__ == "__main__":
    main()
