"""
Brief description: Generate simulated training scenes and save sampled grasp datasets.

Last modified on: 2026-04-20
Commit message: Some reorganization
"""

import argparse
import os
from random import sample, randint

import open3d as o3d
from tqdm import tqdm

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler
from grasp_env.environments.env import GraspEnv
from grasp_env.environments.perception_utils import cloud_from_image_data

from grasp_learning_cl.grasp_heuristics import get_sampler

from grasp_learning_cl.grasp_classification.table_filter import TableFilter


def main():
    parser = argparse.ArgumentParser(
        description="Collect grasp data by executing planner grasps in simulation."
    )
    parser.add_argument("--object_set", type=str, default="vgn-pile-train", help="Object set to use")
    parser.add_argument("--scene_type", type=str, default="packed", help="Scene type (e.g., packed, sparse)")
    parser.add_argument("--max_object_count", type=int, default=5, help="Max number of objects in the scene")
    parser.add_argument("--num_scenes", type=int, default=5000, help="Number of scenes to evaluate")
    parser.add_argument("--max_grasps_per_scene", type=int, default=512, help="Maximum number of candidate grasps to execute per scene")
    parser.add_argument("--num_trials_per_grasp", type=int, default=1, help="How many simulator executions to average for each grasp")
    parser.add_argument("--success_threshold", type=float, default=0.85, help="Number of scenes to evaluate")
    parser.add_argument("--grasp_sampler_type", type=str, default="ensemblehq", help="Type of grasp sampler to use")
    parser.add_argument("--perception_type", type=str, default="singleview", help="Type of grasp sampler to use")
    parser.add_argument("--output_dir", type=str, default="data/training_data", help="Path to output dir")
    parser.add_argument("--experiment_seed", type=int, default=0, help="Path to output dir")
    
    args = parser.parse_args()

    # note: can simulate roughly 30 trials per second (so 3 grasps if we try 10 times)
    # at 256 grasps per scene, this means around 1.5 minutes per scene

    os.makedirs(args.output_dir, exist_ok=True)
    env = GraspEnv(
        seed=args.experiment_seed,
        scene_type=args.scene_type,
        object_set=args.object_set,
        show_gui=False,
        output_video=None,
    )

    grasp_sampler = get_sampler(args.grasp_sampler_type)
    table_filter = TableFilter()

    pbar = tqdm(range(args.num_scenes), total=args.num_scenes, desc="Collecting data")
    for scene_number in pbar:
        scene_folder = os.path.join(args.output_dir, f"scene_{scene_number}")
        os.makedirs(scene_folder, exist_ok=True)
        env.reset(object_count=randint(1, args.max_object_count))

        obs_func = env.get_observation_images_fixed_viewpoints if args.perception_type == "multiview" else env.get_observation_random_viewpoint
        image_data = obs_func()
        pointcloud_w_table = cloud_from_image_data(
            image_data,
            lower_bounds=env.sim.lower - [0, 0, 0.04],
            upper_bounds=env.sim.upper,
            noise_level=0.0008  # we apply noise here, since it influences grasp sampling
        )

        pointcloud_no_table = table_filter(pointcloud_w_table, debug=False)

        if pointcloud_no_table is None:
            continue

        # sample a large amount of grasps
        grasps = grasp_sampler(
            cloud_objects=pointcloud_no_table, 
            cloud_collisions=pointcloud_w_table,
            n_grasps=args.max_grasps_per_scene)
        if len(grasps) > args.max_grasps_per_scene:
            grasps = sample(grasps, k=args.max_grasps_per_scene)

        env.sim.save_state("scene_before_grasp_eval")
        grasp_noise = {'t_x': 0.008, 't_y': 0.008, 't_z': 0.008, 'r_x': 4.0, 'r_y': 4.0, 'r_z': 4.0}
        grasp_noise = None
        for grasp in grasps:
            res = env.step_robust(
                grasp["pose"],
                width_pre=grasp["width"],
                noise=grasp_noise,
                remove_grasped_object=False,
                num_trials=args.num_trials_per_grasp,
            )
            grasp["success_percentage"] = float(res["success"])
            grasp["success_binary"] = float(res["success"] > args.success_threshold)
            grasp["width"] = float(res["width"])

            env.sim.restore_state("scene_before_grasp_eval")
            env.sim.save_state("scene_before_grasp_eval")

        env.sim.restore_state("scene_before_grasp_eval")

        o3d.io.write_point_cloud(os.path.join(scene_folder, "cloud_w_table.ply"), pointcloud_w_table)
        o3d.io.write_point_cloud(os.path.join(scene_folder, "cloud_no_table.ply"), pointcloud_no_table)
        SceneGraspDataHandler.data_to_file(grasps, os.path.join(scene_folder, "grasps.csv"))


if __name__ == "__main__":
    main()
