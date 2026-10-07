import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from grasp_cl_integration.bootstrap import add_repo_paths

add_repo_paths(REPO_ROOT)

from tqdm import tqdm

from grasp_cl_integration.experiment_runtime import EnvWrapper

"""
We use this script to make sure our real-world comparisons are fair.
It starts the panda, and lets the user take top-down rgb pictures with it of object configurations.
During real-world evals, we use these pictures to align the objects in reproducible configurations (by superimposing the saved image and a live view with alpha 0.5)

"""


def main():
    parser = argparse.ArgumentParser(description="Save top-down reference views for real-scene evaluation.")
    parser.add_argument(
        "--scene_views_path", 
        type=str, 
        default="data_local_2/scenes", 
        help="Root output folder; images are saved under scene_views_path/object_set")
    parser.add_argument(
        "--object_set",
        type=str,
        default="category-nivea-qualitative")
    parser.add_argument(
        "--num_scenes", 
        type=int, 
        default=1)


    args = parser.parse_args()

    print("Executing scripts_integrated/exp_15_adaptation/experiment_15_save_views.py")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    scene_views_path = os.path.join(args.scene_views_path, args.object_set)

    env_wrapper = EnvWrapper(
        experiment_seed=0,
        scene_type="packed",
        object_set=args.object_set,
        max_object_count=5,
        perception_type="singleview",
        placement_type="front",
        scene_views_path=scene_views_path,
    )

    for scene_idx in tqdm(range(args.num_scenes), total=args.num_scenes, desc="Save scene views"):
        image_path = env_wrapper._load_scene_reference_path(scene_idx)

        response = input(f"Arrange scene {scene_idx:06d}, then press Enter to capture it (q to abort): ").strip().lower()
        if response == "q":
            break
        image_path = env_wrapper.save_scene_reference(scene_idx, scene_views_path)
        print(f"Saved {image_path}")


if __name__ == "__main__":
    main()
