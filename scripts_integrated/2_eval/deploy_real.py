"""Continuously deploy and optionally adapt a CL policy on the real Panda.

Unlike reproducible evaluation, this script does not use saved scenes, fixed
seeds, experiment directories, policy nicknames, scene limits, or log files.
It runs until the operator aborts a reset or presses Ctrl-C.

The policy is loaded in place from cl_module_path. When outcome learning or
demonstrations change it, it is saved back to that same directory immediately.
update_scorer controls learning from executed-grasp outcomes.
query_demonstrations controls whether the operator may be asked for a
demonstration; accepted demonstrations update recall and scoring through the
CL module's standard demonstration path.
"""

import argparse
import sys
from pathlib import Path

from grasp_cl_integration.common import parse_bool
from simple_panda_grasp.utils.perception_utils import cloud_from_image_data
from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker
from grasp_learning_cl.grasp_classification.table_filter import TableFilter
from omegaconf import OmegaConf
from simple_panda_grasp.collision import ConservativeVoxelMap
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.offline_visualization import visualize_scene_pcd


class RealPandaCollisionChecker(CollisionChecker):
    DEFAULT_COLLISION = [
        ((-0.04, -0.10, -0.12), (0.04, 0.10, -0.04)),
        ((-0.045, -0.045, -0.30), (0.045, 0.045, -0.12)),
        ((0.02, -0.045, -0.08), (0.09, 0.045, -0.04)),
    ]


def ask_yes_no(prompt: str) -> bool:
    response = ""
    while response not in {"y", "n"}:
        response = input(prompt).strip().lower()
    return response == "y"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously deploy and optionally adapt a CL policy on the real robot."
    )
    parser.add_argument(
        "--cl_module_path",
        type=Path,
        default="submodules/grasp_learning_cl/data/2026_05_26_adapted_model",
        help="Existing CL module to load and update in place",
    )
    parser.add_argument("--perception_type", type=str, default="singleview")
    parser.add_argument("--show-grasp-proposals", default=False)

    parser.add_argument("--grasp_score_threshold", type=float, default=0.0)
    parser.add_argument("--grasp_object_type", type=str, default="unknown")
    parser.add_argument("--grasp_type", type=str, default=None)
    parser.add_argument("--score_key", type=str, default="score")
    parser.add_argument("--estimation_mode", type=str, default="posterior")
    parser.add_argument("--use_recall", type=parse_bool, default=False)
    parser.add_argument("--use_sampler", type=parse_bool, default=True)
    parser.add_argument("--use_scoring", type=parse_bool, default=True)

    parser.add_argument(
        "--update_scorer",
        type=parse_bool,
        default=False,
        help="Learn from the outcome of every executed grasp",
    )
    parser.add_argument(
        "--query_demonstrations",
        type=parse_bool,
        default=False,
        help="Allow the deployment loop to ask the operator for demonstrations",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy_path = Path(args.cl_module_path)
    if not policy_path.is_dir():
        raise FileNotFoundError(f"Missing cl_module_path: {policy_path}")

    print("Starting continuous real-robot deployment:")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    print("\tseed: None")
    print("Press Ctrl-C at any time to stop.")

    cl_module = ContinualLearningModule.load_inplace(
        modules_root=policy_path,
        device="cuda",
    )
    panda_interface = PandaGraspInterface("submodules/simple_panda_grasp/configs/config_panda.yaml")

    table_filter = TableFilter()
    real_panda_collision_checker = RealPandaCollisionChecker(max_points=1_000_000)
    conservative_mapper = ConservativeVoxelMap(
        panda_interface.lower,
        panda_interface.upper,
        resolution=0.005,
    )

    while True:
        # reset_result = panda_interface.reset(seed=None)
        reset_result = True
        if reset_result is False:
            print("Deployment stopped by operator.")
            break

        image_data = panda_interface.get_observation_default_viewpoint()
        pointcloud_w_table = cloud_from_image_data(
            image_data,
            lower_bounds=panda_interface.lower,
            upper_bounds=panda_interface.upper,
        )

        pointcloud_no_table = table_filter(pointcloud_w_table, debug=False)
        if len(pointcloud_no_table.points) < 5:
            print("No visible objects. Resetting for the next deployment scene.")
            break

        grasp_proposals = cl_module.predict(
            pointcloud_w_table,
            object_type=args.grasp_object_type,
            grasp_score_threshold=args.grasp_score_threshold,
            score_key=args.score_key,
            estimation_mode=args.estimation_mode,
            use_recall=args.use_recall,
            use_sampler=args.use_sampler,
            use_scoring=args.use_scoring,
            debug=False,
        )

        unsafe_pcd = conservative_mapper(pointcloud_w_table,image_data["eye"][0])
        # makes it faster: given a set of ranked grasps, execute the first one that passes
        # reachability and collision tests
        for g in grasp_proposals:
            filtered_grasps = panda_interface.run_grasp_checks([g], refine_pose=True)
            filtered_grasps = real_panda_collision_checker.filter_grasps(
                unsafe_pcd,
                filtered_grasps,
                debug=False,
                pose_keys=("pose", "pose_pre"),
            )
            if len(filtered_grasps) > 0:
                break
        if args.show_grasp_proposals:
            visualize_scene_pcd(pointcloud_w_table, grasp_proposals[0:min(20, len(grasp_proposals))])

        executed_grasp = filtered_grasps[0]
        success, msg = panda_interface.pick(executed_grasp)
        print(msg)
        if not success and msg.startswith("Failed to move"):
            continue  
        if success:
            panda_interface.place_side()
            executed_grasp["success"] = success
        else:
            panda_interface.place_reset()
            executed_grasp = None
        

        if args.update_scorer and executed_grasp is not None:
            cl_module.add_datapoint(
                pointcloud_w_table=pointcloud_w_table,
                grasps=[executed_grasp],
                object_type=args.grasp_object_type,
            )

        if args.query_demonstrations and ask_yes_no(
            "Provide a demonstration (y/n)? "
        ):
            demonstration_grasps = cl_module.get_demonstrations(
                pointcloud_w_table=pointcloud_w_table,
                grasp_proposals=grasp_proposals,
                max_demonstrations=1,
            )
            if demonstration_grasps:
                cl_module.add_demonstration(
                    pointcloud_w_table=pointcloud_w_table,
                    demo_grasps=demonstration_grasps,
                    object_type=args.grasp_object_type,
                    grasp_type=args.grasp_type,
                    grasp_proposals=grasp_proposals,
                    global_annotation=False,
                    local_annotation=True,
                )
    panda_interface.move_to_start()



if __name__ == "__main__":
    main()
