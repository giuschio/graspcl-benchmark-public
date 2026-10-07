"""
TLDL: Interactively adapt a cl_module to objects belonging to train_object_set via demonstrations and autonomous learning

Arguments:
train_object_set:           what category of objects are we performing interactive training on?
scene_type:                 for sim (packed, singulated, or pile) (we use packed)
max_object_count:
num_training_scenes:

cl_module_source_path:      if no existing cl_module is found at <experiment_path>/<train_object_set>/seed_<experiment_seed>/<policy_name>, we load
                            the initial cl_module from here
perception_type:            singleview (only one random view of the scene)
grasp_score_threshold:      0.0 (we always try out the best current grasp)
grasp_object_type:          "unknown" optionally, we can pass the object category. Currently not used
grasp_type:                 Currently not used

experiment_path:            used for saving results of the training
experiment_seed:
policy_name:                what should we call the policy we are training?

score_key:                  default "score" (leave as is)
estimation_mode:            parameter for the scoring module (default: "posterior")
                            "offline" -> use the Beta prior plus offline evidence.
                            "online" -> use the Beta prior plus online evidence.
                            "posterior" -> use the Beta prior plus offline and online evidence (i.e. always use all available information)
query_demonstrations:       if True, after each grasp the script asks the user if the want to provide a demo. Otherwise, we only learn from grasp outcomes.
save_scene_data:            if True, we save scene data (pointclouds, grasp proposals...) for the scenes we trained on
resume_training_by_default: 


Flow:
-   Start by looking at <experiment_path>/<train_object_set>/seed_<experiment_seed> and <experiment_path>/<policy_name>.
    If we find a cl_module already there, we can resume training from there.
    Otherwise, we copy a cl_module from cl_module_source_path
- Build an EnvWrapper (wraps both the sim and real Panda bindings)
- Training: initialize a scene, get an observation, predict grasp proposals, execute the highest-scored grasp that passes collision checks
- Training: use the grasp outcome (from EnvWrapper) to update the scorer (cl_module.add_datapoint)
- Training: optionally, as the user if they want to provide a demonstration. If yes, this updates both scorer and recall (cl_module.add_demonstration)
- Save data and adapted policy


works both w sim and w the real robots
"""

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

from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule

from grasp_cl_integration.experiment_runtime import ConservativeVoxelMap, EnvWrapper, SceneDataSaver, RealPandaCollisionChecker


def ask_yes_no(prompt: str) -> bool:
    response = ""
    while response not in ["y", "n"]:
        response = input(prompt).strip().lower()
    return response == "y"

def main():
    parser = argparse.ArgumentParser(description="Interactively train a grasping policy with grasp outcomes and optional demonstrations.")
    parser.add_argument("--train_object_set", type=str, default="category-airplane", help="Object set to use")
    parser.add_argument("--scene_type", type=str, default="packed", help="Scene type (e.g., packed, sparse)")
    parser.add_argument("--max_object_count", type=int, default=5, help="Max number of objects in the scene")
    parser.add_argument("--num_training_scenes", type=int, default=20, help="Number of training scenes")

    parser.add_argument("--cl_module_source_path", type=str, default="data/grasping_models/grasp-model", help="Path to the cl module (if any)")
    parser.add_argument("--perception_type", type=str, default="singleview", help="Type of grasp sampler to use")
    parser.add_argument("--grasp_score_threshold", type=float, default=0.)
    parser.add_argument("--grasp_object_type", type=str, default="unknown", help="Grasp type")
    parser.add_argument("--grasp_type", type=str, default=None, help="Grasp type")

    parser.add_argument("--experiment_path", type=str, default="data/grasping_models/grasp-model-airplane")
    parser.add_argument("--experiment_seed", type=int, default=0)
    parser.add_argument("--policy_name", type=str, default="bootstrapped")

    parser.add_argument("--score_key", type=str, default="score")
    parser.add_argument("--estimation_mode", type=str, default="posterior")
    parser.add_argument("--query_demonstrations", type=bool, default=False)
    parser.add_argument("--save_scene_data", type=bool, default=False)
    parser.add_argument("--resume_training_by_default", type=bool, default=False)
    
    args = parser.parse_args()

    if args.experiment_path is None:
        raise ValueError("Missing args.experiment_path")
    
    if args.grasp_object_type == "pass-category":
        args.grasp_object_type = args.train_object_set
    experiment_path = os.path.join(args.experiment_path, args.train_object_set, f"seed_{args.experiment_seed}")
    policy_path = os.path.join(experiment_path, args.policy_name)
    os.makedirs(experiment_path, exist_ok=True)

    if os.path.exists(policy_path):
        user_response = "" if args.resume_training_by_default is False else "y"
        while user_response not in ["y", "n"]:
            user_response = input(f"policy_path already exists: {policy_path}. Continue training from this policy (y/n)? ")
        if user_response == "n":
            print("Training cancelled.")
            return
        else:
            print(f"Policy already exists at {policy_path}")
            print("\tRESUMING TRAINING")
        cl_module = ContinualLearningModule.load_inplace(
            modules_root=policy_path,
            device="cuda"
        )
    else:
        cl_module = ContinualLearningModule.load_copy(
            modules_root=args.cl_module_source_path,
            new_root=policy_path,
            device="cuda"
        )

    # we train on a random seed, to make sure we are not training on the eval scenes directly
    env_wrapper = EnvWrapper(
        experiment_seed=None,
        scene_type=args.scene_type,
        object_set=args.train_object_set,
        max_object_count=args.max_object_count,
        perception_type=args.perception_type,
        placement_type="reset",
    )
    env_wrapper.placement_type = "reset" if env_wrapper.is_simulation else "front"
    
    real_panda_collision_checker = RealPandaCollisionChecker(max_points=5000)
    conservative_mapper = ConservativeVoxelMap(env_wrapper.env.lower, env_wrapper.env.upper, resolution=0.005) if env_wrapper.is_real_robot else None

    num_attempts = 0
    num_successes = 0
    scene_data_saver = None
    if args.save_scene_data:
        scene_data_saver = SceneDataSaver(os.path.join(experiment_path, "scene_data_train", args.policy_name))

    progress_bar = tqdm(range(args.num_training_scenes), total=args.num_training_scenes, desc="Train")
    for _ in progress_bar:
        reset_result = env_wrapper.reset()
        if reset_result is False:
            break
        image_data = env_wrapper.get_image_data()
        pointcloud_w_table = env_wrapper.image_data_to_pcd(image_data)
        grasp_proposals = cl_module.predict(
            pointcloud_w_table,
            object_type=args.grasp_object_type,
            grasp_type=args.grasp_type,
            grasp_score_threshold=args.grasp_score_threshold,
            score_key=args.score_key,
            estimation_mode=args.estimation_mode)
        if env_wrapper.is_real_robot:
            unsafe_pcd = conservative_mapper(pointcloud_w_table, image_data["eye"][0])
            grasp_proposals = env_wrapper.env.run_grasp_checks(grasp_proposals, refine_pose=True)
            grasp_proposals = real_panda_collision_checker.filter_grasps(
                unsafe_pcd,
                grasp_proposals,
                pose_keys=("pose", "pose_pre"),
            )

        executed_grasp, success = env_wrapper.step(grasp_proposals)
        if executed_grasp is not None:
            print(
                f"success: {bool(success)}",
                f"num proposal {len(grasp_proposals)}",
                f'score: {executed_grasp["score"]:.2f}',
                f'proposal_source: {executed_grasp["proposal_source"]}',
            )
            executed_grasp["success"] = success
            cl_module.add_datapoint(
                pointcloud_w_table=pointcloud_w_table,
                grasps=[executed_grasp],
                object_type=args.grasp_object_type,
            )
        else:
            print(f"success: {bool(success)}", f"num proposal {len(grasp_proposals)}", "no grasp executed")

        if args.query_demonstrations is True and ask_yes_no("Get demo (y/n)?") is True:
            demonstration_grasps = cl_module.get_demonstrations(
                pointcloud_w_table=pointcloud_w_table,
                grasp_proposals=[],
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

        if args.save_scene_data:
            scene_data_saver.save(pointcloud_w_table, grasp_proposals, executed_grasp)

        num_attempts += 1
        num_successes += int(success)
        progress_bar.set_postfix(success_rate=f"{num_successes / num_attempts:.2%}")

    cl_module.save(policy_path)
    ContinualLearningModule.update_training_metadata(policy_path, train_object_set=args.train_object_set)
    print(f"trained_policy={policy_path}")


if __name__ == "__main__":
    main()
