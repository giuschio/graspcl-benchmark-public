"""
Brief description: Evaluate an adapted CL policy on the real Panda and log
scene-level real-robot metrics.

Last modified on: 2026-09-16
Commit message: Huge initial cleanup of unused of deprecated stuff
"""
"""
TLDL: Evaluate a CL module on the real Panda using fixed real scenes and
scene-level logging.

Arguments:
cl_module_path:             directory of the CL module to evaluate. The script
                            loads this exact module with load_inplace. It does
                            not create, copy, or search for a policy.
output_path:                directory where eval outputs are written. The script
                            appends per-scene results to
                            <output_path>/real_log.csv and, if save_scene_data
                            is true, writes scene data under a
                            filesystem-safe policy-nickname folder.
                            Each log row also snapshots training metadata from
                            the CL module config/feedback counts.
eval_object_set:            object set to evaluate on. Required. Also selects
                            the reference scene-view folder.
policy_nickname:            optional display/logging name for the policy. If
                            None, defaults to cl_module_path.

scene_type:                 scene type passed to EnvWrapper.
max_object_count:           maximum object count passed to EnvWrapper.

perception_type:            observation/sampling mode passed to EnvWrapper.
grasp_score_threshold:      minimum score retained by the CL module. In the
                            usual greedy eval this is 0.0.
grasp_object_type:          object type passed to CL prediction. If set to
                            "pass-category", this is replaced with
                            eval_object_set.
grasp_type:                 optional grasp type passed to CL prediction.

experiment_seed:            seed passed to EnvWrapper.


score_key:                  grasp score field used by the CL module. Usually
                            "score".
estimation_mode:            scoring evidence mode:
                            "offline" uses prior plus offline evidence;
                            "online" uses prior plus online evidence;
                            "posterior" uses all available evidence.
use_recall:                 if true, add recalled grasps to the proposals.
use_sampler:                if true, add heuristic sampled grasps.
use_scoring:                if true, score/rank proposals before execution. If
                            false, EnvWrapper receives unranked proposals.
eval_scenes:                number of real scenes to evaluate.
eval_steps_per_scene:       maximum grasp attempts per real scene.
scene_views_path:           root folder containing top-down reference images.
                            The script uses
                            <scene_views_path>/<eval_object_set>.
scene_view_alpha:           alpha used by the live/reference reset overlay.
save_scene_data:            if true, save pointclouds and grasp proposal data.

Flow:
- Validate and load cl_module_path in place.
- Infer the training set from CL module metadata. If the metadata is missing,
  ask the user what category the model was trained on.
- Build an EnvWrapper and require that it is connected to the real robot.
- Load existing rows from <output_path>/real_log.csv and skip matching completed
  scene indices so interrupted evals can resume.
- For each pending real scene, reset with the corresponding reference image.
- Repeatedly observe RGBD, convert it to a pointcloud, and filter out the table.
- If no objects are visible after an attempt, ask whether the scene was cleared.
- Call cl_module.predict with the recall/sampling/scoring flags.
- Run real-robot grasp checks, pose refinement, and conservative collision
  filtering.
- Execute the selected grasp. Motion failures that return success=None are
  repeated without logging.
- Stop a scene after success/clearance, after three consecutive failures, or
  after eval_steps_per_scene attempts.
- Append one row per scene to <output_path>/real_log.csv, including the
  resolved training set and per-category training feedback counts.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
for submodule_path in (REPO_ROOT / "submodules").iterdir():
    if submodule_path.is_dir():
        sys.path.insert(0, str(submodule_path))

from grasp_cl_integration.common import (
    filesystem_safe_name,
    grasp_source_key,
    load_completed_scene_rows,
    parse_bool,
    RealEvalLogger,
    resolve_grasp_object_type,
    resolve_policy_nickname,
    resolve_train_object_set,
)


def ask_yes_no(prompt: str) -> bool:
    response = ""
    while response not in ["y", "n"]:
        response = input(prompt).strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a provided CL module on the real robot and log results.")
    parser.add_argument("--cl_module_path", type=str, required=True, help="Path to the CL module to load")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where logs and scene data are saved")
    parser.add_argument("--eval_object_set", type=str, required=True, help="Object set to evaluate on")
    parser.add_argument("--policy_nickname", type=str, default=None)

    parser.add_argument("--scene_type", type=str, default="packed", help="Scene type (e.g., packed, sparse)")
    parser.add_argument("--max_object_count", type=int, default=5, help="Max number of objects in the scene")
    
    parser.add_argument("--perception_type", type=str, default="singleview", help="Type of grasp sampler to use")
    parser.add_argument("--grasp_score_threshold", type=float, default=0.)
    parser.add_argument("--grasp_object_type", type=str, default="unknown", help="Grasp type")
    parser.add_argument("--grasp_type", type=str, default=None, help="Grasp type")

    parser.add_argument("--experiment_seed", type=int, default=0)
    

    parser.add_argument("--score_key", type=str, default="score")
    parser.add_argument("--estimation_mode", type=str, default="posterior")
    parser.add_argument("--use_recall", type=parse_bool, default=True)
    parser.add_argument("--use_sampler", type=parse_bool, default=True)
    parser.add_argument("--use_scoring", type=parse_bool, default=True)
    parser.add_argument("--eval_scenes", type=int, default=20, help="Number of real scenes to evaluate")
    parser.add_argument("--eval_steps_per_scene", type=int, default=10, help="Maximum number of grasp attempts per scene")
    
    
    parser.add_argument("--scene_views_path", type=str, default="data_local_2/scenes", help="Folder with top-down reference images")
    parser.add_argument("--scene_view_alpha", type=float, default=0.5, help="Live RGB alpha in the reset overlay")
    parser.add_argument("--save_scene_data", type=parse_bool, default=True)

    args = parser.parse_args()
    from grasp_learning_cl.grasp_classification.table_filter import TableFilter
    from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule
    from grasp_cl_integration.experiment_runtime import ConservativeVoxelMap, EnvWrapper, SceneDataSaver, RealPandaCollisionChecker

    training_metadata = ContinualLearningModule.load_training_metadata(args.cl_module_path)
    args.train_object_set = resolve_train_object_set(
        ContinualLearningModule,
        args.cl_module_path,
        prompt_if_missing=True,
    )
    args.policy_nickname = resolve_policy_nickname(args.cl_module_path, args.policy_nickname)
    policy_scene_data_name = filesystem_safe_name(args.policy_nickname)
    args.grasp_object_type = resolve_grasp_object_type(args.grasp_object_type, args.eval_object_set)
    scene_views_path = os.path.join(args.scene_views_path, args.eval_object_set)
    os.makedirs(args.output_path, exist_ok=True)

    # RealEvalLogger still names this metadata field cl_module_source_path.
    args.cl_module_source_path = args.cl_module_path

    print("Executing scripts_integrated/exp_15_adaptation/experiment_15_eval_real.py")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    if not os.path.isdir(args.cl_module_path):
        raise FileNotFoundError(f"Missing cl_module_path: {args.cl_module_path}")
    cl_module = ContinualLearningModule.load_inplace(
        modules_root=args.cl_module_path,
        device="cuda",
    )

    env_wrapper = EnvWrapper(
        experiment_seed=args.experiment_seed,
        scene_type=args.scene_type,
        object_set=args.eval_object_set,
        max_object_count=args.max_object_count,
        perception_type=args.perception_type,
        placement_type="front",
        scene_views_path=scene_views_path,
        scene_view_alpha=args.scene_view_alpha,
    )
    if env_wrapper.is_simulation:
        raise RuntimeError("experiment_15_eval_real.py is intended for the real robot only.")

    # The real hand is a bit different from the simulated one: for safety reasons, we define another collision checker
    real_panda_collision_checker = RealPandaCollisionChecker(max_points=1_000_000)
    conservative_mapper = ConservativeVoxelMap(env_wrapper.env.lower, env_wrapper.env.upper, resolution=0.005)
    table_filter = TableFilter()

    scene_data_saver = None
    if args.save_scene_data:
        scene_data_saver = SceneDataSaver(os.path.join(args.output_path, "scene_data_eval", policy_scene_data_name))

    scene_indices = list(range(args.eval_scenes))
    log_path = os.path.join(args.output_path, "real_log.csv")
    logger = RealEvalLogger(
        csv_path=log_path,
        policy=args.policy_nickname,
        object_set=args.train_object_set,
        train_object_set=args.train_object_set,
        eval_object_set=args.eval_object_set,
        cl_module_source_path=args.cl_module_path,
        base_train_object_set=training_metadata["base_train_object_set"],
        adaptation_object_sets=",".join(training_metadata["adaptation_object_sets"]),
        training_feedback_counts=ContinualLearningModule.format_training_feedback_counts(training_metadata),
    )
    completed_scene_rows = load_completed_scene_rows(log_path, args)
    completed_scene_rows = {idx: row for idx, row in completed_scene_rows.items() if idx in scene_indices}
    if completed_scene_rows:
        for row in completed_scene_rows.values():
            logger.add_completed_row(row)
        print(f"Loaded {log_path}: skipping {len(completed_scene_rows)} completed scenes.")

    start_time = time.time()
    pending_scene_indices = [idx for idx in scene_indices if idx not in completed_scene_rows]
    total_eval_steps = len(pending_scene_indices) * args.eval_steps_per_scene
    pbar = tqdm(total=total_eval_steps, desc="Eval")
    for scene_idx in pending_scene_indices:
        reset_result = env_wrapper.reset(scene_idx=scene_idx)
        if reset_result is False:
            break
        logger.start_scene(scene_idx)
        consecutive_failures = 0
        can_log_scene_clearance = False
        scene_outcome_logged = False
        scene_outcome = "failure"

        eval_step = 0
        while eval_step < args.eval_steps_per_scene:
            image_data = env_wrapper.get_image_data()
            pointcloud_w_table = env_wrapper.image_data_to_pcd(image_data)
            pointcloud_no_table = table_filter(pointcloud_w_table, debug=False)
            if len(pointcloud_no_table.points) < 5:
                if can_log_scene_clearance:
                    scene_cleared = ask_yes_no("No visible objects in the scene. Is the scene cleared (y/n)? ")
                    if scene_cleared:
                        scene_outcome = "success"
                        consecutive_failures = 0
                    else:
                        scene_outcome = "failure"
                else:
                    print("No visible objects in the scene before any grasp attempt.")
                    scene_outcome = "failure"
                scene_outcome_logged = True
                break

            grasp_proposals = cl_module.predict(
                pointcloud_w_table,
                object_type=args.grasp_object_type,
                grasp_type=args.grasp_type,
                grasp_score_threshold=args.grasp_score_threshold,
                score_key=args.score_key,
                estimation_mode=args.estimation_mode,
                use_recall=args.use_recall,
                use_sampler=args.use_sampler,
                use_scoring=args.use_scoring,
                debug=False,
            )
            unsafe_pcd = conservative_mapper(pointcloud_w_table, image_data["eye"][0])
            grasp_proposals = env_wrapper.env.run_grasp_checks(grasp_proposals, refine_pose=True)
            grasp_proposals = real_panda_collision_checker.filter_grasps(
                unsafe_pcd,
                grasp_proposals,
                debug=False,
                pose_keys=("pose", "pose_pre"),
            )

            chosen_grasp, success = env_wrapper.step(
                grasp_proposals,
                run_grasp_checks=False,
                allow_repeat_on_motion_failure=True,
            )

            if success is None:
                print("Repeating attempt without logging.")
                continue

            if chosen_grasp is None:
                logger.log_no_grasp()
                consecutive_failures += 1
            else:
                source_key = grasp_source_key(chosen_grasp)
                logger.log_grasp_outcome(source_key, success)
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            can_log_scene_clearance = True
            eval_step += 1
            pbar.update(1)

            if args.save_scene_data:
                scene_data_saver.save(pointcloud_w_table, grasp_proposals, chosen_grasp)

            pbar.set_postfix(
                scene=scene_idx,
                sr=f"{100 * logger.success_rate:.1f}%",
                attempts=logger.total_attempts,
                cr=f"{100 * logger.scene_clearance_rate:.1f}%",
            )

            if consecutive_failures >= 2:
                print("SCENE NOT CLEARED: 2 consecutive failures on the same scene.")
                scene_outcome = "failure"
                scene_outcome_logged = True
                break
        if not scene_outcome_logged:
            print(f"SCENE NOT CLEARED: reached {args.eval_steps_per_scene} attempts for scene {scene_idx:06d}.")
            scene_outcome = "failure"
        logger.log_scene_outcome(cleared=scene_outcome == "success")
    pbar.close()

    elapsed_seconds = time.time() - start_time
    print(
        f"policy={args.policy_nickname} "
        f"train_object_set={args.train_object_set} "
        f"eval_object_set={args.eval_object_set} "
        f"cl_module_path={args.cl_module_path} "
        f"output_path={args.output_path} "
        f"eval_sr={100 * logger.success_rate:.1f}% "
        f"scene_clearance_rate={100 * logger.scene_clearance_rate:.1f}% "
        f"attempts={logger.total_attempts} "
        f"elapsed_seconds={elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
