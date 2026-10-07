"""
Brief description: Evaluate an adapted CL policy and log success by proposal source.

Last modified on: 2026-05-11
Commit message: Log success rate of grasps by source
"""
"""
TLDL: Evaluate a CL module in simulation on objects from eval_object_set.
This is the lightweight simulation evaluator for exp 15 policies.

Arguments:
cl_module_path:             directory of the CL module to evaluate. The script
                            loads this exact module with load_inplace. It does
                            not create, copy, or search for a policy.
output_path:                directory where eval outputs are written. The script
                            appends aggregate results to <output_path>/log.csv
                            and, if save_scene_data is true, writes scene data
                            under a filesystem-safe policy-nickname folder.
                            The log row also snapshots training metadata from
                            the CL module config/feedback counts.
eval_object_set:            object set to evaluate on. Required.
                            scenes.
policy_nickname:            optional display/logging name for the policy. If
                            None, defaults to cl_module_path.

scene_type:                 simulation scene type, e.g. packed, sparse, pile.
max_object_count:           maximum number of objects sampled into each scene.

perception_type:            observation/sampling mode passed to EnvWrapper.
grasp_score_threshold:      minimum score retained by the CL module. In the
                            usual greedy eval this is 0.0.
grasp_object_type:          object type passed to CL prediction. If set to
                            "pass-category", this is replaced with
                            eval_object_set.
grasp_type:                 optional grasp type passed to CL prediction.

experiment_seed:            seed passed to EnvWrapper for reproducible sim


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
eval_steps:                 number of simulated scenes/grasp attempts.
save_scene_data:            if true, save pointclouds and grasp proposal data.

Flow:
- Validate and load cl_module_path in place.
- Infer the training set from CL module metadata. If the metadata is missing,
  ask the user what category the model was trained on.
- Build an EnvWrapper in simulation mode.
- For each eval step, reset the simulated scene.
- Convert the RGBD observation to a pointcloud with EnvWrapper.
- Call cl_module.predict with the recall/sampling/scoring flags.
- Execute the selected grasp through EnvWrapper.
- Update aggregate grasp metrics, including proposal source success rates.
- Append one aggregate row to <output_path>/log.csv, including the resolved
  training set and per-category training feedback counts.
"""

import argparse
import csv
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
    parse_bool,
    resolve_grasp_object_type,
    resolve_policy_nickname,
    resolve_train_object_set,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a provided CL module in simulation and log results.")
    parser.add_argument("--cl_module_path", type=str, default="data/grasping_models/grasp-model-airplane/category-airplane/seed_0/bootstrapped", help="Path to the CL module to load")
    parser.add_argument("--output_path", type=str, default="data/experiments/grasp-model-airplane", help="Directory where logs and scene data are saved")
    parser.add_argument("--eval_object_set", type=str, default="category-airplane", help="Object set to evaluate on")
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
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of evaluation scenes per evaluation phase")
    parser.add_argument("--save_scene_data", type=parse_bool, default=False)

    args = parser.parse_args()
    from grasp_cl_integration.experiment_runtime import EnvWrapper, SceneDataSaver
    from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule

    training_metadata = ContinualLearningModule.load_training_metadata(args.cl_module_path)
    args.train_object_set = resolve_train_object_set(
        ContinualLearningModule,
        args.cl_module_path,
        prompt_if_missing=True,
    )
    args.policy_nickname = resolve_policy_nickname(args.cl_module_path, args.policy_nickname)
    policy_scene_data_name = filesystem_safe_name(args.policy_nickname)
    args.grasp_object_type = resolve_grasp_object_type(args.grasp_object_type, args.eval_object_set)
    os.makedirs(args.output_path, exist_ok=True)

    print("Executing scripts_integrated/exp_15_adaptation/experiment_15_eval.py")
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
        placement_type="front"
    )
    if not env_wrapper.is_simulation:
        raise RuntimeError("experiment_15_eval.py is intended for simulation only.")

    stats = {"total": 0., "success": 0., "failure": 0., "no_grasp": 0.}
    scene_stats = {"success": 0.0, "failure": 0.0, "skipped_empty": 0.0}
    object_counts = []
    executed_grasp_sources = {"sampler": 0, "recall": 0, "unknown": 0}
    executed_grasp_successes = {"sampler": 0, "recall": 0, "unknown": 0}
    scene_data_saver = None
    if args.save_scene_data:
        scene_data_saver = SceneDataSaver(os.path.join(args.output_path, "scene_data_eval", policy_scene_data_name))

    start_time = time.time()
    pbar = tqdm(range(args.eval_steps), total=args.eval_steps, desc="Eval")
    steps_done = 0
    for _ in pbar:
        reset_result = env_wrapper.reset()
        if reset_result is False:
            break
        steps_done += 1

        image_data = env_wrapper.get_image_data()
        pointcloud_w_table = env_wrapper.image_data_to_pcd(image_data)
        object_counts.append(env_wrapper.env.sim.num_objects)
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
        chosen_grasp, success = env_wrapper.step(grasp_proposals)

        stats["total"] += 1
        if chosen_grasp is None:
            stats["no_grasp"] += 1
        else:
            source_key = grasp_source_key(chosen_grasp, known_sources=executed_grasp_sources)
            executed_grasp_sources[source_key] += 1
            if success:
                stats["success"] += 1
                executed_grasp_successes[source_key] += 1
            else:
                stats["failure"] += 1

        if args.save_scene_data:
            scene_data_saver.save(pointcloud_w_table, grasp_proposals, chosen_grasp)

        total = max(stats["total"], 1.0)
        scene_total = max(scene_stats["success"] + scene_stats["failure"], 1.0)
        scene_clearance_rate = scene_stats["success"] / scene_total
        pbar.set_postfix(
            sr=f"{100 * stats['success'] / total:.1f}%",
            fr=f"{100 * stats['failure'] / total:.1f}%",
            ngr=f"{100 * stats['no_grasp'] / total:.1f}%",
            cr=f"{100 * scene_clearance_rate:.1f}%",
        )

    elapsed_seconds = time.time() - start_time
    total = max(stats["total"], 1.0)
    scene_total = max(scene_stats["success"] + scene_stats["failure"], 1.0)
    average_num_objects = sum(object_counts) / max(len(object_counts), 1)
    executed_grasps = sum(executed_grasp_sources.values())
    executed_source_rates = {
        source: count / max(executed_grasps, 1)
        for source, count in executed_grasp_sources.items()
    }
    executed_source_success_rates = {
        source: executed_grasp_successes[source] / max(count, 1)
        for source, count in executed_grasp_sources.items()
    }
    row = {
        "policy": args.policy_nickname,
        "object_set": args.train_object_set,
        "train_object_set": args.train_object_set,
        "eval_object_set": args.eval_object_set,
        "base_train_object_set": training_metadata["base_train_object_set"],
        "adaptation_object_sets": ",".join(training_metadata["adaptation_object_sets"]),
        "training_feedback_counts": ContinualLearningModule.format_training_feedback_counts(training_metadata),
        "experiment_seed": args.experiment_seed,
        "estimation_mode": args.estimation_mode,
        "score_key": args.score_key,
        "eval_steps": args.eval_steps,
        "steps_done": steps_done,
        "success": stats["success"],
        "failure": stats["failure"],
        "no_grasp": stats["no_grasp"],
        "total": stats["total"],
        "success_rate": stats["success"] / total,
        "failure_rate": stats["failure"] / total,
        "no_grasp_rate": stats["no_grasp"] / total,
        "scene_success": scene_stats["success"],
        "scene_failure": scene_stats["failure"],
        "scene_clearance_rate": scene_stats["success"] / scene_total,
        "average_num_objects": average_num_objects,
        "executed_grasps": executed_grasps,
        "executed_sampler_rate": executed_source_rates["sampler"],
        "executed_recall_rate": executed_source_rates["recall"],
        "executed_unknown_rate": executed_source_rates["unknown"],
        "sampled_grasp_success_rate": executed_source_success_rates["sampler"],
        "recalled_grasp_success_rate": executed_source_success_rates["recall"],
        "unknown_grasp_success_rate": executed_source_success_rates["unknown"],
        "elapsed_seconds": elapsed_seconds,
    }

    log_path = os.path.join(args.output_path, "log.csv")
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"policy={args.policy_nickname} "
        f"train_object_set={args.train_object_set} "
        f"eval_object_set={args.eval_object_set} "
        f"eval_sr={100 * row['success_rate']:.1f}% "
        f"eval_fr={100 * row['failure_rate']:.1f}% "
        f"eval_ngr={100 * row['no_grasp_rate']:.1f}% "
        f"scene_clearance_rate={100 * row['scene_clearance_rate']:.1f}% "
        f"average_num_objects={average_num_objects:.2f} "
        f"executed_sampler={100 * row['executed_sampler_rate']:.1f}% "
        f"executed_recall={100 * row['executed_recall_rate']:.1f}% "
        f"executed_unknown={100 * row['executed_unknown_rate']:.1f}% "
        f"sampled_grasp_sr={100 * row['sampled_grasp_success_rate']:.1f}% "
        f"recalled_grasp_sr={100 * row['recalled_grasp_success_rate']:.1f}% "
        f"unknown_grasp_sr={100 * row['unknown_grasp_success_rate']:.1f}% "
        f"elapsed_seconds={elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
