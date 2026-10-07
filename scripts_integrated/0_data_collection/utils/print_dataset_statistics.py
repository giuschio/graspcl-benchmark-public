#!/usr/bin/env python3
"""
Brief description: Compute dataset statistics for saved grasp training scenes.

Last modified on: 2026-04-20
Commit message: Some reorganization
"""


from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np

from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the average success rate across all grasps in a scene dataset.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("data/training_data"),
        help="Folder containing scene subdirectories with cloud and grasps files.",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="success_binary",
        help="Grasp label to average, e.g. success_percentage or success_binary.",
    )
    args = parser.parse_args()

    scene_dirs = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    if not scene_dirs:
        raise ValueError(f"No scenes with grasps found in {args.dataset_root}")

    labels = []
    sampler_grasp_counts = defaultdict(int)
    sampler_labels = defaultdict(list)
    used_scene_count = 0
    for scene_dir in scene_dirs:
        grasps_path = scene_dir / "grasps.csv"
        if not grasps_path.exists():
            continue

        grasps = SceneGraspDataHandler.from_file(str(grasps_path)).data
        scene_labels = [float(grasp[args.label_key]) for grasp in grasps if args.label_key in grasp]
        if not scene_labels:
            continue

        labels.extend(scene_labels)
        used_scene_count += 1

        for grasp in grasps:
            sampler = grasp.get("sampler")
            if sampler is None:
                continue

            sampler_grasp_counts[sampler] += 1
            if args.label_key in grasp:
                sampler_labels[sampler].append(float(grasp[args.label_key]))

    labels = np.asarray(labels, dtype=np.float32)
    if labels.size == 0:
        raise ValueError(f"No grasp labels '{args.label_key}' found in {args.dataset_root}")

    print(f"dataset_root: {args.dataset_root}")
    print(f"label_key: {args.label_key}")
    print(f"num_scenes: {used_scene_count}")
    print(f"num_grasps: {labels.size}")
    print(f"average_success_rate: {labels.mean():.6f}")
    print("per_sampler_statistics:")
    for sampler in sorted(sampler_grasp_counts):
        sampler_num_grasps = sampler_grasp_counts[sampler]
        sampler_scene_average = sampler_num_grasps / used_scene_count
        sampler_success_rate = np.asarray(sampler_labels[sampler], dtype=np.float32).mean()
        print(
            f"  {sampler}: average_grasps_per_scene={sampler_scene_average:.6f}, "
            f"average_success_rate={sampler_success_rate:.6f}, "
        )


if __name__ == "__main__":
    main()
