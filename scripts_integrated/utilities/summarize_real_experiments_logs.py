"""
Brief description: Summarize real-robot adaptation logs into compact outcome tables.

Last modified on: 2026-09-14
Commit message: More plotting scripts
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def empty_summary():
    return {
        "total_attempts": 0,
        "total_successes": 0,
        "total_scenes": 0,
        "cleared_scenes": 0,
        "total_recalled": 0,
        "total_sampled": 0
    }


def add_row_to_summary(summary, row):
    summary["total_attempts"] += int(row["num_grasp_attempts"])
    summary["total_successes"] += int(row["num_successful_grasps"])
    summary["cleared_scenes"] += int(row["scene_cleared"])
    summary["total_scenes"] += 1
    summary["total_recalled"] += int(row["num_recalled_grasps"])
    summary["total_sampled"] += int(row["num_sampled_grasps"])


def finalize_summary(summary):
    total_attempts = summary["total_attempts"]
    total_scenes = summary["total_scenes"]

    summary = dict(summary)
    summary["success_rate"] = (
        summary["total_successes"] / total_attempts if total_attempts else 0.0
    )
    summary["scene_clear_rate"] = (
        summary["cleared_scenes"] / total_scenes if total_scenes else 0.0
    )
    summary["recall_rate"] = (
        summary["total_recalled"] / summary["total_attempts"]
    )
    summary["sample_rate"] = (
        summary["total_sampled"] / summary["total_attempts"]
    )
    return summary


def summarize_log(log_path):
    summary = empty_summary()

    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            add_row_to_summary(summary, row)

    return finalize_summary(summary)


def find_log_paths(eval_path):
    eval_path = Path(eval_path)
    if eval_path.is_file():
        return [eval_path]
    return sorted(eval_path.glob("category-*/seed_*/real_log.csv"))


def category_from_log_path(log_path):
    for parent in log_path.parents:
        if parent.name.startswith("category-"):
            return parent.name
    return "unknown"


def summarize_eval(eval_path):
    summaries = defaultdict(empty_summary)

    for log_path in find_log_paths(eval_path):
        category = category_from_log_path(log_path)
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            policy_key = reader.fieldnames[0]
            for row in reader:
                add_row_to_summary(summaries[(category, row[policy_key])], row)

    return {
        category_policy: finalize_summary(summary)
        for category_policy, summary in summaries.items()
    }


def print_summary(summary, indent=""):
    print(
        f"{indent}success_rate: "
        f"{100 * summary['success_rate']:.1f}% "
        f"({summary['total_successes']}/{summary['total_attempts']})"
    )
    print(
        f"{indent}scene_clear_rate: "
        f"{100 * summary['scene_clear_rate']:.1f}% "
        f"({summary['cleared_scenes']}/{summary['total_scenes']})"
    )
    print(
        f"{indent}recall_rate: "
        f"{100 * summary['recall_rate']:.1f}% "
        f"({summary['total_recalled']}/{summary['total_attempts']})"
    )
    print(
        f"{indent}sample_rate: "
        f"{100 * summary['sample_rate']:.1f}% "
        f"({summary['total_sampled']}/{summary['total_attempts']})"
    )



def print_eval_summary(summaries):
    current_category = None
    for category, policy in sorted(summaries):
        if category != current_category:
            if current_category is not None:
                print()
            print(f"{category}:")
            current_category = category
        print(f"  {policy}:")
        print_summary(summaries[(category, policy)], indent="    ")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize real robot eval logs by category and policy."
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default='data_rw_tests_08_09_rebuttal/evals/2026_05_13_ours_eval',
        help="Path to an eval folder or a single real_log.csv.",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    # if eval_path.is_file():
    #     summary = summarize_log(eval_path)
    #     print(f"log_path: {eval_path}")
    #     print_summary(summary)
    #     return

    summaries = summarize_eval(eval_path)
    print(f"eval_path: {eval_path}")
    if not summaries:
        print("No real_log.csv files found.")
        return
    print_eval_summary(summaries)


if __name__ == "__main__":
    main()
