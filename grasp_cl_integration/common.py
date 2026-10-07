import csv
import os


def log_results_to_csv(csv_path, **kwargs):
    """Logs key-value pairs as a CSV row. Adds header if file doesn't exist."""
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(kwargs.keys())

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(kwargs)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def default_policy_nickname(cl_module_path):
    return cl_module_path


def prompt_for_train_object_set(cl_module_path):
    print(f"Could not find what this model was trained on in {cl_module_path}.")
    while True:
        response = input("What category was this model trained on? ").strip()
        if response:
            return response
        print("Please enter a non-empty category name.")


def resolve_train_object_set(cl_module, cl_module_path, prompt_if_missing=False):
    inferred_train_object_set = cl_module.infer_train_object_set(cl_module_path)
    if inferred_train_object_set is not None:
        return inferred_train_object_set
    if prompt_if_missing:
        try:
            return prompt_for_train_object_set(cl_module_path)
        except EOFError as exc:
            raise ValueError(
                "Missing train_object_set and could not infer it from the CL module. "
                "Answer the prompt interactively or add train_object_set to the module config.json."
            ) from exc
    else:
        raise ValueError(
            "Missing train_object_set and could not infer it from the CL module. "
            "Answer the prompt interactively or add train_object_set to the module config.json."
        )


def resolve_policy_nickname(cl_module_path, policy_nickname):
    return default_policy_nickname(cl_module_path) if policy_nickname is None else policy_nickname


def filesystem_safe_name(name):
    return str(name).strip().replace(os.sep, "_")


def resolve_grasp_object_type(grasp_object_type, eval_object_set):
    if grasp_object_type == "pass-category":
        return eval_object_set
    return grasp_object_type


REAL_LOG_FIELDNAMES = [
    "policy",
    "object_set",
    "train_object_set",
    "eval_object_set",
    "cl_module_source_path",
    "base_train_object_set",
    "adaptation_object_sets",
    "training_feedback_counts",
    "scene_idx",
    "num_grasp_attempts",
    "num_sampled_grasps",
    "num_recalled_grasps",
    "num_successful_grasps",
    "num_successful_sampled_grasps",
    "num_successful_recalled_grasps",
    "scene_cleared",
]


def grasp_source_key(grasp, known_sources=None) -> str:
    source = grasp.get("proposal_source", "")
    if source.startswith("recall"):
        source_key = "recall"
    else:
        source_key = source or "sampler"
    if known_sources is not None and source_key not in known_sources:
        return "unknown"
    return source_key


def scene_log_matches(row, args) -> bool:
    return (
        row.get("policy") == args.policy_nickname
        and row.get("train_object_set") == args.train_object_set
        and row.get("eval_object_set") == args.eval_object_set
        and row.get("cl_module_source_path", args.cl_module_source_path) == args.cl_module_source_path
    )


def load_completed_scene_rows(scene_log_path, args):
    if not os.path.isfile(scene_log_path):
        return {}
    completed = {}
    with open(scene_log_path, newline="") as f:
        for row in csv.DictReader(f):
            if scene_log_matches(row, args):
                completed[int(row["scene_idx"])] = row
    return completed


class RealEvalLogger:
    def __init__(
        self,
        csv_path,
        policy,
        object_set,
        train_object_set,
        eval_object_set,
        cl_module_source_path,
        base_train_object_set="",
        adaptation_object_sets="",
        training_feedback_counts="",
    ):
        self.csv_path = csv_path
        self.metadata = {
            "policy": policy,
            "object_set": object_set,
            "train_object_set": train_object_set,
            "eval_object_set": eval_object_set,
            "cl_module_source_path": cl_module_source_path,
            "base_train_object_set": base_train_object_set,
            "adaptation_object_sets": adaptation_object_sets,
            "training_feedback_counts": training_feedback_counts,
        }
        self.total_attempts = 0
        self.total_successes = 0
        self.total_cleared_scenes = 0
        self.total_scenes = 0
        self.start_scene(None)

    @property
    def success_rate(self) -> float:
        return self.total_successes / max(self.total_attempts, 1)

    @property
    def scene_clearance_rate(self) -> float:
        return self.total_cleared_scenes / max(self.total_scenes, 1)

    def add_completed_row(self, row) -> None:
        self.total_attempts += int(row.get("num_grasp_attempts", 0))
        self.total_successes += int(row.get("num_successful_grasps", 0))
        self.total_cleared_scenes += int(row.get("scene_cleared", 0))
        self.total_scenes += 1

    def start_scene(self, scene_idx) -> None:
        self.scene_idx = scene_idx
        self.num_grasp_attempts = 0
        self.num_sampled_grasps = 0
        self.num_recalled_grasps = 0
        self.num_successful_grasps = 0
        self.num_successful_sampled_grasps = 0
        self.num_successful_recalled_grasps = 0

    def log_no_grasp(self) -> None:
        self.num_grasp_attempts += 1
        self.total_attempts += 1

    def log_grasp_outcome(self, source, successful) -> None:
        self.num_grasp_attempts += 1
        self.total_attempts += 1

        if source == "recall":
            self.num_recalled_grasps += 1
        else:
            self.num_sampled_grasps += 1

        if not successful:
            return

        self.num_successful_grasps += 1
        self.total_successes += 1
        if source == "recall":
            self.num_successful_recalled_grasps += 1
        else:
            self.num_successful_sampled_grasps += 1

    def log_scene_outcome(self, cleared) -> None:
        row = {
            **self.metadata,
            "scene_idx": self.scene_idx,
            "num_grasp_attempts": self.num_grasp_attempts,
            "num_sampled_grasps": self.num_sampled_grasps,
            "num_recalled_grasps": self.num_recalled_grasps,
            "num_successful_grasps": self.num_successful_grasps,
            "num_successful_sampled_grasps": self.num_successful_sampled_grasps,
            "num_successful_recalled_grasps": self.num_successful_recalled_grasps,
            "scene_cleared": int(bool(cleared)),
        }
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REAL_LOG_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self.total_scenes += 1
        if cleared:
            self.total_cleared_scenes += 1
