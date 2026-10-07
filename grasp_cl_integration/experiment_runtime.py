import csv
import json
import os
import time

import numpy as np
import open3d as o3d

from omegaconf import OmegaConf

from grasp_env.environments.perception_utils import cloud_from_image_data, refine_image_data
from grasp_learning_cl.grasp_classification.collision_checker import CollisionChecker
from simple_panda_grasp.collision import ConservativeVoxelMap

class RealPandaCollisionChecker(CollisionChecker):
    DEFAULT_COLLISION = [
        ((-0.04, -0.10, -0.12), (0.04, 0.10, -0.04)),
        ((-0.045, -0.045, -0.30), (0.045, 0.045, -0.12)),
        ((0.02, -0.045, -0.08), (0.09, 0.045, -0.04)),
    ]


class EnvWrapper:
    def __init__(
        self,
        experiment_seed,
        scene_type,
        object_set,
        max_object_count,
        perception_type,
        placement_type="side",
        scene_views_path=None,
        scene_view_alpha=0.5,
    ):
        """Initialize the environment wrapper.

        Args:
            experiment_seed: Seed used for environment setup and reset sampling.
            scene_type: Scene configuration passed to the simulated environment.
            object_set: Object set passed to the simulated environment.
            max_object_count: Maximum number of objects sampled on reset.
            perception_type: Observation mode, such as "multiview" or random viewpoint.
            placement_type: Placement mode, either "side" or "reset".
            scene_views_path: Optional folder with top-down RGB reference images for guided real resets.
            scene_view_alpha: Alpha used for live/reference overlay during guided real resets.
        """
        machine_config = OmegaConf.load("./machine_config.yaml")
        self.is_simulation = machine_config["machine_type"] == "simulation"
        self.is_real_robot = not self.is_simulation
        self.max_object_count = max_object_count
        self.perception_type = perception_type
        self.rng = np.random.default_rng(experiment_seed)
        self.placement_type = placement_type
        self.scene_views_path = scene_views_path
        self.scene_view_alpha = scene_view_alpha
        self._next_failed_observation_pose_idx = None

        if self.is_simulation:
            from grasp_env.environments.env import GraspEnv
            self.env = GraspEnv(
                seed=experiment_seed,
                scene_type=scene_type,
                object_set=object_set,
                randomize_object_scale=False,
                show_gui=False,
                output_video=None)
        else:
            from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
            self.env = PandaGraspInterface(machine_config["panda_config_path"])
            
    def reset(self, scene_idx=None):
        self._next_failed_observation_pose_idx = None
        if scene_idx is not None and self.is_real_robot and self.scene_views_path is not None:
            return self.reset_to_scene(scene_idx)
        return self.env.reset(object_count=self.rng.integers(1, self.max_object_count, endpoint=True))

    def get_top_down_image_data(self):
        if hasattr(self.env, "get_observation_top_down"):
            return self.env.get_observation_top_down()
        return self.get_image_data()

    def get_top_down_rgb(self):
        image_data = self.get_top_down_image_data()
        return image_data["rgb_imgs"][0]

    def get_live_rgb(self):
        if hasattr(self.env, "get_picture"):
            camera = getattr(self.env, "camera", None)
            if camera is not None and getattr(camera, "pipeline", None) is not None:
                frames = camera.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    raise RuntimeError("Failed to capture color frame.")
                color = np.asanyarray(color_frame.get_data())
                return np.ascontiguousarray(color[:, :, ::-1])
            color, _ = self.env.get_picture()
            return color[:, :, ::-1]
        return self.get_top_down_rgb()

    def reset_to_scene(self, scene_idx):
        if self.is_simulation:
            return self.reset()
        if self.scene_views_path is None:
            return self.env.reset()

        reference_rgb = self._load_scene_reference(scene_idx)
        import cv2

        window_name = f"Reset to scene {scene_idx:06d} - Enter=done, Esc/Q=abort"
        print(f"Reset scene {scene_idx:06d}. Press Enter in the image window when done, or Esc/Q to abort.")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        self._maximize_cv2_window(window_name)
        reference_rgb = np.rot90(reference_rgb, 2)
        self.get_top_down_image_data()
        try:
            while True:
                live_rgb = np.rot90(self.get_live_rgb(), 2)
                display_rgb = self._make_scene_reset_overlay(live_rgb, reference_rgb)
                display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, display_bgr)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Aborted by user")
                    return False
                key = cv2.waitKey(1) & 0xFF
                if key in (10, 13):
                    return True
                if key in (27, ord("q"), ord("Q")):
                    print("Aborted by user")
                    return False
                time.sleep(0.03)
        finally:
            cv2.destroyWindow(window_name)

    def _maximize_cv2_window(self, window_name):
        import cv2

        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            cv2.moveWindow(window_name, 0, 0)
            cv2.resizeWindow(window_name, width, height)
        except Exception:
            cv2.resizeWindow(window_name, 1600, 1000)

    def save_scene_reference(self, scene_idx, scene_views_path=None):
        scene_views_path = scene_views_path or self.scene_views_path
        if scene_views_path is None:
            raise ValueError("Missing scene_views_path")
        os.makedirs(scene_views_path, exist_ok=True)

        image_path = os.path.join(scene_views_path, f"scene_{scene_idx:06d}.png")
        rgb = self.get_top_down_rgb()
        import cv2

        cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return image_path

    def _load_scene_reference(self, scene_idx):
        image_path = self._load_scene_reference_path(scene_idx)
        if image_path is not None:
            import cv2

            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Failed to read scene reference image: {image_path}")
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidates = self._scene_reference_candidates(scene_idx)
        raise FileNotFoundError(
            f"Missing reference image for scene_idx={scene_idx}. Tried: {', '.join(candidates)}"
        )

    def _load_scene_reference_path(self, scene_idx):
        for image_path in self._scene_reference_candidates(scene_idx):
            if os.path.exists(image_path):
                return image_path
        return None

    def _scene_reference_candidates(self, scene_idx):
        return [
            os.path.join(self.scene_views_path, f"scene_{scene_idx:06d}.png"),
            os.path.join(self.scene_views_path, f"scene_{scene_idx}.png"),
            os.path.join(self.scene_views_path, f"{scene_idx}.png"),
        ]

    def _make_scene_reset_overlay(self, live_rgb, reference_rgb):
        import cv2

        if live_rgb.shape[:2] != reference_rgb.shape[:2]:
            reference_rgb = cv2.resize(
                reference_rgb,
                (live_rgb.shape[1], live_rgb.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        alpha = float(np.clip(self.scene_view_alpha, 0.0, 1.0))
        overlay = cv2.addWeighted(live_rgb, alpha, reference_rgb, 1.0 - alpha, 0.0)
        return overlay.astype(np.uint8)

    def get_image_data(self):
        if self._should_use_failed_observation_pose():
            image_data = self._get_failed_observation_pose_image_data()
        else:
            obs_func = (
                self.env.get_observation_images_fixed_viewpoints
                if self.perception_type == "multiview"
                else self.env.get_observation_random_viewpoint
            )
            image_data = obs_func()
        if self.is_real_robot is True:
            image_data = refine_image_data(
                image_data,
                voxel_size=0.003,
                depth_trunc=0.7,
                max_correspondence_distance=0.015
            )
        self.last_image_data = image_data
        return image_data

    def _should_use_failed_observation_pose(self):
        return (
            self.is_real_robot
            and self.perception_type != "multiview"
            and self._next_failed_observation_pose_idx is not None
        )

    def _get_failed_observation_pose_image_data(self):
        from simple_panda_grasp.panda_constants import SCANNING_JOINTS_RANDOM
        from simple_panda_grasp.utils.dataclasses import CameraIntrinsic

        pose_idx = self._next_failed_observation_pose_idx
        joint_positions = SCANNING_JOINTS_RANDOM[pose_idx]
        success = self.env.move_jj(joint_positions)
        if success is False:
            raise RuntimeError(
                f"Failed to move camera to SCANNING_JOINTS_RANDOM[{pose_idx}]"
            )
        time.sleep(1.0)

        n_views = 1
        eyes = np.empty((n_views, 3), np.float32)
        extrinsics = np.empty((n_views, 7), np.float32)
        depth_imgs = np.empty((n_views, self.env.camera.height, self.env.camera.width), np.float32)
        rgb_imgs = np.empty((n_views, self.env.camera.height, self.env.camera.width, 3), np.uint8)

        cam2world = self.env.get_cam2world()
        color, depth = self.env.get_picture()
        color = color[:, :, ::-1]

        extrinsics[0] = cam2world.inverse().to_list()
        depth_imgs[0] = depth
        rgb_imgs[0] = color
        eyes[0] = cam2world.translation

        intrinsics = CameraIntrinsic(
            width=self.env.camera.color_intrinsics["width"],
            height=self.env.camera.color_intrinsics["height"],
            fx=self.env.camera.color_intrinsics["fy"],
            fy=self.env.camera.color_intrinsics["fx"],
            cx=self.env.camera.color_intrinsics["cx"],
            cy=self.env.camera.color_intrinsics["cy"],
        )

        return {
            "depth_imgs": depth_imgs,
            "rgb_imgs": rgb_imgs,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "eye": eyes,
        }

    def _schedule_failed_observation_pose(self):
        if not self.is_real_robot or self.perception_type == "multiview":
            return
        from simple_panda_grasp.panda_constants import SCANNING_JOINTS_RANDOM

        if len(SCANNING_JOINTS_RANDOM) == 0:
            return
        if self._next_failed_observation_pose_idx is None:
            self._next_failed_observation_pose_idx = 0
        else:
            self._next_failed_observation_pose_idx += 1
            self._next_failed_observation_pose_idx %= len(SCANNING_JOINTS_RANDOM)
        print(
            "Next observation will use "
            f"SCANNING_JOINTS_RANDOM[{self._next_failed_observation_pose_idx}]."
        )

    def _record_grasp_result(self, success):
        if success:
            self._next_failed_observation_pose_idx = None
        else:
            self._schedule_failed_observation_pose()

    def image_data_to_pcd(self, image_data):
        obs = cloud_from_image_data(
            image_data,
            lower_bounds=self.env.sim.lower - [0, 0, 0.04] if self.is_simulation else self.env.lower,
            upper_bounds=self.env.sim.upper if self.is_simulation else self.env.upper,
            noise_level=0.0008 if self.is_simulation else 0.)
        self.last_observation = obs
        return obs

    def get_observation(self):
        image_data = self.get_image_data()
        return self.image_data_to_pcd(image_data)

    def confirm_grasp_execution(self, grasp):
        import open3d as o3d
        from grasp_learning_cl.visualization import visualize_scene_pcd_util
        from grasp_learning_cl.grasp_classification.table_filter import TableFilter

        state = {"execute": None}

        def execute(vis):
            state["execute"] = True
            vis.close()
            return True

        def skip(vis):
            state["execute"] = False
            vis.close()
            return True

        observation = o3d.geometry.PointCloud(self.last_observation)
        try:
            pointcloud_no_table = TableFilter()(observation, debug=False)
            full_points = np.asarray(observation.points)
            no_table_points = np.asarray(pointcloud_no_table.points)
            if full_points.size:
                no_table_set = {tuple(point) for point in no_table_points}
                table_mask = np.array(
                    [tuple(point) not in no_table_set for point in full_points],
                    dtype=bool,
                )
                colors = np.asarray(observation.colors)
                if colors.shape[0] != full_points.shape[0]:
                    colors = np.tile(np.array([0.6, 0.6, 0.6]), (full_points.shape[0], 1))
                colors[table_mask] = np.array([0.0, 0.0, 1.0])
                observation.colors = o3d.utility.Vector3dVector(colors)
        except Exception as exc:
            print(f"TableFilter visualization coloring failed: {exc}")

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Execute grasp? Y=yes, N=no")
        for geometry in visualize_scene_pcd_util(
            observation,
            [grasp],
            visualize_ref_system=True,
            color_key="score",
            visualize_grasp_rotation=False
        ):
            vis.add_geometry(geometry)
        vis.register_key_callback(ord("Y"), execute)
        vis.register_key_callback(ord("N"), skip)
        vis.run()
        vis.destroy_window()

        return bool(state["execute"])

    def _ask_yes_no(self, prompt: str) -> bool:
        response = ""
        while response not in ["y", "n"]:
            response = input(prompt).strip().lower()
        return response == "y"

    def step(self, grasps, run_grasp_checks=True, allow_repeat_on_motion_failure=False):
        if self.is_real_robot and run_grasp_checks:
            grasps = self.env.run_grasp_checks(grasps, refine_pose=True)

        if len(grasps) == 0:
            print("No grasps found in the scene")
            self._record_grasp_result(False)
            return None, False

        chosen_grasp = grasps[0]
        if self.is_simulation:
            res = self.env.step_robust(
                chosen_grasp["pose"],
                width_pre=chosen_grasp["width"],
                noise=None,
                remove_grasped_object=True,
                num_trials=1)
            success = res["success"]
        else:
            # if self.confirm_grasp_execution(chosen_grasp):
            if True:
                pick_success, pick_msg = self.env.pick(chosen_grasp)
                motion_failed = not pick_success and pick_msg.startswith("Failed to move")
                if motion_failed:
                    print(pick_msg)
                    if allow_repeat_on_motion_failure and self._ask_yes_no(
                        "Repeat this observation and grasp attempt without logging (y/n)? "
                    ):
                        return chosen_grasp, None
                    success = False
                else:
                    res = ""
                    while res not in ["y", "n"]:
                        res = input("Was the grasp successful (y/n)? ")
                    success = bool(res == "y")
                place_fns = {
                    "side": self.env.place,
                    "reset": self.env.place_back,
                    "front": self.env.place_front,
                }

                place_fn = place_fns[self.placement_type]
                place_fn = place_fn if bool(success) is True else self.env.place_reset
                place_fn()
            else:
                success, msg = False, "User preempted the grasp"
            # self.env.move_to_start()

        chosen_grasp["success"] = success
        self._record_grasp_result(success)
        return chosen_grasp, success

class Logger:
    def __init__(self, folder):
        self.folder = folder
        self.params_path = os.path.join(folder, "run_state.json")
        self.scalars_path = os.path.join(folder, "scalars.csv")
        self.params = {}

        os.makedirs(folder, exist_ok=True)
        with open(self.params_path, "w") as f:
            json.dump(self.params, f, indent=2)
        with open(self.scalars_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "key", "value"])

    def log_scalar(self, key, value, step):
        with open(self.scalars_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, key, value])

    def log_param(self, key, value):
        self.params[key] = value
        with open(self.params_path, "w") as f:
            json.dump(self.params, f, indent=2)

    @staticmethod
    def load(folder):
        params_path = os.path.join(folder, "run_state.json")
        scalars_path = os.path.join(folder, "scalars.csv")
        params = {}
        scalars = []

        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = json.load(f)
        if os.path.exists(scalars_path):
            with open(scalars_path, "r", newline="") as f:
                scalars = list(csv.DictReader(f))

        return {"params": params, "scalars": scalars}


class SceneDataSaver:
    def __init__(self, scene_data_path):
        import open3d as o3d
        from grasp_learning_cl.grasp_datagen.datasets import SceneGraspDataHandler

        self.scene_data_path = scene_data_path
        self.o3d = o3d
        self.data_handler = SceneGraspDataHandler
        os.makedirs(scene_data_path, exist_ok=True)
        self.next_scene_data_idx = self._find_next_scene_data_idx()

    def _find_next_scene_data_idx(self):
        max_scene_data_idx = -1
        for entry in os.scandir(self.scene_data_path):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("scene_") and name[6:].isdigit():
                max_scene_data_idx = max(max_scene_data_idx, int(name[6:]))
        return max_scene_data_idx + 1

    def _json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def save(self, pointcloud_w_table, grasp_proposals, executed_grasp):
        scene_path = os.path.join(self.scene_data_path, f"scene_{self.next_scene_data_idx:06d}")
        self.next_scene_data_idx += 1
        os.makedirs(scene_path, exist_ok=True)

        colors = np.asarray(pointcloud_w_table.colors)
        colors = np.clip(colors, 0.0, 1.0)
        pointcloud_w_table.colors = self.o3d.utility.Vector3dVector(colors)

        self.o3d.io.write_point_cloud(os.path.join(scene_path, "cloud_w_table.ply"), pointcloud_w_table)
        grasp_proposals = [self._json_safe(grasp) for grasp in grasp_proposals]
        self.data_handler.data_to_file(grasp_proposals, os.path.join(scene_path, "grasp_proposals.csv"))
        executed_grasps = []
        if executed_grasp is not None:
            executed_grasp = self._json_safe(executed_grasp.copy())
            if "success" in executed_grasp:
                executed_grasp["success"] = bool(executed_grasp["success"])
            executed_grasps.append(executed_grasp)
        self.data_handler.data_to_file(executed_grasps, os.path.join(scene_path, "grasps.csv"))

