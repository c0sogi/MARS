import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from library.config import (
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
    DOWN_RATIO,
    CLASS_NAMES,
    INPUT_DIR,
    WORKING_DIR,
    OUT_SIZE_FACTOR,
)
from library.utils import draw_gaussian, gaussian_radius


def quaternion_to_matrix(q):
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


class LidarDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", load_cached_data=True, num_samples=None
    ):
        self.metadata_path = metadata_path
        self.mode = mode
        self.df = pd.read_csv(metadata_path)

        if num_samples is not None:
            self.df = self.df.iloc[:num_samples]

        # Determine data source directory
        if "test" in mode or "test" in metadata_path:
            self.json_dir = os.path.join(INPUT_DIR, "test_data")
        else:
            self.json_dir = os.path.join(INPUT_DIR, "train_data")

        self.class_map = {name: i for i, name in enumerate(CLASS_NAMES)}

        # Cache file for calibration maps
        cache_file = os.path.join(WORKING_DIR, f"{mode}_calibration_cache.npy")

        if load_cached_data and os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file, allow_pickle=True).item()
                self.sample_to_calib = cached_data["sample_to_calib"]
                self.sample_to_ego = cached_data["sample_to_ego"]
                self.calib_data = cached_data["calib_data"]
                self.ego_data = cached_data["ego_data"]
            except Exception:
                self._parse_jsons(cache_file)
        else:
            self._parse_jsons(cache_file)

    def _parse_jsons(self, cache_file):
        # Load JSON tables
        with open(os.path.join(self.json_dir, "sample_data.json"), "r") as f:
            sample_data = json.load(f)
        with open(os.path.join(self.json_dir, "calibrated_sensor.json"), "r") as f:
            calibrated_sensor = json.load(f)
        with open(os.path.join(self.json_dir, "ego_pose.json"), "r") as f:
            ego_pose = json.load(f)

        # Build lookups
        self.calib_data = {}
        for item in calibrated_sensor:
            self.calib_data[item["token"]] = {
                "translation": np.array(item["translation"]),
                "rotation": np.array(item["rotation"]),  # w, x, y, z
            }

        self.ego_data = {}
        for item in ego_pose:
            self.ego_data[item["token"]] = {
                "translation": np.array(item["translation"]),
                "rotation": np.array(item["rotation"]),
            }

        self.sample_to_calib = {}
        self.sample_to_ego = {}

        # Map filename basename to tokens
        filename_to_tokens = {}
        for item in sample_data:
            if item["filename"].endswith(".bin"):
                fname = os.path.basename(item["filename"])
                filename_to_tokens[fname] = (
                    item["calibrated_sensor_token"],
                    item["ego_pose_token"],
                )

        # Map dataset samples
        for idx, row in self.df.iterrows():
            lidar_path = row["lidar_path"]
            fname = os.path.basename(lidar_path)
            if fname in filename_to_tokens:
                c_tok, e_tok = filename_to_tokens[fname]
                self.sample_to_calib[row["sample_token"]] = c_tok
                self.sample_to_ego[row["sample_token"]] = e_tok

        # Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(
            cache_file,
            {
                "sample_to_calib": self.sample_to_calib,
                "sample_to_ego": self.sample_to_ego,
                "calib_data": self.calib_data,
                "ego_data": self.ego_data,
            },
        )

    def get_transformation_matrices(self, sample_token):
        calib_token = self.sample_to_calib.get(sample_token)
        ego_token = self.sample_to_ego.get(sample_token)

        if calib_token is None or ego_token is None:
            return np.eye(4), np.eye(4)

        calib = self.calib_data[calib_token]
        ego = self.ego_data[ego_token]

        # Sensor -> Ego
        R_calib = quaternion_to_matrix(calib["rotation"])
        T_calib = calib["translation"]
        M_calib = np.eye(4)
        M_calib[:3, :3] = R_calib
        M_calib[:3, 3] = T_calib

        # Ego -> Global
        R_ego = quaternion_to_matrix(ego["rotation"])
        T_ego = ego["translation"]
        M_ego = np.eye(4)
        M_ego[:3, :3] = R_ego
        M_ego[:3, 3] = T_ego

        # Inverses
        M_global_to_ego = np.linalg.inv(M_ego)
        M_ego_to_sensor = np.linalg.inv(M_calib)

        return M_global_to_ego, M_ego_to_sensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_token = row["sample_token"]
        lidar_path = os.path.join(INPUT_DIR, row["lidar_path"])

        # Load Point Cloud
        if os.path.exists(lidar_path):
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            points = points[:, :4]  # x, y, z, intensity
        else:
            # Fallback for missing files (should not happen in valid data)
            points = np.zeros((1, 4), dtype=np.float32)

        # Get Transformations
        M_global_to_ego, M_ego_to_sensor = self.get_transformation_matrices(
            sample_token
        )

        gt_boxes = []
        if "label" in row and pd.notna(row["label"]):
            label_str = str(row["label"]).split()
            stride = 8
            num_objs = len(label_str) // stride

            for i in range(num_objs):
                base = i * stride
                # Global coordinates
                cx, cy, cz = (
                    float(label_str[base]),
                    float(label_str[base + 1]),
                    float(label_str[base + 2]),
                )
                w, l, h = (
                    float(label_str[base + 3]),
                    float(label_str[base + 4]),
                    float(label_str[base + 5]),
                )
                yaw = float(label_str[base + 6])
                cls_name = label_str[base + 7]

                if cls_name not in self.class_map:
                    continue

                cls_id = self.class_map[cls_name]

                # Transform Center: Global -> Ego -> Sensor
                center_global = np.array([cx, cy, cz, 1.0])
                center_ego = M_global_to_ego @ center_global
                center_sensor = M_ego_to_sensor @ center_ego

                # Transform Yaw
                R_box_global = np.array(
                    [
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1],
                    ]
                )

                R_global_to_ego_3x3 = M_global_to_ego[:3, :3]
                R_ego_to_sensor_3x3 = M_ego_to_sensor[:3, :3]
                R_box_sensor = R_ego_to_sensor_3x3 @ R_global_to_ego_3x3 @ R_box_global
                new_yaw = np.arctan2(R_box_sensor[1, 0], R_box_sensor[0, 0])

                gt_boxes.append(
                    [
                        center_sensor[0],
                        center_sensor[1],
                        center_sensor[2],
                        w,
                        l,
                        h,
                        new_yaw,
                        cls_id,
                    ]
                )

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.zeros((0, 8), dtype=np.float32)
        )

        # Augmentation (Train only)
        if self.mode == "train":
            points, gt_boxes = self._augment(points, gt_boxes)

        # Filter Points to Range
        mask = (
            (points[:, 0] >= POINT_CLOUD_RANGE[0])
            & (points[:, 0] <= POINT_CLOUD_RANGE[3])
            & (points[:, 1] >= POINT_CLOUD_RANGE[1])
            & (points[:, 1] <= POINT_CLOUD_RANGE[4])
            & (points[:, 2] >= POINT_CLOUD_RANGE[2])
            & (points[:, 2] <= POINT_CLOUD_RANGE[5])
        )
        points = points[mask]

        # Filter Boxes to Range
        if len(gt_boxes) > 0:
            box_mask = (
                (gt_boxes[:, 0] >= POINT_CLOUD_RANGE[0])
                & (gt_boxes[:, 0] <= POINT_CLOUD_RANGE[3])
                & (gt_boxes[:, 1] >= POINT_CLOUD_RANGE[1])
                & (gt_boxes[:, 1] <= POINT_CLOUD_RANGE[4])
            )
            gt_boxes = gt_boxes[box_mask]

        # Generate Targets
        target_dict = self._generate_targets(gt_boxes)

        return {
            "points": points,
            "gt_boxes": gt_boxes,
            "sample_token": sample_token,
            "transformation_matrix": (M_global_to_ego, M_ego_to_sensor),
            **target_dict,
        }

    def _augment(self, points, gt_boxes):
        # Flip X
        if np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]
            if len(gt_boxes) > 0:
                gt_boxes[:, 1] = -gt_boxes[:, 1]
                gt_boxes[:, 6] = -gt_boxes[:, 6]

        # Flip Y
        if np.random.rand() < 0.5:
            points[:, 0] = -points[:, 0]
            if len(gt_boxes) > 0:
                gt_boxes[:, 0] = -gt_boxes[:, 0]
                gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)

        # Rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])

        points[:, :2] = points[:, :2] @ rot_mat.T
        if len(gt_boxes) > 0:
            gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_mat.T
            gt_boxes[:, 6] += angle

        # Scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] *= scale
            gt_boxes[:, 3:6] *= scale

        return points, gt_boxes

    def _generate_targets(self, gt_boxes):
        H_out = GRID_SIZE[1] // DOWN_RATIO
        W_out = GRID_SIZE[0] // DOWN_RATIO
        num_classes = len(CLASS_NAMES)

        heatmap = np.zeros((num_classes, H_out, W_out), dtype=np.float32)
        dim_map = np.zeros((3, H_out, W_out), dtype=np.float32)
        rot_map = np.zeros((2, H_out, W_out), dtype=np.float32)
        reg_map = np.zeros((2, H_out, W_out), dtype=np.float32)
        z_map = np.zeros((1, H_out, W_out), dtype=np.float32)

        indices = []
        masks = []

        for box in gt_boxes:
            x, y, z, w, l, h, yaw, cls_id = box
            cls_id = int(cls_id)

            grid_x = (x - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0] / DOWN_RATIO
            grid_y = (y - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1] / DOWN_RATIO

            gx_int, gy_int = int(grid_x), int(grid_y)

            if 0 <= gx_int < W_out and 0 <= gy_int < H_out:
                w_grid = w / VOXEL_SIZE[0] / DOWN_RATIO
                l_grid = l / VOXEL_SIZE[1] / DOWN_RATIO
                radius = gaussian_radius((l_grid, w_grid), min_overlap=0.7)
                radius = max(0, int(radius))

                draw_gaussian(heatmap[cls_id], (gx_int, gy_int), radius)

                dim_map[:, gy_int, gx_int] = np.log([l, w, h])
                rot_map[:, gy_int, gx_int] = [np.sin(yaw), np.cos(yaw)]
                reg_map[:, gy_int, gx_int] = [grid_x - gx_int, grid_y - gy_int]
                z_map[:, gy_int, gx_int] = z

                indices.append(gy_int * W_out + gx_int)
                masks.append(1)

        max_objs = 200
        num_objs = len(indices)
        indices = np.array(indices + [0] * (max_objs - num_objs), dtype=np.int64)[
            :max_objs
        ]
        masks = np.array(masks + [0] * (max_objs - num_objs), dtype=np.float32)[
            :max_objs
        ]

        return {
            "heatmap": heatmap,
            "dim_map": dim_map,
            "rot_map": rot_map,
            "reg_map": reg_map,
            "z_map": z_map,
            "indices": indices,
            "mask": masks,
        }


def collate_fn(batch):
    points = [torch.from_numpy(b["points"]) for b in batch]
    heatmaps = torch.stack([torch.from_numpy(b["heatmap"]) for b in batch])
    dim_maps = torch.stack([torch.from_numpy(b["dim_map"]) for b in batch])
    rot_maps = torch.stack([torch.from_numpy(b["rot_map"]) for b in batch])
    reg_maps = torch.stack([torch.from_numpy(b["reg_map"]) for b in batch])
    z_maps = torch.stack([torch.from_numpy(b["z_map"]) for b in batch])
    indices = torch.stack([torch.from_numpy(b["indices"]) for b in batch])
    masks = torch.stack([torch.from_numpy(b["mask"]) for b in batch])

    sample_tokens = [b["sample_token"] for b in batch]
    transforms = [b["transformation_matrix"] for b in batch]

    return {
        "points": points,
        "heatmap": heatmaps,
        "dim": dim_maps,
        "rot": rot_maps,
        "reg": reg_maps,
        "z_map": z_maps,
        "ind": indices,
        "mask": masks,
        "sample_tokens": sample_tokens,
        "transforms": transforms,
    }
