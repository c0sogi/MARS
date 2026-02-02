import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import DataConfig, VoxelConfig, ModelConfig, TrainConfig
from library.utils import (
    create_voxel_grid,
    draw_umich_gaussian,
    gaussian_radius,
    BoxUtils,
)


class CalibrationSystem:
    """
    Handles the creation and retrieval of Global -> Sensor transformation matrices.
    """

    def __init__(self, data_root, cache_dir, split):
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.split = split
        # Determine source directory based on split
        self.source_dir = os.path.join(
            data_root, "test_data" if split == "test" else "train_data"
        )
        self.cache_file = os.path.join(cache_dir, f"calibration_lookup_{split}.parquet")

        self.lookup = self._get_lookup()

    def _load_json_df(self, filename):
        path = os.path.join(self.source_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def _compute_matrix(self, row):
        # Quaternion to Rotation Matrix
        # row keys: rotation (list), translation (list)
        # Ego pose
        r_ego = BoxUtils.quaternion_to_matrix(np.array(row["ego_rotation"]))
        t_ego = np.array(row["ego_translation"]).reshape(3, 1)
        tf_ego = np.eye(4)
        tf_ego[:3, :3] = r_ego
        tf_ego[:3, 3] = t_ego.flatten()

        # Sensor pose (calibrated sensor)
        r_sens = BoxUtils.quaternion_to_matrix(np.array(row["sensor_rotation"]))
        t_sens = np.array(row["sensor_translation"]).reshape(3, 1)
        tf_sens = np.eye(4)
        tf_sens[:3, :3] = r_sens
        tf_sens[:3, 3] = t_sens.flatten()

        # Global -> Sensor = (Sensor -> Ego * Ego -> Global)^-1
        # Global -> Sensor = (Ego -> Global)^-1 * (Sensor -> Ego)^-1 ... wait
        # Standard: P_global = T_ego_global * T_sens_ego * P_sens
        # P_sens = T_sens_ego^-1 * T_ego_global^-1 * P_global

        # T_ego_global is tf_ego
        # T_sens_ego is tf_sens

        # Global -> Sensor = inv(tf_sens) @ inv(tf_ego)
        # Or more simply: inv(tf_ego @ tf_sens)

        # Let's verify chain: Sensor -> Ego -> Global
        # T_total = T_ego @ T_sens
        T_total = tf_ego @ tf_sens
        T_inv = np.linalg.inv(T_total)

        return T_inv.flatten()

    def _build_cache(self):
        # Load necessary tables
        df_sd = self._load_json_df("sample_data.json")
        df_ep = self._load_json_df("ego_pose.json")
        df_cs = self._load_json_df("calibrated_sensor.json")

        # Filter for LIDAR_TOP
        # We need to find the channel name. Assuming 'LIDAR_TOP' based on standard NuScenes.
        # If 'channel' column exists, use it.
        if "channel" not in df_cs.columns:
            # Fallback: usually token map or we assume the link is correct in sample_data
            pass

        # Join tables
        # sample_data -> calibrated_sensor (to get channel)
        df_sd = df_sd.merge(
            df_cs[["token", "channel", "translation", "rotation"]],
            left_on="calibrated_sensor_token",
            right_on="token",
            suffixes=("", "_cs"),
        )

        # Filter for LIDAR_TOP
        df_sd = df_sd[df_sd["channel"] == "LIDAR_TOP"]

        # Join with Ego Pose
        df_sd = df_sd.merge(
            df_ep[["token", "translation", "rotation"]],
            left_on="ego_pose_token",
            right_on="token",
            suffixes=("_sens", "_ego"),
        )

        # We only need rows that have a sample_token (key frames)
        df_sd = df_sd[df_sd["sample_token"].notna() & (df_sd["sample_token"] != "")]

        # Rename cols for clarity
        df_sd = df_sd.rename(
            columns={
                "translation_sens": "sensor_translation",
                "rotation_sens": "sensor_rotation",
                "translation_ego": "ego_translation",
                "rotation_ego": "ego_rotation",
            }
        )

        # Compute matrices
        # This can be slow, but it's done once.
        matrices = []
        tokens = []

        for _, row in df_sd.iterrows():
            mat = self._compute_matrix(row)
            matrices.append(mat)
            tokens.append(row["sample_token"])

        df_lookup = pd.DataFrame({"sample_token": tokens, "matrix": matrices})

        # Deduplicate (just in case)
        df_lookup = df_lookup.drop_duplicates(subset=["sample_token"])

        return df_lookup

    def _get_lookup(self):
        if os.path.exists(self.cache_file):
            return pd.read_parquet(self.cache_file)
        else:
            df = self._build_cache()
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            df.to_parquet(self.cache_file)
            return df

    def get_matrix(self, sample_token):
        # Return 4x4 numpy array
        row = self.lookup[self.lookup["sample_token"] == sample_token]
        if row.empty:
            return np.eye(4)  # Fallback
        mat_flat = row.iloc[0]["matrix"]
        return np.array(mat_flat).reshape(4, 4)


class LidarDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        split,
        enable_augmentation=False,
        has_targets=True,
        subset_size=None,
    ):
        self.metadata_path = metadata_path
        self.split = split
        self.enable_augmentation = enable_augmentation
        self.has_targets = has_targets

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Parse JSON columns
        self.df["file_paths"] = self.df["file_paths"].apply(json.loads)
        self.df["annotations"] = self.df["annotations"].apply(json.loads)

        # Debug subset
        if subset_size is not None:
            self.df = self.df.iloc[:subset_size]

        # Initialize Calibration System
        self.calib_system = CalibrationSystem(
            DataConfig.data_root,
            DataConfig.cache_dir,
            split if split in ["train", "val"] else "test",  # val uses train_data
        )

        # Configs
        self.voxel_config = VoxelConfig()
        self.model_config = ModelConfig()
        self.train_config = TrainConfig()

        self.grid_size = self.voxel_config.grid_size  # [W, H, D]
        self.out_size_factor = self.train_config.out_size_factor

        # Pre-calculate output grid size
        self.feature_map_size = [
            self.grid_size[0] // self.out_size_factor,
            self.grid_size[1] // self.out_size_factor,
        ]

    def __len__(self):
        return len(self.df)

    def _load_points(self, paths_dict):
        # Find LIDAR file
        lidar_path = paths_dict.get("LIDAR_TOP")
        if lidar_path is None:
            # Fallback: search for any .bin file
            for k, v in paths_dict.items():
                if v.endswith(".bin"):
                    lidar_path = v
                    break

        if lidar_path is None:
            raise FileNotFoundError("No LiDAR file found in paths")

        full_path = os.path.join(DataConfig.data_root, lidar_path)

        # Load binary
        points = np.fromfile(full_path, dtype=np.float32)

        # Reshape
        if points.shape[0] % 5 == 0:
            points = points.reshape(-1, 5)
        elif points.shape[0] % 4 == 0:
            points = points.reshape(-1, 4)
        else:
            # Fallback for weird shapes, truncate to multiple of 4
            n = (points.shape[0] // 4) * 4
            points = points[:n].reshape(-1, 4)

        # Take first 4 columns: x, y, z, intensity
        return points[:, :4]

    def _transform_annotations(self, annotations, matrix):
        if not annotations:
            return np.zeros((0, 7), dtype=np.float32), []

        boxes = []
        classes = []

        for ann in annotations:
            # Center (Global)
            center = np.array([ann["center_x"], ann["center_y"], ann["center_z"], 1.0])

            # Transform Center
            center_sens = matrix @ center

            # Transform Yaw
            # Vector pointing in yaw direction
            yaw = ann["yaw"]
            vec = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
            vec_sens = matrix @ vec
            yaw_sens = np.arctan2(vec_sens[1], vec_sens[0])

            # Dimensions (w, l, h)
            # NuScenes: w=y-dim, l=x-dim in object frame?
            # Standard: dx, dy, dz.
            # Ann: width, length, height.
            # We keep them as is.

            box = [
                center_sens[0],
                center_sens[1],
                center_sens[2],
                ann["width"],
                ann["length"],
                ann["height"],
                yaw_sens,
            ]
            boxes.append(box)
            classes.append(ann["class_name"])

        return np.array(boxes, dtype=np.float32), classes

    def _augment(self, points, boxes):
        # Random Flip X (Mirror across Y axis)
        if np.random.random() < 0.5:
            points[:, 0] = -points[:, 0]
            if len(boxes) > 0:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = np.pi - boxes[:, 6]  # yaw flip

        # Random Flip Y (Mirror across X axis)
        if np.random.random() < 0.5:
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]

        # Global Rotation
        rot_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s], [s, c]])

        points[:, :2] = points[:, :2] @ rot_mat.T
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] @ rot_mat.T
            boxes[:, 6] += rot_angle

        # Global Scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :6] *= scale  # Scale center and dims

        return points, boxes

    def _generate_targets(self, boxes, classes):
        # Initialize targets
        hm = np.zeros(
            (
                self.model_config.num_classes,
                self.feature_map_size[1],
                self.feature_map_size[0],
            ),
            dtype=np.float32,
        )

        # Regression maps
        # center_z (1), dim (3), rot (2), reg (2)
        target_z = np.zeros(
            (1, self.feature_map_size[1], self.feature_map_size[0]), dtype=np.float32
        )
        target_dim = np.zeros(
            (3, self.feature_map_size[1], self.feature_map_size[0]), dtype=np.float32
        )
        target_rot = np.zeros(
            (2, self.feature_map_size[1], self.feature_map_size[0]), dtype=np.float32
        )
        target_reg = np.zeros(
            (2, self.feature_map_size[1], self.feature_map_size[0]), dtype=np.float32
        )

        mask_reg = np.zeros(
            (1, self.feature_map_size[1], self.feature_map_size[0]), dtype=np.float32
        )

        if len(boxes) == 0:
            return {
                "hm": hm,
                "center_z": target_z,
                "dim": target_dim,
                "rot": target_rot,
                "reg": target_reg,
                "mask_reg": mask_reg,
            }

        # Grid params
        pc_range = self.voxel_config.point_cloud_range
        voxel_size = self.voxel_config.voxel_size

        # Filter boxes outside range
        mask = (
            (boxes[:, 0] >= pc_range[0])
            & (boxes[:, 0] < pc_range[3])
            & (boxes[:, 1] >= pc_range[1])
            & (boxes[:, 1] < pc_range[4])
        )
        boxes = boxes[mask]

        # We need to filter classes list too, but it's a list.
        # Re-indexing list is slow, let's just zip and iterate if efficient
        # Or just use the index mask
        valid_indices = np.where(mask)[0]

        for idx in valid_indices:
            box = boxes[idx]
            cls_name = classes[idx]

            if cls_name not in self.model_config.class_names:
                continue

            cls_id = self.model_config.class_names.index(cls_name)

            # Project center to grid
            # coord = (pos - min) / size
            x = (box[0] - pc_range[0]) / voxel_size[0]
            y = (box[1] - pc_range[1]) / voxel_size[1]

            # Downsample
            x = x / self.out_size_factor
            y = y / self.out_size_factor

            x_int, y_int = int(x), int(y)

            # Boundary check
            if not (
                0 <= x_int < self.feature_map_size[0]
                and 0 <= y_int < self.feature_map_size[1]
            ):
                continue

            # Gaussian Radius
            # Need w, l in grid units
            # box[3] is width (x-dim?), box[4] is length.
            # Voxel size is 0.2.
            w_grid = box[3] / voxel_size[0] / self.out_size_factor
            l_grid = box[4] / voxel_size[1] / self.out_size_factor

            radius = gaussian_radius(
                l_grid, w_grid, min_overlap=self.train_config.gaussian_overlap
            )
            radius = max(self.train_config.min_radius, int(radius))

            # Draw Gaussian
            draw_umich_gaussian(hm[cls_id], (x_int, y_int), radius)

            # Fill Regression
            target_z[0, y_int, x_int] = box[2]

            # Log dimensions? Standard CenterPoint uses log(dim)
            target_dim[:, y_int, x_int] = np.log(box[3:6] + 1e-6)

            # Rotation: sin, cos
            target_rot[0, y_int, x_int] = np.sin(box[6])
            target_rot[1, y_int, x_int] = np.cos(box[6])

            # Local Offset
            target_reg[0, y_int, x_int] = x - x_int
            target_reg[1, y_int, x_int] = y - y_int

            # Mask
            mask_reg[0, y_int, x_int] = 1

        return {
            "hm": hm,
            "center_z": target_z,
            "dim": target_dim,
            "rot": target_rot,
            "reg": target_reg,
            "mask_reg": mask_reg,
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        token = row["token"]

        # 1. Load Points
        points = self._load_points(row["file_paths"])

        # 2. Load and Transform Annotations
        matrix = self.calib_system.get_matrix(token)
        boxes, classes = self._transform_annotations(row["annotations"], matrix)

        # 3. Augmentation
        if self.enable_augmentation:
            points, boxes = self._augment(points, boxes)

        # 4. Voxelization
        pillar_features, pillar_coords, pillar_num_points = create_voxel_grid(
            points, self.voxel_config
        )

        # 5. Targets
        targets = {}
        if self.has_targets:
            targets = self._generate_targets(boxes, classes)

        return {
            "pillar_features": pillar_features,
            "pillar_coords": pillar_coords,
            "pillar_num_points": pillar_num_points,
            "targets": targets,
            "token": token,
            "matrix": matrix,  # Useful for inference (transform back)
            "gt_boxes": boxes,  # Useful for debug/eval
        }


def collate_fn(batch):
    # batch is list of dicts

    # Stack Pillar Features
    # We need to concatenate them and keep track of batch index

    feature_list = []
    coords_list = []

    batch_size = len(batch)

    # Targets
    target_keys = batch[0]["targets"].keys()
    batched_targets = {k: [] for k in target_keys}

    tokens = []
    matrices = []

    for i, item in enumerate(batch):
        feature_list.append(torch.from_numpy(item["pillar_features"]))

        # Add batch index to coords: (z, y, x) -> (batch_idx, z, y, x)
        coords = torch.from_numpy(item["pillar_coords"])
        batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
        coords_with_batch = torch.cat([batch_idx, coords], dim=1)
        coords_list.append(coords_with_batch)

        tokens.append(item["token"])
        matrices.append(torch.from_numpy(item["matrix"]))

        for k in target_keys:
            batched_targets[k].append(torch.from_numpy(item["targets"][k]))

    # Concatenate
    pillar_features = torch.cat(feature_list, dim=0)
    pillar_coords = torch.cat(coords_list, dim=0)

    # Stack targets
    final_targets = {}
    for k in target_keys:
        final_targets[k] = torch.stack(batched_targets[k], dim=0)

    return {
        "pillar_features": pillar_features,
        "pillar_coords": pillar_coords,
        "batch_size": batch_size,
        "targets": final_targets,
        "tokens": tokens,
        "matrices": torch.stack(matrices),
    }
