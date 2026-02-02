import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import math
import json
from library.config import Config


class NuScenesHelper:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sample_data = self._load_json("sample_data.json")
        self.ego_pose = self._load_map("ego_pose.json")
        self.calibrated_sensor = self._load_map("calibrated_sensor.json")

        # Index sample_token -> lidar_record
        self.sample_to_lidar = {}
        for item in self.sample_data:
            if item["filename"].endswith(".bin"):
                self.sample_to_lidar[item["sample_token"]] = item

    def _load_json(self, name):
        with open(os.path.join(self.root_dir, name), "r") as f:
            return json.load(f)

    def _load_map(self, name):
        data = self._load_json(name)
        return {item["token"]: item for item in data}

    def get_matrix(self, record):
        q = record["rotation"]
        t = np.array(record["translation"], dtype=np.float32)

        # Quaternion to Matrix (w, x, y, z)
        w, x, y, z = q
        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ],
            dtype=np.float32,
        )

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def global_to_sensor(self, box, sample_token):
        # box: [x, y, z, w, l, h, yaw]
        if sample_token not in self.sample_to_lidar:
            return box

        sd_record = self.sample_to_lidar[sample_token]
        ego_record = self.ego_pose[sd_record["ego_pose_token"]]
        cs_record = self.calibrated_sensor[sd_record["calibrated_sensor_token"]]

        # Global -> Ego -> Sensor
        M_ego = self.get_matrix(ego_record)
        M_sensor = self.get_matrix(cs_record)

        # M_global_to_sensor = inv(M_sensor) @ inv(M_ego)
        M_inv = np.linalg.inv(M_ego @ M_sensor)

        # Transform Center
        center = np.array([box[0], box[1], box[2], 1.0], dtype=np.float32)
        center_local = M_inv @ center

        # Transform Yaw
        # Rotate unit vector
        yaw = box[6]
        vec = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0], dtype=np.float32)
        vec_local = M_inv @ vec
        yaw_local = np.arctan2(vec_local[1], vec_local[0])

        return np.array(
            [
                center_local[0],
                center_local[1],
                center_local[2],
                box[3],
                box[4],
                box[5],
                yaw_local,
            ],
            dtype=np.float32,
        )


class GTDatabase:
    """
    Manages a database of ground truth objects for 'Copy-Paste' augmentation.
    Implements the Geometric Unit Test to ensure no empty boxes are sampled.
    """

    def __init__(self, metadata_df, load_cached_data=True):
        self.metadata_df = metadata_df
        # Cite debug_lesson_1: Invalidate cache when changing logic
        self.db_path = os.path.join(
            Config.GT_DATABASE_DIR, "gt_database_sensor.parquet"
        )
        self.points_path = os.path.join(Config.GT_DATABASE_DIR, "gt_points_sensor.bin")

        # Cite debug_lesson_7: Fix data mismatch globally
        self.helper = NuScenesHelper(Config.TRAIN_DATA_ROOT)

        self.database = None
        self.indices = {}

        if (
            load_cached_data
            and os.path.exists(self.db_path)
            and os.path.exists(self.points_path)
        ):
            self.database = pd.read_parquet(self.db_path)
        else:
            self._generate_database()

        if self.database is not None:
            for cls_name in Config.CLASS_NAMES:
                self.indices[cls_name] = self.database[
                    self.database["class_name"] == cls_name
                ].index.values

    def _generate_database(self):
        os.makedirs(Config.GT_DATABASE_DIR, exist_ok=True)
        print("Generating GT Database (this may take a while)...")

        db_infos = []
        current_offset = 0

        # Open binary file for writing point data
        with open(self.points_path, "wb") as f_out:
            for idx, row in self.metadata_df.iterrows():
                lidar_path = os.path.join(Config.INPUT_DIR, row["lidar_path"])
                label_str = row["label"]

                if (
                    pd.isna(label_str)
                    or label_str == ""
                    or not os.path.exists(lidar_path)
                ):
                    continue

                # Load points (N, 4)
                raw_points = np.fromfile(lidar_path, dtype=np.float32)
                if raw_points.size % 5 == 0:
                    points = raw_points.reshape(-1, 5)[:, : Config.NUM_POINT_FEATURES]
                else:
                    points = raw_points.reshape(-1, Config.NUM_POINT_FEATURES)

                # Parse labels
                parts = str(label_str).strip().split()
                if len(parts) % 8 != 0:
                    continue

                num_objs = len(parts) // 8
                for i in range(num_objs):
                    offset = i * 8
                    try:
                        # Parse box: cx, cy, cz, w, l, h, yaw
                        box_global = np.array(
                            [float(parts[offset + j]) for j in range(7)],
                            dtype=np.float32,
                        )
                        cls_name = parts[offset + 7]

                        if cls_name not in Config.CLASS_NAMES:
                            continue

                        # Transform to Sensor Frame
                        # Cite debug_lesson_7: Transform GT to sensor frame
                        box = self.helper.global_to_sensor(
                            box_global, row["sample_token"]
                        )

                        # --- Geometric Unit Test & Cropping ---
                        # 1. Rough filter
                        center = box[:3]
                        dims = box[3:6]
                        max_dim = np.max(dims)
                        dist = np.linalg.norm(points[:, :3] - center, axis=1)
                        # Generous radius check
                        cand_mask = dist < (max_dim * 1.5)
                        cand_points = points[cand_mask]

                        if len(cand_points) < Config.MIN_POINTS_IN_GT:
                            continue

                        # 2. Precise Crop (Transform to local coordinates)
                        rel_pos = cand_points[:, :3] - center
                        yaw = box[6]
                        c, s = np.cos(yaw), np.sin(yaw)

                        # Rotate by -yaw to align with box axes
                        local_x = rel_pos[:, 0] * c + rel_pos[:, 1] * s
                        local_y = -rel_pos[:, 0] * s + rel_pos[:, 1] * c
                        local_z = rel_pos[:, 2]

                        # Check bounds
                        in_box_mask = (
                            (np.abs(local_x) <= dims[0] / 2)
                            & (np.abs(local_y) <= dims[1] / 2)
                            & (np.abs(local_z) <= dims[2] / 2)
                        )

                        valid_points = cand_points[in_box_mask]

                        if len(valid_points) >= Config.MIN_POINTS_IN_GT:
                            # Save in LOCAL coordinates
                            points_to_save = np.zeros_like(valid_points)
                            points_to_save[:, 0] = local_x[in_box_mask]
                            points_to_save[:, 1] = local_y[in_box_mask]
                            points_to_save[:, 2] = local_z[in_box_mask]
                            points_to_save[:, 3] = valid_points[:, 3]  # Intensity

                            points_bytes = points_to_save.tobytes()
                            f_out.write(points_bytes)

                            db_infos.append(
                                {
                                    "class_name": cls_name,
                                    "box_w": dims[0],
                                    "box_l": dims[1],
                                    "box_h": dims[2],
                                    "box_z": center[2],
                                    "file_offset": current_offset,
                                    "byte_length": len(points_bytes),
                                    "num_points": len(valid_points),
                                }
                            )
                            current_offset += len(points_bytes)

                    except Exception:
                        continue

        self.database = pd.DataFrame(db_infos)
        self.database.to_parquet(self.db_path)
        print(f"GT Database generated with {len(self.database)} valid samples.")

    def sample(self, existing_boxes):
        """
        Sample objects from DB and return points (sensor coords) and boxes.
        existing_boxes: (N, 7)
        """
        if self.database is None or self.database.empty:
            return [], [], []

        sampled_points = []
        sampled_boxes = []
        sampled_classes = []

        current_boxes_np = (
            np.array(existing_boxes) if len(existing_boxes) > 0 else np.zeros((0, 7))
        )

        with open(self.points_path, "rb") as f:
            for cls_name, count in Config.DB_SAMPLER.items():
                if cls_name not in self.indices or len(self.indices[cls_name]) == 0:
                    continue

                indices = np.random.choice(
                    self.indices[cls_name], size=count, replace=True
                )

                for idx in indices:
                    info = self.database.iloc[idx]

                    # Read points
                    f.seek(info["file_offset"])
                    pts_bytes = f.read(info["byte_length"])
                    local_points = np.frombuffer(pts_bytes, dtype=np.float32).reshape(
                        -1, 4
                    )

                    # Attempt placement (Simple collision check)
                    placed = False
                    for _ in range(10):  # Try 10 positions
                        rx = np.random.uniform(
                            Config.POINT_CLOUD_RANGE[0], Config.POINT_CLOUD_RANGE[3]
                        )
                        ry = np.random.uniform(
                            Config.POINT_CLOUD_RANGE[1], Config.POINT_CLOUD_RANGE[4]
                        )
                        rz = info["box_z"]  # Use original Z
                        ryaw = np.random.uniform(-np.pi, np.pi)

                        cand_box = np.array(
                            [
                                rx,
                                ry,
                                rz,
                                info["box_w"],
                                info["box_l"],
                                info["box_h"],
                                ryaw,
                            ]
                        )

                        # Check collision with existing scene
                        if self._check_collision(cand_box, current_boxes_np):
                            continue

                        # Check collision with already sampled
                        if len(sampled_boxes) > 0 and self._check_collision(
                            cand_box, np.array(sampled_boxes)
                        ):
                            continue

                        # Success
                        c, s = np.cos(ryaw), np.sin(ryaw)
                        x_glob = local_points[:, 0] * c - local_points[:, 1] * s + rx
                        y_glob = local_points[:, 0] * s + local_points[:, 1] * c + ry
                        z_glob = local_points[:, 2] + rz

                        new_pts = np.stack(
                            [x_glob, y_glob, z_glob, local_points[:, 3]], axis=1
                        )

                        sampled_points.append(new_pts)
                        sampled_boxes.append(cand_box)
                        sampled_classes.append(cls_name)
                        placed = True
                        break

        return sampled_points, sampled_boxes, sampled_classes

    def _check_collision(self, box, boxes_array):
        if len(boxes_array) == 0:
            return False
        # BEV Circle approximation
        r_box = np.sqrt(box[3] ** 2 + box[4] ** 2) / 2
        r_arr = np.sqrt(boxes_array[:, 3] ** 2 + boxes_array[:, 4] ** 2) / 2
        dists = np.linalg.norm(boxes_array[:, :2] - box[:2], axis=1)
        return np.any(dists < (r_box + r_arr))


class LidarDataset(Dataset):
    def __init__(self, split="train", subset_size=None, load_cached_data=True):
        self.split = split
        self.is_train = split == "train"

        # Select Metadata
        if split == "train":
            path = Config.TRAIN_METADATA
        elif split == "val":
            path = Config.VAL_METADATA
        else:
            path = Config.TEST_METADATA

        self.metadata = pd.read_csv(path)
        if subset_size:
            self.metadata = self.metadata.iloc[:subset_size]

        # Initialize GT Database for training
        self.gt_db = None
        if self.is_train and Config.USE_GT_AUGMENTATION:
            # We load the full train metadata for DB generation, not just the subset
            full_train_meta = pd.read_csv(Config.TRAIN_METADATA)
            self.gt_db = GTDatabase(full_train_meta, load_cached_data=load_cached_data)

        # Voxelization Parameters
        self.voxel_size = np.array(Config.VOXEL_SIZE, dtype=np.float32)
        self.pc_range = np.array(Config.POINT_CLOUD_RANGE, dtype=np.float32)
        self.grid_size = np.array(Config.GRID_SIZE, dtype=np.int32)
        self.max_voxels = (
            Config.MAX_VOXELS_TRAIN if self.is_train else Config.MAX_VOXELS_TEST
        )

        # Helper for coordinate transformation
        # Cite debug_lesson_7: Fix data mismatch globally
        data_root = (
            Config.TRAIN_DATA_ROOT
            if split in ["train", "val"]
            else Config.TEST_DATA_ROOT
        )
        self.helper = NuScenesHelper(data_root)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["sample_token"]
        lidar_path = os.path.join(Config.INPUT_DIR, row["lidar_path"])

        # 1. Load Point Cloud
        if os.path.exists(lidar_path):
            raw_points = np.fromfile(lidar_path, dtype=np.float32)
            if raw_points.size % 5 == 0:
                points = raw_points.reshape(-1, 5)[:, : Config.NUM_POINT_FEATURES]
            else:
                points = raw_points.reshape(-1, Config.NUM_POINT_FEATURES)
        else:
            # Fallback for missing files (should not happen based on metadata check)
            points = np.zeros((0, Config.NUM_POINT_FEATURES), dtype=np.float32)

        # 2. Load Labels
        gt_boxes = []
        gt_names = []
        if "label" in row and pd.notna(row["label"]):
            parts = str(row["label"]).strip().split()
            if len(parts) % 8 == 0:
                num_objs = len(parts) // 8
                for i in range(num_objs):
                    off = i * 8
                    box_global = [float(parts[off + j]) for j in range(7)]
                    name = parts[off + 7]
                    if name in Config.CLASS_NAMES:
                        # Transform to Sensor Frame
                        box_sensor = self.helper.global_to_sensor(
                            box_global, sample_token
                        )
                        gt_boxes.append(box_sensor)
                        gt_names.append(name)

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.zeros((0, 7), dtype=np.float32)
        )

        # 3. Augmentation (GT Sampling)
        if self.is_train and self.gt_db is not None:
            sampled_pts, sampled_boxes, sampled_cls = self.gt_db.sample(gt_boxes)
            if len(sampled_pts) > 0:
                # Concatenate points
                points = np.vstack([points] + sampled_pts)

                # Update boxes
                new_boxes = np.array(sampled_boxes, dtype=np.float32)
                if len(gt_boxes) > 0:
                    gt_boxes = np.vstack([gt_boxes, new_boxes])
                else:
                    gt_boxes = new_boxes

                gt_names.extend(sampled_cls)

        # 4. Shuffle Points
        np.random.shuffle(points)

        # 5. Voxelization (PointPillars)
        voxels, coordinates, num_points = self.voxelize(points)

        # 6. Prepare Labels
        # Convert class names to indices
        gt_labels = np.zeros(len(gt_names), dtype=np.int64)
        for i, name in enumerate(gt_names):
            gt_labels[i] = Config.CLASS_NAMES.index(name)

        return {
            "voxels": voxels,  # (M, 32, 4)
            "num_points": num_points,  # (M,)
            "coordinates": coordinates,  # (M, 3) -> (z, y, x)
            "gt_boxes": gt_boxes,  # (N, 7)
            "gt_labels": gt_labels,  # (N,)
            "sample_token": sample_token,
        }

    def voxelize(self, points):
        if len(points) == 0:
            return (
                np.zeros((0, Config.MAX_POINTS_PER_VOXEL, 4)),
                np.zeros((0, 3)),
                np.zeros((0,)),
            )

        # Filter points outside range
        mask = (
            (points[:, 0] >= self.pc_range[0])
            & (points[:, 0] < self.pc_range[3])
            & (points[:, 1] >= self.pc_range[1])
            & (points[:, 1] < self.pc_range[4])
            & (points[:, 2] >= self.pc_range[2])
            & (points[:, 2] < self.pc_range[5])
        )
        points = points[mask]

        if len(points) == 0:
            return (
                np.zeros((0, Config.MAX_POINTS_PER_VOXEL, 4)),
                np.zeros((0, 3)),
                np.zeros((0,)),
            )

        # Compute voxel indices
        # coor = (val - min) / size
        voxel_coords = ((points[:, :3] - self.pc_range[:3]) / self.voxel_size).astype(
            np.int32
        )

        # Cite debug_lesson_9: Clamp Discretized Indices to Grid Bounds
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, self.grid_size[0] - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, self.grid_size[1] - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, self.grid_size[2] - 1)

        # Convert to (z, y, x) for standard sparse format
        # voxel_coords is (N, 3) -> (x_idx, y_idx, z_idx)
        # We want (z_idx, y_idx, x_idx)
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Group points using unique coordinates
        # We use a trick to make rows unique hashable or use lexsort
        # Lexsort is stable. Sort by z, y, x
        keys = (
            voxel_coords[:, 0] * (self.grid_size[1] * self.grid_size[0])
            + voxel_coords[:, 1] * self.grid_size[0]
            + voxel_coords[:, 2]
        )

        sorted_ind = np.argsort(keys)
        points = points[sorted_ind]
        voxel_coords = voxel_coords[sorted_ind]
        keys = keys[sorted_ind]

        # Find unique voxels
        unique_keys, unique_indices, unique_counts = np.unique(
            keys, return_index=True, return_counts=True
        )

        # Limit max voxels
        num_voxels = len(unique_keys)
        if num_voxels > self.max_voxels:
            # Shuffle indices to randomly drop voxels if too many (or just take first N)
            # Taking first N is biased if sorted. Random choice is better.
            choice = np.random.choice(num_voxels, self.max_voxels, replace=False)
            unique_keys = unique_keys[choice]
            unique_indices = unique_indices[choice]
            unique_counts = unique_counts[choice]
            num_voxels = self.max_voxels

        # Prepare output tensors
        voxels = np.zeros(
            (num_voxels, Config.MAX_POINTS_PER_VOXEL, points.shape[1]), dtype=np.float32
        )
        coordinates = voxel_coords[unique_indices]  # (M, 3)
        num_points_per_voxel = np.zeros((num_voxels,), dtype=np.int32)

        # Scatter points into voxels
        # Since we sorted, points for a voxel are contiguous.
        # But we might have subsampled voxels, so we can't just slice.
        # We need to iterate or use advanced indexing.
        # Fast approximate method:

        for i in range(num_voxels):
            start = unique_indices[i]
            count = unique_counts[i]
            pts = points[start : start + count]

            if count > Config.MAX_POINTS_PER_VOXEL:
                # Subsample points within voxel
                choice = np.random.choice(
                    count, Config.MAX_POINTS_PER_VOXEL, replace=False
                )
                pts = pts[choice]
                count = Config.MAX_POINTS_PER_VOXEL

            voxels[i, :count, :] = pts
            num_points_per_voxel[i] = count

        return voxels, coordinates, num_points_per_voxel


def collate_fn(batch):
    """
    Collate function to stack voxels and add batch indices to coordinates.
    """
    voxel_list = []
    coords_list = []
    num_points_list = []
    gt_boxes_list = []
    gt_labels_list = []
    tokens = []

    for i, sample in enumerate(batch):
        voxel_list.append(torch.from_numpy(sample["voxels"]))
        num_points_list.append(torch.from_numpy(sample["num_points"]))

        # Add batch index to coordinates: (z, y, x) -> (batch_idx, z, y, x)
        coors = torch.from_numpy(sample["coordinates"])
        batch_idx = torch.full((coors.shape[0], 1), i, dtype=torch.int32)
        coords_list.append(torch.cat([batch_idx, coors], dim=1))

        gt_boxes_list.append(torch.from_numpy(sample["gt_boxes"]))
        gt_labels_list.append(torch.from_numpy(sample["gt_labels"]))
        tokens.append(sample["sample_token"])

    return {
        "voxels": torch.cat(voxel_list, dim=0),
        "num_points": torch.cat(num_points_list, dim=0),
        "coordinates": torch.cat(coords_list, dim=0),
        "gt_boxes": gt_boxes_list,  # List of tensors (variable length)
        "gt_labels": gt_labels_list,
        "sample_tokens": tokens,
    }
