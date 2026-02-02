import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_json_table, world_to_sensor

# Constants
OUTPUT_STRIDE = 4  # Downsampling factor for the detection head (512 -> 128)


class CalibrationLookup:
    """
    Helper class to build and cache a lookup table for sensor calibration data.
    Maps sample_token -> (lidar_path, ego_pose, calibrated_sensor).
    """

    def __init__(self, root_dir, split, cache_dir):
        self.root_dir = root_dir
        self.split = split
        self.cache_path = os.path.join(cache_dir, "calibration_lookup.parquet")
        self.lookup_df = self._load_or_build_lookup()

    def _load_or_build_lookup(self):
        # 1. Try to load from cache
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_parquet(self.cache_path)
                # Check if it contains data for the requested split
                # We assume the cache contains all processed data.
                # If specific split data is missing from a shared cache, we might need to rebuild.
                # For simplicity, we assume separate caches or a single build step.
                # Here we check if we need to append data or if it's sufficient.
                # To be safe and simple: if cache exists, we use it.
                # If the split requires data not in cache (e.g. switching from train to test),
                # we might need a more complex logic.
                # Given the task, we'll build a combined cache or separate based on existence.
                # Let's check if the cache covers the current split's folder.
                if self.split in ["train", "val"]:
                    if "train_data" in df["source_folder"].values:
                        return df
                elif self.split == "test":
                    if "test_data" in df["source_folder"].values:
                        return df
            except Exception:
                pass  # Corrupt or error, rebuild

        # 2. Build from scratch
        return self._build_lookup()

    def _build_lookup(self):
        # We need to process both train_data and test_data to ensure the cache is complete
        # or just process the one needed. Let's process both if possible to make a unified cache.

        dfs = []

        for folder_name in ["train_data", "test_data"]:
            data_path = os.path.join(self.root_dir, folder_name)
            if not os.path.exists(data_path):
                continue

            # Load tables
            try:
                sample_df = load_json_table(os.path.join(data_path, "sample.json"))
                sample_data_df = load_json_table(
                    os.path.join(data_path, "sample_data.json")
                )
                ego_pose_df = load_json_table(os.path.join(data_path, "ego_pose.json"))
                calib_sensor_df = load_json_table(
                    os.path.join(data_path, "calibrated_sensor.json")
                )
            except FileNotFoundError:
                continue

            # Filter for LIDAR_TOP
            # Map channel from calibrated_sensor if not present in sample_data
            if "channel" not in sample_data_df.columns:
                channel_col = (
                    "channel" if "channel" in calib_sensor_df.columns else "token"
                )
                sensor_map = dict(
                    zip(calib_sensor_df["token"], calib_sensor_df[channel_col])
                )
                sample_data_df["channel"] = sample_data_df[
                    "calibrated_sensor_token"
                ].map(sensor_map)

            lidar_data = sample_data_df[sample_data_df["channel"] == "LIDAR_TOP"].copy()

            # Merge with Ego Pose
            lidar_data = lidar_data.merge(
                ego_pose_df,
                left_on="ego_pose_token",
                right_on="token",
                suffixes=("", "_ego"),
            )

            # Merge with Calibrated Sensor
            lidar_data = lidar_data.merge(
                calib_sensor_df,
                left_on="calibrated_sensor_token",
                right_on="token",
                suffixes=("", "_calib"),
            )

            # Select and Rename Columns
            # We need: sample_token, translation (ego), rotation (ego), translation (calib), rotation (calib)
            cols = {
                "sample_token": "sample_token",
                "filename": "lidar_path",
                "translation": "ego_translation",
                "rotation": "ego_rotation",
                "translation_calib": "sensor_translation",
                "rotation_calib": "sensor_rotation",
            }

            # Handle column name collisions from merges
            # ego_pose has 'translation', 'rotation'
            # calibrated_sensor has 'translation', 'rotation'
            # The merge suffixes handle this.
            # suffixes=("", "_ego") -> ego_pose cols get "_ego" if collision?
            # Wait, left is sample_data (no trans/rot), right is ego_pose (has trans/rot).
            # So ego_pose cols are 'translation', 'rotation'.
            # Next merge: left (has trans/rot), right (calib, has trans/rot).
            # Suffixes ("", "_calib"). Left keeps 'translation', right gets 'translation_calib'.

            # Actually, sample_data DOES NOT have translation/rotation.
            # 1. sample_data + ego_pose -> 'translation', 'rotation' (from ego)
            # 2. + calibrated_sensor -> 'translation' (ego), 'rotation' (ego), 'translation_calib', 'rotation_calib'

            subset = lidar_data[
                [
                    "sample_token",
                    "filename",
                    "translation",
                    "rotation",
                    "translation_calib",
                    "rotation_calib",
                ]
            ].rename(columns=cols)

            subset["source_folder"] = folder_name
            dfs.append(subset)

        if not dfs:
            # Return empty structure if nothing found
            return pd.DataFrame(
                columns=[
                    "sample_token",
                    "lidar_path",
                    "ego_translation",
                    "ego_rotation",
                    "sensor_translation",
                    "sensor_rotation",
                    "source_folder",
                ]
            )

        final_df = pd.concat(dfs, ignore_index=True)

        # Cache it
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        final_df.to_parquet(self.cache_path)

        return final_df

    def get_calibration(self, sample_token):
        row = self.lookup_df[self.lookup_df["sample_token"] == sample_token]
        if row.empty:
            return None

        # Extract data
        rec = row.iloc[0]

        ego_pose = {
            "translation": (
                rec["ego_translation"].tolist()
                if isinstance(rec["ego_translation"], np.ndarray)
                else rec["ego_translation"]
            ),
            "rotation": (
                rec["ego_rotation"].tolist()
                if isinstance(rec["ego_rotation"], np.ndarray)
                else rec["ego_rotation"]
            ),
        }

        calib_sensor = {
            "translation": (
                rec["sensor_translation"].tolist()
                if isinstance(rec["sensor_translation"], np.ndarray)
                else rec["sensor_translation"]
            ),
            "rotation": (
                rec["sensor_rotation"].tolist()
                if isinstance(rec["sensor_rotation"], np.ndarray)
                else rec["sensor_rotation"]
            ),
        }

        return ego_pose, calib_sensor


class LidarDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached calibration data.
        """
        self.split = split
        self.load_cached_data = load_cached_data

        # Load Metadata
        if split == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            meta_path = Config.VAL_METADATA_PATH
        else:
            meta_path = Config.TEST_METADATA_PATH

        self.metadata = pd.read_csv(meta_path)

        # Parse JSON columns in metadata
        self.metadata["file_paths"] = self.metadata["file_paths"].apply(json.loads)
        self.metadata["annotations"] = self.metadata["annotations"].apply(json.loads)

        # Initialize Calibration Lookup
        self.calib_lookup = CalibrationLookup(
            Config.INPUT_DIR, split, Config.WORKING_DIR
        )

        # Grid Parameters
        self.bev_h = Config.BEV_HEIGHT
        self.bev_w = Config.BEV_WIDTH
        self.voxel_size = np.array(Config.VOXEL_SIZE)
        self.pc_range = np.array(Config.PC_RANGE)

        # Output Grid for Targets
        self.out_h = self.bev_h // OUTPUT_STRIDE
        self.out_w = self.bev_w // OUTPUT_STRIDE

        # Anchors
        self.anchors = np.array(Config.ANCHORS)  # (N_anchors, 2) [w, l]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        token = row["token"]

        # 1. Get Calibration & Paths
        calib_info = self.calib_lookup.get_calibration(token)
        if calib_info is None:
            # Fallback if lookup fails (shouldn't happen if data is consistent)
            # Return zero tensors
            return self._empty_sample(token)

        ego_pose, calib_sensor = calib_info

        # Get Lidar Path from metadata (more reliable for relative path)
        # The metadata 'file_paths' dict usually contains keys like 'LIDAR_TOP'
        paths = row["file_paths"]
        lidar_rel_path = paths.get("LIDAR_TOP")

        if not lidar_rel_path:
            return self._empty_sample(token)

        lidar_path = os.path.join(Config.INPUT_DIR, lidar_rel_path)

        # 2. Load LiDAR Points
        points = self._load_lidar(lidar_path)

        # 3. Rasterize
        bev_map = self._rasterize_lidar(points)  # (3, H, W)

        # 4. Build Targets (if train/val)
        targets = None
        if self.split in ["train", "val"]:
            annotations = row["annotations"]
            targets = self._build_targets(annotations, ego_pose, calib_sensor)

        # Return
        # For DataLoader, we return tensors
        bev_tensor = torch.from_numpy(bev_map).float()

        if targets is not None:
            targets_tensor = torch.from_numpy(targets).float()
            return bev_tensor, targets_tensor, token
        else:
            return bev_tensor, token

    def _empty_sample(self, token):
        bev = torch.zeros(
            (Config.IN_CHANNELS, self.bev_h, self.bev_w), dtype=torch.float32
        )
        if self.split in ["train", "val"]:
            # Target shape: (Num_Anchors, Out_H, Out_W, Attributes)
            # Attributes: [valid, x, y, w, l, z, h, sin, cos, class] -> 10
            t_shape = (len(self.anchors), self.out_h, self.out_w, 10)
            targets = torch.zeros(t_shape, dtype=torch.float32)
            return bev, targets, token
        else:
            return bev, token

    def _load_lidar(self, path):
        if not os.path.exists(path):
            return np.zeros((0, 4), dtype=np.float32)

        try:
            # Standard NuScenes/Lyft format: float32, usually 5 fields (x,y,z,intensity,ring)
            # or 4 fields. We reshape based on size.
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size % 5 == 0:
                points = raw.reshape(-1, 5)
            elif raw.size % 4 == 0:
                points = raw.reshape(-1, 4)
            else:
                points = raw.reshape(-1, 3)  # minimal fallback
            return points
        except Exception:
            return np.zeros((0, 4), dtype=np.float32)

    def _rasterize_lidar(self, points):
        """
        Convert points (N, C) to BEV (3, H, W).
        Channels: Max Height, Mean Intensity, Density
        """
        if points.shape[0] == 0:
            return np.zeros((3, self.bev_h, self.bev_w), dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros_like(z)

        # Filter out of range
        mask = (
            (x >= self.pc_range[0])
            & (x < self.pc_range[3])
            & (y >= self.pc_range[1])
            & (y < self.pc_range[4])
            & (z >= self.pc_range[2])
            & (z < self.pc_range[5])
        )

        x = x[mask]
        y = y[mask]
        z = z[mask]
        intensity = intensity[mask]

        if x.shape[0] == 0:
            return np.zeros((3, self.bev_h, self.bev_w), dtype=np.float32)

        # Discretize
        x_idx = ((x - self.pc_range[0]) / self.voxel_size[0]).astype(np.int32)
        y_idx = ((y - self.pc_range[1]) / self.voxel_size[1]).astype(np.int32)

        # Clip just in case
        x_idx = np.clip(x_idx, 0, self.bev_w - 1)
        y_idx = np.clip(y_idx, 0, self.bev_h - 1)

        # Flatten indices for vectorization
        flat_idx = y_idx * self.bev_w + x_idx

        # Sort by index to use reduceat
        sort_order = np.argsort(flat_idx)
        flat_idx_sorted = flat_idx[sort_order]
        z_sorted = z[sort_order]
        int_sorted = intensity[sort_order]

        # Find boundaries of unique indices
        unique_indices, unique_counts = np.unique(flat_idx_sorted, return_counts=True)

        # We need the start positions of each unique index in the sorted array
        # np.unique with return_index gives the first occurrence
        _, start_indices = np.unique(flat_idx_sorted, return_index=True)

        # 1. Max Height
        # np.maximum.reduceat requires sorted indices? No, it requires slice boundaries.
        # start_indices are exactly that.
        max_heights = np.maximum.reduceat(z_sorted, start_indices)

        # 2. Mean Intensity
        sum_intensity = np.add.reduceat(int_sorted, start_indices)
        mean_intensity = sum_intensity / unique_counts

        # 3. Density (Normalized Log Count)
        # min(1.0, log(count + 1) / 64)
        density = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))

        # Fill Map
        bev_map = np.zeros((3, self.bev_h * self.bev_w), dtype=np.float32)

        # Initialize height with min value
        bev_map[0, :] = -5.0  # default low height

        bev_map[0, unique_indices] = max_heights
        bev_map[1, unique_indices] = mean_intensity
        bev_map[2, unique_indices] = density

        return bev_map.reshape(3, self.bev_h, self.bev_w)

    def _build_targets(self, annotations, ego_pose, calib_sensor):
        """
        Build YOLO-style targets.
        Output: (Num_Anchors, Out_H, Out_W, 10)
        Channels: [valid, dx, dy, dw, dl, z, h, sin, cos, class_idx]
        """
        num_anchors = len(self.anchors)
        targets = np.zeros((num_anchors, self.out_h, self.out_w, 10), dtype=np.float32)

        if not annotations:
            return targets

        # Parse Annotations
        # center_x, center_y, center_z, width, length, height, yaw, class_name
        gt_boxes = []
        classes = []

        for ann in annotations:
            if ann["class_name"] not in Config.CLASS_MAP:
                continue

            # Global Frame
            center = np.array([ann["center_x"], ann["center_y"], ann["center_z"]])
            dims = np.array([ann["width"], ann["length"], ann["height"]])
            yaw = ann["yaw"]

            # Transform Center to Sensor Frame
            # Note: utils.world_to_sensor expects (N, 3)
            center_sensor = world_to_sensor(
                center.reshape(1, 3), ego_pose, calib_sensor
            )[0]

            # Transform Yaw
            # Global Yaw is rotation around Z.
            # We need to apply the rotation part of the transformation matrices.
            # Simplified: Sensor Yaw = Global Yaw - Ego Yaw - Sensor Yaw offset
            # But full quaternion math is safer.
            # For BEV approximation: rotate the unit vector (cos(yaw), sin(yaw), 0)
            vec = np.array([[np.cos(yaw), np.sin(yaw), 0.0]])
            # Apply rotation only (translation doesn't affect direction)
            # We can use world_to_sensor with 0 translation logic, or just transform 2 points
            p1 = center
            p2 = center + vec[0]
            p1_s = world_to_sensor(p1.reshape(1, 3), ego_pose, calib_sensor)[0]
            p2_s = world_to_sensor(p2.reshape(1, 3), ego_pose, calib_sensor)[0]
            diff = p2_s - p1_s
            yaw_sensor = np.arctan2(diff[1], diff[0])

            gt_boxes.append(np.concatenate([center_sensor, dims, [yaw_sensor]]))
            classes.append(Config.CLASS_MAP[ann["class_name"]])

        if not gt_boxes:
            return targets

        gt_boxes = np.array(gt_boxes)  # (N, 7) [x, y, z, w, l, h, yaw]

        # Grid Resolution for Output
        out_res_x = self.voxel_size[0] * OUTPUT_STRIDE
        out_res_y = self.voxel_size[1] * OUTPUT_STRIDE

        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            cls_idx = classes[i]

            x, y, z, w, l, h, yaw = box

            # Check range
            if not (
                self.pc_range[0] < x < self.pc_range[3]
                and self.pc_range[1] < y < self.pc_range[4]
            ):
                continue

            # Grid Coordinates
            gx = int((x - self.pc_range[0]) / out_res_x)
            gy = int((y - self.pc_range[1]) / out_res_y)

            if not (0 <= gx < self.out_w and 0 <= gy < self.out_h):
                continue

            # Determine best anchor
            # IoU between box (w, l) and anchors (w, l)
            # Assume centers aligned
            # Intersection
            inter_w = np.minimum(w, self.anchors[:, 0])
            inter_l = np.minimum(l, self.anchors[:, 1])
            inter_area = inter_w * inter_l
            union_area = (
                (w * l) + (self.anchors[:, 0] * self.anchors[:, 1]) - inter_area
            )
            iou = inter_area / (union_area + 1e-6)

            best_anchor_idx = np.argmax(iou)

            # If this cell/anchor is already assigned, skip or overwrite?
            # Overwrite is standard.

            # Targets
            # 1. Offsets (sigmoid targets usually, but here linear relative to cell)
            # Center of cell in meters
            cx_cell = self.pc_range[0] + gx * out_res_x + out_res_x / 2
            cy_cell = self.pc_range[1] + gy * out_res_y + out_res_y / 2

            dx = (x - cx_cell) / out_res_x  # Range approx -0.5 to 0.5
            dy = (y - cy_cell) / out_res_y

            # Dimensions (log scale relative to anchor)
            dw = np.log(w / self.anchors[best_anchor_idx, 0])
            dl = np.log(l / self.anchors[best_anchor_idx, 1])

            # Height/Z (direct regression or log h)
            # z is usually bottom or center. Dataset says center.
            # We regress z directly.
            # h is positive, use log.
            dh = np.log(h)

            # Yaw
            sin_y = np.sin(yaw)
            cos_y = np.cos(yaw)

            # Assign
            targets[best_anchor_idx, gy, gx, 0] = 1.0  # Valid
            targets[best_anchor_idx, gy, gx, 1] = dx
            targets[best_anchor_idx, gy, gx, 2] = dy
            targets[best_anchor_idx, gy, gx, 3] = dw
            targets[best_anchor_idx, gy, gx, 4] = dl
            targets[best_anchor_idx, gy, gx, 5] = z
            targets[best_anchor_idx, gy, gx, 6] = dh
            targets[best_anchor_idx, gy, gx, 7] = sin_y
            targets[best_anchor_idx, gy, gx, 8] = cos_y
            targets[best_anchor_idx, gy, gx, 9] = cls_idx

        return targets
