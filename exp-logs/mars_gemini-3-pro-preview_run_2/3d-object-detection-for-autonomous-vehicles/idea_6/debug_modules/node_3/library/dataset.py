import os
import json
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import (
    load_table,
    get_transformation_matrix,
    transform_points,
    load_point_cloud,
    get_yaw_from_quaternion,
    transform_box_to_global,
)


class NuScenesDataset(Dataset):
    def __init__(self, is_train=True, load_cached_data=True):
        """
        Dataset class for Multi-Sweep Pillar-UNet Object Detection.
        """
        self.is_train = is_train
        self.config = Config.get_dataset_config(is_train)

        # 1. Load Metadata
        self.metadata = pd.read_csv(self.config["metadata_path"])

        # Parse JSON columns
        self.metadata["file_paths"] = self.metadata["file_paths"].apply(json.loads)
        self.metadata["annotations"] = self.metadata["annotations"].apply(json.loads)

        # Debug limit
        if self.config["debug_limit"]:
            self.metadata = self.metadata.iloc[: self.config["debug_limit"]]

        # 2. Load NuScenes Tables for Coordinate Transforms
        # We need sample_data, ego_pose, and calibrated_sensor
        data_dir = self.config["data_dir"]
        train_data_dir = os.path.join(data_dir, "train_data")
        test_data_dir = os.path.join(data_dir, "test_data")

        # Determine which folder to look in based on split, but we might need both if mixed
        # However, usually train/val are in train_data, test is in test_data.
        # The metadata tells us the split.

        # To be safe and robust, we load from the directory corresponding to the metadata split
        # If is_train is True, we are in train or val mode -> train_data
        # If is_train is False, we check if it's validation or test.
        # Actually, the Config separates TRAIN_DATA_DIR and TEST_DATA_DIR.

        if "test" in self.config["metadata_path"]:
            base_dir = test_data_dir
        else:
            base_dir = train_data_dir

        self.sample_data = load_table(base_dir, "sample_data", load_cached_data)
        self.ego_pose = load_table(base_dir, "ego_pose", load_cached_data)
        self.calibrated_sensor = load_table(
            base_dir, "calibrated_sensor", load_cached_data
        )

        # Ensure tokens are unique before creating index-based lookups (Cite debug_lesson_3)
        self.sample_data.drop_duplicates(subset="token", inplace=True)
        self.ego_pose.drop_duplicates(subset="token", inplace=True)
        self.calibrated_sensor.drop_duplicates(subset="token", inplace=True)

        # Cite debug_lesson_7: Denormalize Foreign Attributes Before Filtering Relational Data
        # The 'channel' attribute is in calibrated_sensor, not sample_data. We must map it.
        if "channel" in self.calibrated_sensor.columns:
            sensor_map = dict(
                zip(self.calibrated_sensor["token"], self.calibrated_sensor["channel"])
            )
            self.sample_data["channel"] = self.sample_data[
                "calibrated_sensor_token"
            ].map(sensor_map)
        else:
            self.sample_data["channel"] = None

        # 3. Create Lookups for O(1) access
        # Map token -> record
        self.sd_lookup = self.sample_data.set_index("token").to_dict("index")
        self.ep_lookup = self.ego_pose.set_index("token").to_dict("index")
        self.cs_lookup = self.calibrated_sensor.set_index("token").to_dict("index")

        # Map sample_token -> LIDAR_TOP sample_data token
        # We filter sample_data for LIDAR_TOP and create a map
        lidar_records = self.sample_data[
            (self.sample_data["channel"] == "LIDAR_TOP")
            | (self.sample_data["filename"].str.contains("LIDAR_TOP", na=False))
        ]
        # Note: Some datasets use 'LIDAR_TOP', others might differ. NuScenes uses LIDAR_TOP.
        # Fallback: if channel column missing or empty, rely on filename or calibrated_sensor

        self.sample_to_lidar_token = dict(
            zip(lidar_records["sample_token"], lidar_records["token"])
        )

        # 4. Pre-calculate Grid Parameters
        self.voxel_size = np.array(self.config["voxel_size"], dtype=np.float32)
        self.pc_range = np.array(self.config["point_cloud_range"], dtype=np.float32)
        self.grid_size = np.array(Config.GRID_SIZE, dtype=np.int32)

        self.class_map = {name: i for i, name in enumerate(self.config["class_names"])}

    def __len__(self):
        return len(self.metadata)

    def get_pose(self, token, lookup):
        rec = lookup[token]
        return np.array(rec["translation"]), np.array(rec["rotation"])

    def get_sweep(self, sample_data_token):
        """
        Loads points for a specific sample_data token and returns points + pose info.
        """
        sd_rec = self.sd_lookup[sample_data_token]

        # Load Points
        # Path resolution: The metadata has resolved paths, but for sweeps we only have the filename
        # from sample_data. We need to prepend the appropriate directory.
        # We assume the directory structure matches the input root.

        filename = sd_rec["filename"]
        # Filename is like 'samples/LIDAR_TOP/...' or 'sweeps/LIDAR_TOP/...'
        # In this dataset structure: 'train_lidar/filename.bin' or 'test_lidar/filename.bin'
        # We need to handle the flat structure provided in the prompt description.
        # The prompt says: "train_lidar/host-a004_lidar1_..."
        # The filename in sample_data might be relative to standard nuScenes root.
        # We need to extract the basename and find it in the flat lidar folders.

        basename = os.path.basename(filename)
        if "test" in self.config["metadata_path"]:
            full_path = os.path.join(Config.INPUT_DIR, "test_lidar", basename)
        else:
            full_path = os.path.join(Config.INPUT_DIR, "train_lidar", basename)

        points = load_point_cloud(full_path)

        # Get Pose Info
        cs_token = sd_rec["calibrated_sensor_token"]
        ep_token = sd_rec["ego_pose_token"]

        cs_trans, cs_rot = self.get_pose(cs_token, self.cs_lookup)
        ep_trans, ep_rot = self.get_pose(ep_token, self.ep_lookup)

        return points, cs_trans, cs_rot, ep_trans, ep_rot, sd_rec["prev"]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["token"]

        # 1. Load Multi-Sweep Point Cloud
        # -------------------------------
        points_list = []

        # Start with current frame
        curr_sd_token = self.sample_to_lidar_token.get(sample_token)

        if curr_sd_token is None:
            # Initialize transformation matrices to None as we lack pose info
            T_global_to_curr_ego = None
            curr_ep_r = None

            # Fallback for robustness: try to find any lidar file in file_paths
            # This happens if sample_data.json is incomplete or mismatch
            # We just load the single frame from metadata file_paths
            paths = row["file_paths"]
            lidar_path = None
            for k, v in paths.items():
                if "LIDAR" in k:
                    lidar_path = os.path.join(Config.INPUT_DIR, v)
                    break
            if lidar_path:
                pts = load_point_cloud(lidar_path)
                # Add time lag 0
                pts = np.hstack([pts[:, :4], np.zeros((pts.shape[0], 1))])
                points_list.append(pts)
            # Cannot do sweeps without tokens
        else:
            # Load Current Frame Pose (Reference Frame)
            # We want everything in Current Frame's EGO coordinates
            # T_global_to_curr_ego = inv(T_curr_ego_to_global)

            # Get current frame info
            sd_rec = self.sd_lookup[curr_sd_token]
            curr_cs_t, curr_cs_r = self.get_pose(
                sd_rec["calibrated_sensor_token"], self.cs_lookup
            )
            curr_ep_t, curr_ep_r = self.get_pose(
                sd_rec["ego_pose_token"], self.ep_lookup
            )

            # Matrix: Sensor -> Ego
            T_curr_sens_to_ego = get_transformation_matrix(curr_cs_t, curr_cs_r)
            # Matrix: Ego -> Global
            T_curr_ego_to_global = get_transformation_matrix(curr_ep_t, curr_ep_r)
            # Matrix: Global -> Ego (Inverse)
            T_global_to_curr_ego = np.linalg.inv(T_curr_ego_to_global)

            # Traverse sweeps
            next_sd_token = curr_sd_token
            for i in range(self.config["max_sweeps"]):
                if next_sd_token == "":
                    break

                try:
                    pts, cs_t, cs_r, ep_t, ep_r, prev_token = self.get_sweep(
                        next_sd_token
                    )
                except Exception:
                    break

                # Transform: Sweep Sensor -> Sweep Ego
                T_sens_to_ego = get_transformation_matrix(cs_t, cs_r)
                pts_ego = transform_points(pts[:, :3], T_sens_to_ego)

                # Transform: Sweep Ego -> Global
                T_ego_to_global = get_transformation_matrix(ep_t, ep_r)
                pts_global = transform_points(pts_ego, T_ego_to_global)

                # Transform: Global -> Current Ego
                pts_curr_ego = transform_points(pts_global, T_global_to_curr_ego)

                # Calculate Time Lag
                # We don't have exact timestamps in this simplified lookup,
                # but we can use the sweep index * 0.05s (approx) or just the index.
                # Ideally use 'timestamp' in sample_data.
                # Here we use a simple relative time encoding: 0, 0.1, 0.2 ...
                time_lag = i * 0.1

                # Append features: x, y, z, intensity, time_lag
                # Assuming input pts has intensity at index 3
                intensity = (
                    pts[:, 3:4] if pts.shape[1] > 3 else np.zeros((pts.shape[0], 1))
                )
                pts_final = np.hstack(
                    [pts_curr_ego, intensity, np.full((pts.shape[0], 1), time_lag)]
                )

                points_list.append(pts_final)

                next_sd_token = prev_token

        if not points_list:
            # Empty fallback
            points = np.zeros((0, 5), dtype=np.float32)
        else:
            points = np.vstack(points_list).astype(np.float32)

        # 2. Augmentation (Training Only)
        # -------------------------------
        gt_boxes = []
        if (
            self.is_train
            and self.config["augment"]
            and T_global_to_curr_ego is not None
        ):
            # Load boxes
            anns = row["annotations"]
            for ann in anns:
                if ann["class_name"] in self.class_map:
                    # Convert to list: [x, y, z, w, l, h, yaw]
                    # Note: Annotations are in GLOBAL coordinates.
                    # We need to convert them to CURRENT EGO coordinates for training.

                    # Center in Global
                    center_g = np.array(
                        [ann["center_x"], ann["center_y"], ann["center_z"]]
                    ).reshape(1, 3)
                    # Center in Ego
                    center_e = transform_points(center_g, T_global_to_curr_ego)[0]

                    # Yaw in Global -> Yaw in Ego
                    # Global Yaw = Ego Yaw + Local Yaw
                    # Local Yaw = Global Yaw - Ego Yaw
                    ego_yaw = get_yaw_from_quaternion(curr_ep_r)
                    local_yaw = ann["yaw"] - ego_yaw
                    # Normalize
                    local_yaw = (local_yaw + np.pi) % (2 * np.pi) - np.pi

                    box = [
                        center_e[0],
                        center_e[1],
                        center_e[2],
                        ann["width"],
                        ann["length"],
                        ann["height"],
                        local_yaw,
                        self.class_map[ann["class_name"]],
                    ]
                    gt_boxes.append(box)

            gt_boxes = (
                np.array(gt_boxes, dtype=np.float32)
                if gt_boxes
                else np.zeros((0, 8), dtype=np.float32)
            )

            # Apply Augmentation
            points, gt_boxes = self.augment(points, gt_boxes)

        else:
            # Validation/Test: Just prepare boxes in Ego frame for metric calculation if needed
            # But for validation, we usually evaluate in Global frame.
            # However, the model predicts in Ego frame.
            # We will generate targets in Ego frame.
            if len(row["annotations"]) > 0 and T_global_to_curr_ego is not None:
                anns = row["annotations"]
                for ann in anns:
                    if ann["class_name"] in self.class_map:
                        # Transform to Ego
                        center_g = np.array(
                            [ann["center_x"], ann["center_y"], ann["center_z"]]
                        ).reshape(1, 3)
                        center_e = transform_points(center_g, T_global_to_curr_ego)[0]
                        ego_yaw = get_yaw_from_quaternion(curr_ep_r)
                        local_yaw = ann["yaw"] - ego_yaw
                        local_yaw = (local_yaw + np.pi) % (2 * np.pi) - np.pi

                        box = [
                            center_e[0],
                            center_e[1],
                            center_e[2],
                            ann["width"],
                            ann["length"],
                            ann["height"],
                            local_yaw,
                            self.class_map[ann["class_name"]],
                        ]
                        gt_boxes.append(box)
                gt_boxes = (
                    np.array(gt_boxes, dtype=np.float32)
                    if gt_boxes
                    else np.zeros((0, 8), dtype=np.float32)
                )
            else:
                gt_boxes = np.zeros((0, 8), dtype=np.float32)

        # 3. Voxelization
        # -------------------------------
        voxels, coordinates, num_points = self.voxelize(points)

        # 4. Generate Targets
        # -------------------------------
        target_dict = {}
        if self.is_train or (
            len(gt_boxes) > 0 and "test" not in self.config["metadata_path"]
        ):
            target_dict = self.generate_targets(gt_boxes)

        return {
            "voxels": voxels,
            "coordinates": coordinates,
            "num_points": num_points,
            "sample_token": sample_token,
            "gt_boxes": gt_boxes,
            **target_dict,
        }

    def augment(self, points, boxes):
        # 1. Flip X
        if np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]

        # 2. Flip Y
        if np.random.rand() < 0.5:
            points[:, 0] = -points[:, 0]
            if len(boxes) > 0:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = -(boxes[:, 6] + np.pi)

        # 3. Global Rotation
        rot_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        rot_mat = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # Rotate points
        points[:, :2] = points[:, :2] @ rot_mat.T

        # Rotate boxes
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] @ rot_mat.T
            boxes[:, 6] += rot_angle

        # 4. Global Scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :6] *= scale  # Scale xyz and wlh

        return points, boxes

    def voxelize(self, points):
        # Filter out of range
        mask = (
            (points[:, 0] >= self.pc_range[0])
            & (points[:, 0] < self.pc_range[3])
            & (points[:, 1] >= self.pc_range[1])
            & (points[:, 1] < self.pc_range[4])
            & (points[:, 2] >= self.pc_range[2])
            & (points[:, 2] < self.pc_range[5])
        )
        points = points[mask]

        # Calculate grid indices
        # coor = (pos - min) / size
        coor_x = ((points[:, 0] - self.pc_range[0]) / self.voxel_size[0]).astype(
            np.int32
        )
        coor_y = ((points[:, 1] - self.pc_range[1]) / self.voxel_size[1]).astype(
            np.int32
        )
        coor_z = ((points[:, 2] - self.pc_range[2]) / self.voxel_size[2]).astype(
            np.int32
        )

        # Create unique voxel hash
        # We use a simple linear index for unique finding
        # Assuming grid size is within limits
        voxel_indices = np.stack([coor_z, coor_y, coor_x], axis=1)

        # Grouping
        # We use lexsort to group points by voxel index
        # Sort by z, y, x
        sort_idx = np.lexsort((coor_x, coor_y, coor_z))
        points_sorted = points[sort_idx]
        voxel_indices_sorted = voxel_indices[sort_idx]

        # Find unique changes
        # diff checks where rows change
        diff = np.any(voxel_indices_sorted[1:] != voxel_indices_sorted[:-1], axis=1)
        # indices where changes happen
        change_indices = np.concatenate(
            ([0], np.nonzero(diff)[0] + 1, [len(voxel_indices_sorted)])
        )

        num_pillars = len(change_indices) - 1
        max_pillars = self.config["max_pillars"]
        max_points = self.config["max_points_per_pillar"]

        # Limit pillars
        if num_pillars > max_pillars:
            # Shuffle indices to randomly select pillars if too many (optional, but good for training)
            # Here we just take the first N for speed
            num_pillars = max_pillars
            change_indices = change_indices[: num_pillars + 1]

        # Initialize Output Tensors
        # voxels: (P, N, D)
        voxels = np.zeros((num_pillars, max_points, points.shape[1]), dtype=np.float32)
        # coordinates: (P, 3) -> (z, y, x)
        coordinates = np.zeros((num_pillars, 3), dtype=np.int32)
        # num_points: (P, )
        num_points_per_pillar = np.zeros((num_pillars,), dtype=np.int32)

        # Fill tensors
        for i in range(num_pillars):
            start = change_indices[i]
            end = change_indices[i + 1]
            pts_in_pillar = points_sorted[start:end]

            # Count
            n = min(len(pts_in_pillar), max_points)

            voxels[i, :n, :] = pts_in_pillar[:n]
            num_points_per_pillar[i] = n
            coordinates[i] = voxel_indices_sorted[start]

        return voxels, coordinates, num_points_per_pillar

    def generate_targets(self, gt_boxes):
        # gt_boxes: (N, 8) -> [x, y, z, w, l, h, yaw, class_idx]

        H, W = self.grid_size[1], self.grid_size[0]
        num_classes = len(self.config["class_names"])

        # Heatmap
        hm = np.zeros((num_classes, H, W), dtype=np.float32)

        # Regression Targets
        # We use a max of 500 objects for fixed tensor size, or return sparse
        max_objs = 500

        # Indices in the flattened heatmap
        ind = np.zeros((max_objs), dtype=np.int64)
        # Mask for valid objects
        mask = np.zeros((max_objs), dtype=np.uint8)
        # Category
        cat = np.zeros((max_objs), dtype=np.int64)

        # Regressions: [z, log(w), log(l), log(h), sin, cos, rot_z, rot_y_dummy]
        # We stick to: [z, w, l, h, sin, cos] usually + 2 offsets
        # Let's define reg target as: [offset_x, offset_y, z, w, l, h, sin, cos] -> 8 dims
        target_reg = np.zeros((max_objs, 8), dtype=np.float32)

        num_objs = min(len(gt_boxes), max_objs)

        for k in range(num_objs):
            box = gt_boxes[k]
            cls_id = int(box[7])

            # Project to grid
            x, y, z, w, l, h, yaw = box[:7]

            # Map x, y to grid coords
            # Grid is [0, W-1], [0, H-1]
            # World (0,0) is at center? No, range is [-51.2, 51.2]
            # index = (coord - min) / size

            gx = (x - self.pc_range[0]) / self.voxel_size[0]
            gy = (y - self.pc_range[1]) / self.voxel_size[1]

            gx_int, gy_int = int(gx), int(gy)

            if gx_int < 0 or gx_int >= W or gy_int < 0 or gy_int >= H:
                continue

            # Gaussian Radius
            # Heuristic: radius based on object size
            # Use min(w, l) in grid units
            # radius = max(0, int(gaussian_radius((l, w), min_overlap=0.7)))
            # Simplified: radius = 2 for car, 1 for ped?
            # Let's use a dynamic calculation
            radius = self.gaussian_radius(
                h=l / self.voxel_size[0], w=w / self.voxel_size[1], min_overlap=0.7
            )
            radius = max(0, int(radius))

            self.draw_gaussian(hm[cls_id], (gx_int, gy_int), radius)

            ind[k] = gy_int * W + gx_int
            mask[k] = 1
            cat[k] = cls_id

            # Regression targets
            # Offsets
            off_x = gx - gx_int
            off_y = gy - gy_int

            # Orientation
            sin_y = np.sin(yaw)
            cos_y = np.cos(yaw)

            # Log dims usually better for regression stability
            # But standard L1 loss on raw dims is also fine if normalized.
            # We use log dims as per CenterPoint paper
            target_reg[k] = [
                off_x,
                off_y,
                z,
                np.log(w),
                np.log(l),
                np.log(h),
                sin_y,
                cos_y,
            ]

        return {
            "hm": hm,
            "ind": ind,
            "mask": mask,
            "cat": cat,
            "target_reg": target_reg,
        }

    def gaussian_radius(self, h, w, min_overlap=0.7):
        # From CornerNet / CenterPoint
        a1 = 1
        b1 = h + w
        c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (h + w)
        c2 = (1 - min_overlap) * w * h
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (h + w)
        c3 = (min_overlap - 1) * w * h
        sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = center[0], center[1]

        height, width = heatmap.shape[0], heatmap.shape[1]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian_2d(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable voxel counts and stack them.
        """
        # Batch is a list of dicts

        res = {}

        # 1. Stack Voxels and Coords
        # We need to add batch index to coordinates
        voxels_list = []
        coords_list = []
        num_points_list = []

        for i, sample in enumerate(batch):
            voxels_list.append(torch.from_numpy(sample["voxels"]))
            num_points_list.append(torch.from_numpy(sample["num_points"]))

            # Add batch index to coords: (z, y, x) -> (batch, z, y, x)
            c = torch.from_numpy(sample["coordinates"])
            b = torch.full((c.shape[0], 1), i, dtype=torch.int32)
            coords_list.append(torch.cat([b, c], dim=1))

        res["voxels"] = torch.cat(voxels_list, dim=0)  # (Total_Pillars, 32, 5)
        res["coordinates"] = torch.cat(coords_list, dim=0)  # (Total_Pillars, 4)
        res["num_points"] = torch.cat(num_points_list, dim=0)  # (Total_Pillars, )
        res["batch_size"] = len(batch)

        # 2. Stack Targets if present
        if "hm" in batch[0]:
            res["hm"] = torch.stack([torch.from_numpy(s["hm"]) for s in batch])
            res["ind"] = torch.stack([torch.from_numpy(s["ind"]) for s in batch])
            res["mask"] = torch.stack([torch.from_numpy(s["mask"]) for s in batch])
            res["cat"] = torch.stack([torch.from_numpy(s["cat"]) for s in batch])
            res["target_reg"] = torch.stack(
                [torch.from_numpy(s["target_reg"]) for s in batch]
            )

        # 3. Metadata
        res["sample_token"] = [s["sample_token"] for s in batch]
        res["gt_boxes"] = [s["gt_boxes"] for s in batch]

        return res
