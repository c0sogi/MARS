import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import iou2d_nearest


class Voxelizer:
    def __init__(self):
        self.voxel_size = np.array(Config.VOXEL_SIZE, dtype=np.float32)
        self.point_cloud_range = np.array(Config.POINT_CLOUD_RANGE, dtype=np.float32)
        self.max_points = Config.MAX_POINTS_PER_PILLAR
        self.max_pillars = Config.MAX_PILLARS
        self.grid_size = np.array(Config.GRID_SIZE, dtype=np.int32)

    def __call__(self, points):
        # points: (N, 4) [x, y, z, i]

        # 1. Filter points outside range
        mask = (
            (points[:, 0] >= self.point_cloud_range[0])
            & (points[:, 0] < self.point_cloud_range[3])
            & (points[:, 1] >= self.point_cloud_range[1])
            & (points[:, 1] < self.point_cloud_range[4])
            & (points[:, 2] >= self.point_cloud_range[2])
            & (points[:, 2] < self.point_cloud_range[5])
        )
        points = points[mask]

        if len(points) == 0:
            return (
                np.zeros((1, self.max_points, 9), dtype=np.float32),
                np.zeros((1, 2), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
            )

        # 2. Calculate Grid Coordinates
        # coords: (N, 2) [y, x] (Z is collapsed)
        coords = (
            (points[:, :2] - self.point_cloud_range[:2]) / self.voxel_size[:2]
        ).astype(np.int32)
        coords = coords[:, [1, 0]]  # Swap to (y, x)

        # 3. Group points by grid index
        # Create unique keys for sorting: y * W + x
        keys = coords[:, 0] * self.grid_size[0] + coords[:, 1]
        sorted_indices = np.argsort(keys)
        points = points[sorted_indices]
        coords = coords[sorted_indices]
        keys = keys[sorted_indices]

        # Identify unique pillars
        _, unique_indices, counts = np.unique(
            keys, return_index=True, return_counts=True
        )

        # Limit to max_pillars
        num_pillars = min(len(unique_indices), self.max_pillars)
        unique_indices = unique_indices[:num_pillars]
        counts = counts[:num_pillars]

        # Prepare Output Tensors
        # Features: [x, y, z, i, xc, yc, zc, xp, yp]
        pillars = np.zeros((num_pillars, self.max_points, 9), dtype=np.float32)
        pillar_coords = np.zeros((num_pillars, 2), dtype=np.int32)
        num_points_per_pillar = np.zeros((num_pillars,), dtype=np.int32)

        # Fill Pillars
        for i in range(num_pillars):
            start = unique_indices[i]
            count = counts[i]
            # Limit points per pillar
            num_pts = min(count, self.max_points)

            pts = points[start : start + count]
            if count > self.max_points:
                # Random sample if too many points
                choice = np.random.choice(count, self.max_points, replace=False)
                pts = pts[choice]

            # Geometric center of the pillar (voxel center)
            # x_p = (coords_x * v_x + min_x + v_x/2)
            y_idx, x_idx = coords[start]
            pillar_coords[i] = [y_idx, x_idx]

            x_p = (
                x_idx * self.voxel_size[0]
                + self.point_cloud_range[0]
                + self.voxel_size[0] / 2
            )
            y_p = (
                y_idx * self.voxel_size[1]
                + self.point_cloud_range[1]
                + self.voxel_size[1] / 2
            )
            # z_p is usually ignored or set to center of range, but here we use offset from arithmetic mean mostly

            # Arithmetic mean
            pts_mean = pts[:, :3].mean(axis=0)

            # Fill features
            # [x, y, z, i]
            pillars[i, :num_pts, :4] = pts
            # [xc, yc, zc] (offset from cluster center)
            pillars[i, :num_pts, 4:7] = pts[:, :3] - pts_mean
            # [xp, yp] (offset from pillar center)
            pillars[i, :num_pts, 7] = pts[:, 0] - x_p
            pillars[i, :num_pts, 8] = pts[:, 1] - y_p

            num_points_per_pillar[i] = num_pts

        return pillars, pillar_coords, num_points_per_pillar


class GTDatabase:
    def __init__(self, metadata_df, load_cached_data=True):
        self.cache_dir = Config.WORKING_DIR
        self.db_meta_path = os.path.join(self.cache_dir, "gt_database.parquet")
        self.db_points_path = os.path.join(self.cache_dir, "gt_database_points.bin")

        if (
            load_cached_data
            and os.path.exists(self.db_meta_path)
            and os.path.exists(self.db_points_path)
        ):
            print("Loading GT Database from cache...")
            self.metadata = pd.read_parquet(self.db_meta_path)
            self.points_data = np.memmap(
                self.db_points_path, dtype=np.float32, mode="r"
            )
        else:
            print("Generating GT Database...")
            self._generate(metadata_df)

        # Group by class for sampling
        self.db_by_class = {
            c: self.metadata[self.metadata["class_name"] == c]
            for c in Config.CLASS_NAMES
        }

    def _generate(self, metadata_df):
        os.makedirs(self.cache_dir, exist_ok=True)

        db_infos = []
        all_points = []
        current_offset = 0

        # Iterate all training samples
        for idx, row in metadata_df.iterrows():
            lidar_path = os.path.join(Config.INPUT_DIR, row["lidar_path"])
            if not os.path.exists(lidar_path):
                continue

            # Load Points
            raw_points = np.fromfile(lidar_path, dtype=np.float32)
            if raw_points.shape[0] % 5 == 0:
                points = raw_points.reshape(-1, 5)[:, :4]
            else:
                points = raw_points.reshape(-1, 4)

            # Parse Labels
            label_str = row["label"]
            if pd.isna(label_str):
                continue
            parts = str(label_str).split()
            if len(parts) % 8 != 0:
                continue

            num_objs = len(parts) // 8
            for i in range(num_objs):
                base = i * 8
                cx, cy, cz = (
                    float(parts[base]),
                    float(parts[base + 1]),
                    float(parts[base + 2]),
                )
                w, l, h = (
                    float(parts[base + 3]),
                    float(parts[base + 4]),
                    float(parts[base + 5]),
                )
                yaw = float(parts[base + 6])
                cls_name = parts[base + 7]

                if cls_name not in Config.CLASS_NAMES:
                    continue

                # Crop Points
                # Simple rotation logic to align points to box frame for cropping
                # Translate to box center
                pts_local = points[:, :3] - np.array([cx, cy, cz])
                # Rotate -yaw
                cos_a = np.cos(-yaw)
                sin_a = np.sin(-yaw)
                x_rot = pts_local[:, 0] * cos_a - pts_local[:, 1] * sin_a
                y_rot = pts_local[:, 0] * sin_a + pts_local[:, 1] * cos_a
                z_rot = pts_local[:, 2]

                # Check bounds
                mask = (
                    (np.abs(x_rot) <= w / 2)
                    & (np.abs(y_rot) <= l / 2)
                    & (np.abs(z_rot) <= h / 2)
                )
                obj_points = points[mask]

                if len(obj_points) > 0:
                    # Save info
                    # We store points relative to the object center (cx, cy, cz) to easily paste them elsewhere
                    # Actually, standard practice is to store them centered at (0,0,0) and rotate them back when pasting
                    # Let's store centered points
                    obj_points_centered = obj_points.copy()
                    obj_points_centered[:, :3] -= np.array([cx, cy, cz])

                    # Add to list
                    flat_pts = obj_points_centered.flatten()
                    all_points.append(flat_pts)

                    db_infos.append(
                        {
                            "class_name": cls_name,
                            "w": w,
                            "l": l,
                            "h": h,
                            "yaw": yaw,  # Original dims
                            "num_points": len(obj_points),
                            "offset": current_offset,
                            "length": len(flat_pts),
                        }
                    )
                    current_offset += len(flat_pts)

        # Save
        if not db_infos:
            self.metadata = pd.DataFrame(
                columns=[
                    "class_name",
                    "w",
                    "l",
                    "h",
                    "yaw",
                    "num_points",
                    "offset",
                    "length",
                ]
            )
        else:
            self.metadata = pd.DataFrame(db_infos)
        self.metadata.to_parquet(self.db_meta_path)

        if len(all_points) > 0:
            total_points_array = np.concatenate(all_points)
            # Save as binary
            with open(self.db_points_path, "wb") as f:
                f.write(total_points_array.tobytes())

            # Reload as memmap
            self.points_data = np.memmap(
                self.db_points_path, dtype=np.float32, mode="r"
            )
        else:
            # Handle empty case
            with open(self.db_points_path, "wb") as f:
                pass
            self.points_data = np.array([], dtype=np.float32)

    def sample(self, class_counts):
        """
        Sample objects to inject.
        class_counts: dict {class_name: number_to_sample}
        """
        sampled_objs = []
        for cls, count in class_counts.items():
            if count <= 0:
                continue
            if cls not in self.db_by_class or self.db_by_class[cls].empty:
                continue

            # Randomly select rows
            subset = self.db_by_class[cls]
            if len(subset) < count:
                rows = subset
            else:
                rows = subset.sample(n=count, replace=True)

            for _, row in rows.iterrows():
                # Retrieve points
                offset = int(row["offset"])
                length = int(row["length"])
                pts = np.array(self.points_data[offset : offset + length]).reshape(
                    -1, 4
                )

                sampled_objs.append(
                    {
                        "points": pts,
                        "box": [
                            0,
                            0,
                            0,
                            row["w"],
                            row["l"],
                            row["h"],
                            row["yaw"],
                        ],  # Centered at 0
                        "class_name": cls,
                    }
                )
        return sampled_objs


class LyftDataset(Dataset):
    def __init__(self, metadata_path, mode="train", load_cached_data=True):
        self.mode = mode
        self.metadata = pd.read_csv(metadata_path)
        self.voxelizer = Voxelizer()

        # GT Database for training
        self.gt_db = None
        if self.mode == "train":
            self.gt_db = GTDatabase(self.metadata, load_cached_data=load_cached_data)

        # Augmentation Settings
        self.aug_rot_range = Config.AUG_ROT_RANGE
        self.aug_scale_range = Config.AUG_SCALE_RANGE

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["sample_token"]

        # 1. Load Lidar
        lidar_path = os.path.join(Config.INPUT_DIR, row["lidar_path"])
        if os.path.exists(lidar_path):
            raw_points = np.fromfile(lidar_path, dtype=np.float32)
            if raw_points.shape[0] % 5 == 0:
                points = raw_points.reshape(-1, 5)[:, :4]
            else:
                points = raw_points.reshape(-1, 4)
        else:
            # Fallback for missing files (should not happen based on metadata check)
            points = np.zeros((1, 4), dtype=np.float32)

        # 2. Parse Labels
        gt_boxes = []
        gt_names = []

        if "label" in row and pd.notna(row["label"]):
            parts = str(row["label"]).split()
            num_objs = len(parts) // 8
            for i in range(num_objs):
                base = i * 8
                # x, y, z, w, l, h, yaw, class
                box = [float(parts[base + j]) for j in range(7)]
                cls = parts[base + 7]
                if cls in Config.CLASS_NAMES:
                    gt_boxes.append(box)
                    gt_names.append(cls)

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.zeros((0, 7), dtype=np.float32)
        )
        gt_names = np.array(gt_names)

        # 3. Augmentation (Train Only)
        if self.mode == "train":
            points, gt_boxes, gt_names = self._augment(points, gt_boxes, gt_names)

        # 4. Voxelization
        pillars, pillar_coords, num_points = self.voxelizer(points)

        # 5. Prepare Targets
        # Convert class names to IDs
        gt_classes = (
            np.array([Config.CLASS_TO_ID[n] for n in gt_names], dtype=np.int32)
            if len(gt_names) > 0
            else np.zeros((0,), dtype=np.int32)
        )

        return {
            "pillars": pillars,  # (P, N, 9)
            "pillar_coords": pillar_coords,  # (P, 2)
            "num_points": num_points,  # (P,)
            "gt_boxes": gt_boxes,  # (M, 7)
            "gt_classes": gt_classes,  # (M,)
            "sample_token": sample_token,
        }

    def _augment(self, points, gt_boxes, gt_names):
        # A. GT Sampling (Injection)
        # Define desired counts per class (simplified strategy: try to have at least N of each)
        # For this implementation, we'll just inject a few random objects to boost density
        if self.gt_db:
            # Simple sampling strategy: inject 2 cars, 2 peds, etc. if possible
            to_sample = {"car": 2, "pedestrian": 2, "truck": 1}
            sampled_objs = self.gt_db.sample(to_sample)

            new_pts = []
            new_boxes = []
            new_names = []

            # For injection, we need valid locations.
            # Randomly placing on the ground plane range [-40, 40]
            for obj in sampled_objs:
                # Random translation
                tx = np.random.uniform(-40, 40)
                ty = np.random.uniform(-40, 40)
                tz = -1.8  # Approximate ground height

                # Check collision with existing boxes (simple center distance check)
                # dist > radius_sum
                collision = False
                if len(gt_boxes) > 0:
                    dists = np.sqrt(
                        (gt_boxes[:, 0] - tx) ** 2 + (gt_boxes[:, 1] - ty) ** 2
                    )
                    # Approx radius 4m
                    if np.any(dists < 4.0):
                        collision = True

                if not collision:
                    # Transform points
                    pts = obj["points"].copy()
                    pts[:, 0] += tx
                    pts[:, 1] += ty
                    pts[:, 2] += tz

                    # Transform box
                    box = np.array(obj["box"])
                    box[0] += tx
                    box[1] += ty
                    box[2] += tz

                    new_pts.append(pts)
                    new_boxes.append(box)
                    new_names.append(obj["class_name"])

            if new_pts:
                points = np.concatenate([points] + new_pts, axis=0)
                if len(gt_boxes) > 0:
                    gt_boxes = np.concatenate([gt_boxes, np.array(new_boxes)], axis=0)
                    gt_names = np.concatenate([gt_names, np.array(new_names)], axis=0)
                else:
                    gt_boxes = np.array(new_boxes)
                    gt_names = np.array(new_names)

        # B. Global Rotation
        noise_rot = np.random.uniform(self.aug_rot_range[0], self.aug_rot_range[1])
        cos_r = np.cos(noise_rot)
        sin_r = np.sin(noise_rot)

        # Rotate points
        x = points[:, 0] * cos_r - points[:, 1] * sin_r
        y = points[:, 0] * sin_r + points[:, 1] * cos_r
        points[:, 0] = x
        points[:, 1] = y

        # Rotate boxes
        if len(gt_boxes) > 0:
            bx = gt_boxes[:, 0] * cos_r - gt_boxes[:, 1] * sin_r
            by = gt_boxes[:, 0] * sin_r + gt_boxes[:, 1] * cos_r
            gt_boxes[:, 0] = bx
            gt_boxes[:, 1] = by
            gt_boxes[:, 6] += noise_rot

        # C. Global Scaling
        noise_scale = np.random.uniform(
            self.aug_scale_range[0], self.aug_scale_range[1]
        )
        points[:, :3] *= noise_scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :6] *= noise_scale  # Scale position and dims

        return points, gt_boxes, gt_names


def collate_fn(batch):
    """
    Collate function to stack pillars and add batch indices.
    """
    pillars_list = []
    coords_list = []
    num_points_list = []
    gt_boxes_list = []
    gt_classes_list = []
    sample_tokens = []

    for i, sample in enumerate(batch):
        pillars_list.append(torch.from_numpy(sample["pillars"]))
        num_points_list.append(torch.from_numpy(sample["num_points"]))
        sample_tokens.append(sample["sample_token"])

        # Coords: Add batch index (i) to the first column
        # Original coords: (P, 2) [y, x]
        # New coords: (P, 3) [batch_idx, y, x] (Standard for Scatter)
        coords = sample["pillar_coords"]
        batch_idx = np.full((coords.shape[0], 1), i, dtype=np.int32)
        coords_with_batch = np.concatenate([batch_idx, coords], axis=1)
        coords_list.append(torch.from_numpy(coords_with_batch))

        gt_boxes_list.append(torch.from_numpy(sample["gt_boxes"]))
        gt_classes_list.append(torch.from_numpy(sample["gt_classes"]))

    # Stack dense tensors
    pillars = torch.stack(pillars_list)  # (B, MaxP, MaxPts, 9)
    num_points = torch.stack(num_points_list)  # (B, MaxP)

    # Concatenate sparse coordinates (M, 3)
    coords = torch.cat(coords_list, dim=0)

    # GT boxes are variable length, keep as list or pad?
    # Usually kept as list of tensors for loss calculation

    return {
        "pillars": pillars,
        "pillar_coords": coords,
        "num_points": num_points,
        "gt_boxes": gt_boxes_list,
        "gt_classes": gt_classes_list,
        "sample_tokens": sample_tokens,
    }
