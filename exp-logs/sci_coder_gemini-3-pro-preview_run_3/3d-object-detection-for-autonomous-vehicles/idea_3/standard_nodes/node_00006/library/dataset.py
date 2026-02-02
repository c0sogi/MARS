import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import math
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    MAX_POINTS_PER_PILLAR,
    MAX_PILLARS_TRAIN,
    MAX_PILLARS_TEST,
    CLASS_TO_ID,
    NUM_POINT_FEATURES,
    NUM_PILLAR_FEATURES,
)
from library.utils import box_iou_3d_pair, get_corners_2d


class LidarDataset(Dataset):
    def __init__(
        self,
        split="train",
        root_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        load_cached_data=True,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            root_dir (str): Root directory of input data.
            metadata_dir (str): Directory containing metadata CSVs.
            load_cached_data (bool): Whether to load/save cached GT database.
        """
        self.split = split
        self.root_dir = root_dir
        self.metadata_dir = metadata_dir
        self.load_cached_data = load_cached_data

        # Load Metadata
        meta_file = os.path.join(metadata_dir, f"{split}_metadata.csv")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        self.metadata = pd.read_csv(meta_file)

        # Set Mode
        self.is_train = split == "train"

        # Configuration
        self.max_pillars = MAX_PILLARS_TRAIN if self.is_train else MAX_PILLARS_TEST

        # Ground Truth Database for Augmentation (Only for Train)
        self.gt_database = None
        if self.is_train:
            self._prepare_gt_database()

    def _prepare_gt_database(self):
        """
        Builds or loads the Ground Truth Database for Data Augmentation.
        Saves as parquet to avoid pickle.
        """
        cache_path = os.path.join(WORKING_DIR, "gt_database.parquet")

        if self.load_cached_data and os.path.exists(cache_path):
            print(f"Loading GT Database from {cache_path}...")
            try:
                self.gt_database = pd.read_parquet(cache_path)
                # Convert list columns back to numpy if necessary (parquet handles lists usually)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        print("Building GT Database from training data...")
        database = []

        # Iterate over all training samples
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            lidar_path = os.path.join(self.root_dir, row["lidar_path"])
            label_str = row["label"]

            if pd.isna(label_str):
                continue

            # Load Points
            try:
                points = np.fromfile(lidar_path, dtype=np.float32)
                # Reshape: try 5 then 4
                if points.shape[0] % 5 == 0:
                    points = points.reshape(-1, 5)
                elif points.shape[0] % 4 == 0:
                    points = points.reshape(-1, 4)
                else:
                    continue  # Skip malformed
                points = points[:, :4]  # x,y,z,i
            except Exception:
                continue

            # Parse Labels
            gt_boxes, classes = self._parse_label(label_str)
            if len(gt_boxes) == 0:
                continue

            # Crop Points for each object
            for i in range(len(gt_boxes)):
                box = gt_boxes[i]
                cls_name = classes[i]

                # Simple crop logic
                # Translate points to box center
                local_points = points.copy()
                local_points[:, 0] -= box[0]
                local_points[:, 1] -= box[1]
                local_points[:, 2] -= box[2]

                # Rotate points by -yaw to align with axis
                yaw = -box[6]
                c, s = np.cos(yaw), np.sin(yaw)
                rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                local_points[:, :3] = local_points[:, :3] @ rotation_matrix.T

                # Filter points inside box dimensions (w, l, h)
                # Box dims are full length. Local frame: [-w/2, w/2], [-l/2, l/2]
                # Note: box[3]=w, box[4]=l, box[5]=h
                # Check alignment: In utils, we assumed L is X.
                # Let's assume standard: x is length/forward, y is width/left.
                # Bounds: x in [-l/2, l/2], y in [-w/2, w/2], z in [-h/2, h/2]

                mask = (
                    (np.abs(local_points[:, 0]) < box[4] / 2)
                    & (np.abs(local_points[:, 1]) < box[3] / 2)
                    & (np.abs(local_points[:, 2]) < box[5] / 2)
                )
                obj_points = points[mask]

                if len(obj_points) > 5:  # Only keep objects with sufficient points
                    # We store points relative to center for easier pasting?
                    # No, usually store centered points.
                    # Let's store centered points (local_points[mask])
                    # We need to store the box info too.
                    database.append(
                        {
                            "class_name": cls_name,
                            "box": box.tolist(),  # [x,y,z,w,l,h,yaw]
                            "points": local_points[mask].tolist(),  # Centered points
                        }
                    )

        # Save
        df = pd.DataFrame(database)
        os.makedirs(WORKING_DIR, exist_ok=True)
        df.to_parquet(cache_path)
        self.gt_database = df
        print(f"GT Database built with {len(df)} objects.")

    def _parse_label(self, label_str):
        if pd.isna(label_str) or label_str == "":
            return np.zeros((0, 7), dtype=np.float32), []

        parts = str(label_str).strip().split()
        boxes = []
        classes = []
        stride = 8

        num_objects = len(parts) // stride
        for i in range(num_objects):
            offset = i * stride
            try:
                # x, y, z, w, l, h, yaw
                box = [float(parts[offset + j]) for j in range(7)]
                cls = parts[offset + 7]
                boxes.append(box)
                classes.append(cls)
            except ValueError:
                continue

        return np.array(boxes, dtype=np.float32), classes

    def _augment_data(self, points, gt_boxes, classes):
        """
        Applies GT Sampling, Global Rotation, and Global Scaling.
        """
        if self.gt_database is None or self.gt_database.empty:
            return points, gt_boxes, classes

        # 1. GT Database Sampling
        # Sample a few objects per class to paste
        samples_per_class = {
            "car": 2,
            "truck": 3,
            "bus": 3,
            "pedestrian": 2,
            "bicycle": 2,
            "other_vehicle": 2,
        }

        new_boxes = []
        new_points = []
        new_classes = []

        # Current existing boxes for collision check
        existing_boxes = gt_boxes.copy() if len(gt_boxes) > 0 else np.empty((0, 7))

        for cls_name, count in samples_per_class.items():
            subset = self.gt_database[self.gt_database["class_name"] == cls_name]
            if subset.empty:
                continue

            # Randomly select indices
            indices = np.random.choice(
                len(subset), min(len(subset), count), replace=False
            )

            for idx in indices:
                row = subset.iloc[idx]
                box_params = np.array(row["box"])  # [x,y,z,w,l,h,yaw]
                obj_points = np.array(row["points"])  # Centered

                # We need to place this object somewhere?
                # Usually GT Sampling pastes objects *back to their original location*
                # or random locations? Standard is original location from the DB.
                # Because the ground plane is not flat, random placement is hard.

                # Check collision with existing boxes
                # Simple BEV check
                # If collision, skip
                if len(existing_boxes) > 0:
                    # Compute distance
                    dists = np.linalg.norm(
                        existing_boxes[:, :2] - box_params[:2], axis=1
                    )
                    # Approx radius sum
                    r_new = np.sqrt(box_params[3] ** 2 + box_params[4] ** 2) / 2
                    r_ex = (
                        np.sqrt(existing_boxes[:, 3] ** 2 + existing_boxes[:, 4] ** 2)
                        / 2
                    )
                    if np.any(dists < (r_new + r_ex)):
                        continue

                # Transform points back to world (original location)
                # Rotate
                yaw = box_params[6]
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                pts_world = obj_points @ R.T
                pts_world[:, 0] += box_params[0]
                pts_world[:, 1] += box_params[1]
                pts_world[:, 2] += box_params[2]

                new_boxes.append(box_params)
                new_points.append(pts_world)
                new_classes.append(cls_name)

                # Update existing boxes for next check
                existing_boxes = np.vstack([existing_boxes, box_params.reshape(1, 7)])

        if len(new_boxes) > 0:
            # Append to main data
            gt_boxes = (
                np.vstack([gt_boxes, np.array(new_boxes)])
                if len(gt_boxes) > 0
                else np.array(new_boxes)
            )
            points = np.vstack([points, np.vstack(new_points)])
            classes.extend(new_classes)

        # 2. Global Rotation
        noise_rotation = np.random.uniform(-np.pi / 4, np.pi / 4)
        c, s = np.cos(noise_rotation), np.sin(noise_rotation)
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Rotate points
        points[:, :3] = points[:, :3] @ rot_mat.T

        # Rotate boxes
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] = gt_boxes[:, :3] @ rot_mat.T
            gt_boxes[:, 6] += noise_rotation

        # 3. Global Scaling
        noise_scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= noise_scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :6] *= noise_scale  # Scale pos and dims

        return points, gt_boxes, classes

    def _pillarize_point_cloud(self, points):
        """
        Converts point cloud to pillars.
        Returns:
            pillar_features: (M, 32, 4)
            pillar_coords: (M, 3) [z_idx, y_idx, x_idx] (z_idx always 0)
            num_points: (M,)
        """
        # Filter out of range points
        mask = (
            (points[:, 0] >= POINT_CLOUD_RANGE[0])
            & (points[:, 0] < POINT_CLOUD_RANGE[3])
            & (points[:, 1] >= POINT_CLOUD_RANGE[1])
            & (points[:, 1] < POINT_CLOUD_RANGE[4])
            & (points[:, 2] >= POINT_CLOUD_RANGE[2])
            & (points[:, 2] < POINT_CLOUD_RANGE[5])
        )
        points = points[mask]

        if len(points) == 0:
            return (
                np.zeros(
                    (0, MAX_POINTS_PER_PILLAR, NUM_POINT_FEATURES), dtype=np.float32
                ),
                np.zeros((0, 3), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        # Calculate grid indices
        x_idx = ((points[:, 0] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]).astype(np.int32)
        y_idx = ((points[:, 1] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]).astype(np.int32)

        # Use Pandas for grouping (stable and reasonably fast)
        df = pd.DataFrame(
            {
                "x": x_idx,
                "y": y_idx,
                "p0": points[:, 0],
                "p1": points[:, 1],
                "p2": points[:, 2],
                "p3": points[:, 3] if points.shape[1] > 3 else np.zeros(len(points)),
            }
        )

        # Group by grid cell
        # We need to limit to MAX_PILLARS
        # We also need to limit MAX_POINTS_PER_PILLAR

        # Assign a unique ID to each grid cell
        df["grid_id"] = df["y"].astype(np.int64) * 100000 + df["x"].astype(np.int64)

        # Sample pillars if too many
        unique_grids = df["grid_id"].unique()
        if len(unique_grids) > self.max_pillars:
            selected_grids = np.random.choice(
                unique_grids, self.max_pillars, replace=False
            )
            df = df[df["grid_id"].isin(selected_grids)]

        # Limit points per pillar
        # This is the bottleneck. We use a groupby head.
        df = df.groupby("grid_id").head(MAX_POINTS_PER_PILLAR)

        # Now construct the tensors
        # We need to group again to stack them into (M, 32, 4)
        # Sort by grid_id to ensure contiguous memory for reshaping/splitting?
        # No, pandas groupby iteration is easier.

        grouped = df.groupby("grid_id")

        num_pillars = len(grouped)
        pillar_features = np.zeros(
            (num_pillars, MAX_POINTS_PER_PILLAR, NUM_POINT_FEATURES), dtype=np.float32
        )
        pillar_coords = np.zeros((num_pillars, 3), dtype=np.int32)  # z, y, x
        num_points = np.zeros((num_pillars,), dtype=np.int32)

        for i, (grid_id, group) in enumerate(grouped):
            pts = group[["p0", "p1", "p2", "p3"]].values
            n = len(pts)
            pillar_features[i, :n, :] = pts

            # Coords
            # Recover x, y from first point in group
            x = group.iloc[0]["x"]
            y = group.iloc[0]["y"]
            pillar_coords[i] = [0, y, x]  # z is always 0 for pillars
            num_points[i] = n

            # Add offsets to features (x-xc, y-yc, z-zc, ...)
            # Calculate geometric center
            center = np.mean(pts[:, :3], axis=0)

            # We usually append these offsets.
            # But the config says NUM_POINT_FEATURES=4.
            # If the model expects 4, we don't append.
            # If the model expects more (e.g. 9 or 10), we append.
            # Config says NUM_POINT_FEATURES = 4.
            # Config says NUM_PILLAR_FEATURES = 64 (output of encoder).
            # So input is just x,y,z,i.
            # The PillarFeatureNet inside the model usually does the augmentation.
            # We will return raw points (padded).

        return pillar_features, pillar_coords, num_points

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["sample_token"]
        lidar_path = os.path.join(self.root_dir, row["lidar_path"])

        # Load Lidar
        try:
            points = np.fromfile(lidar_path, dtype=np.float32)
            if points.shape[0] % 5 == 0:
                points = points.reshape(-1, 5)
            elif points.shape[0] % 4 == 0:
                points = points.reshape(-1, 4)
            else:
                points = np.zeros((0, 4), dtype=np.float32)
            points = points[:, :4]  # Ensure 4 features
        except Exception:
            points = np.zeros((0, 4), dtype=np.float32)

        # Load Labels
        if self.is_train or self.split == "val":
            gt_boxes, classes = self._parse_label(row["label"])
            # Map class names to IDs
            gt_labels = np.array(
                [CLASS_TO_ID.get(c, -1) for c in classes], dtype=np.int64
            )
            # Filter unknown classes
            mask = gt_labels != -1
            gt_boxes = gt_boxes[mask]
            gt_labels = gt_labels[mask]
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
            classes = []

        # Augmentation (Train only)
        if self.is_train:
            points, gt_boxes, aug_classes = self._augment_data(
                points, gt_boxes, [classes[i] for i in range(len(classes)) if mask[i]]
            )
            # Re-map labels after augmentation
            gt_labels = np.array(
                [CLASS_TO_ID.get(c, -1) for c in aug_classes], dtype=np.int64
            )

        # Pillarization
        pillar_features, pillar_coords, num_points = self._pillarize_point_cloud(points)

        # Convert to Torch
        return {
            "pillar_features": torch.from_numpy(pillar_features),
            "pillar_coords": torch.from_numpy(pillar_coords),
            "num_points": torch.from_numpy(num_points),
            "gt_boxes": torch.from_numpy(gt_boxes),
            "gt_labels": torch.from_numpy(gt_labels),
            "sample_token": sample_token,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Batches pillars.
        """
        pillar_features = []
        pillar_coords = []
        num_points = []
        gt_boxes = []
        gt_labels = []
        sample_tokens = []

        for i, item in enumerate(batch):
            pillar_features.append(item["pillar_features"])
            num_points.append(item["num_points"])
            sample_tokens.append(item["sample_token"])

            # Coords need batch index
            coords = item["pillar_coords"]
            batch_idx = torch.full((len(coords), 1), i, dtype=torch.int32)
            # Prepend batch index: (batch_idx, z, y, x)
            coords_with_batch = torch.cat([batch_idx, coords], dim=1)
            pillar_coords.append(coords_with_batch)

            gt_boxes.append(item["gt_boxes"])
            gt_labels.append(item["gt_labels"])

        return {
            "pillar_features": torch.cat(pillar_features, dim=0),
            "pillar_coords": torch.cat(pillar_coords, dim=0),
            "num_points": torch.cat(num_points, dim=0),
            "gt_boxes": gt_boxes,  # List of tensors (variable length)
            "gt_labels": gt_labels,
            "sample_tokens": sample_tokens,
            "batch_size": len(batch),
        }
