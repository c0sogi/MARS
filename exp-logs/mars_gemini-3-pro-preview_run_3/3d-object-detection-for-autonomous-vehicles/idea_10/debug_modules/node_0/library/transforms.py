import os
import numpy as np
import pandas as pd
import pickle
import torch
from library.config import Config
from library.utils import points_in_boxes_cpu, box3d_to_corners


class GTDatabaseSampler:
    """
    Augmentation that samples objects from a ground truth database and pastes them
    into the current scene.

    Includes a 'build_database' method that performs a Geometric Unit Test:
    only objects with > MIN_POINTS_IN_GT are stored.
    """

    def __init__(self, load_cached_data=True, db_info_path=None):
        self.db_info_path = db_info_path if db_info_path else Config.DB_INFO_PATH
        self.db_points_path = self.db_info_path.replace(".parquet", "_points.bin")

        # Sampling configuration (number of objects to attempt to paste per class)
        self.sample_groups = {
            "car": 15,
            "truck": 5,
            "bus": 5,
            "bicycle": 10,
            "pedestrian": 10,
            "motorcycle": 10,
            "other_vehicle": 5,
            "emergency_vehicle": 2,
            "animal": 2,
        }

        self.db_infos = {}
        self.db_points_mmap = None

        # Build or Load Database
        if (
            load_cached_data
            and os.path.exists(self.db_info_path)
            and os.path.exists(self.db_points_path)
        ):
            print(f"Loading GT Database from {self.db_info_path}...")
            df = pd.read_parquet(self.db_info_path)
            # Group by class name for faster sampling
            for class_name, group in df.groupby("class_name"):
                self.db_infos[class_name] = group.to_dict("records")

            # Open points file as memory map for fast random access
            self.db_points_mmap = np.memmap(
                self.db_points_path, dtype=np.float32, mode="r"
            )
            # Reshape is tricky with mmap if we don't know total length, but we treat it as flat
            # and reshape slices based on metadata.
        else:
            print("Building GT Database from scratch...")
            self.build_database()
            # Reload after building
            df = pd.read_parquet(self.db_info_path)
            for class_name, group in df.groupby("class_name"):
                self.db_infos[class_name] = group.to_dict("records")
            self.db_points_mmap = np.memmap(
                self.db_points_path, dtype=np.float32, mode="r"
            )

    def build_database(self):
        """
        Iterates over training data, crops GT objects, verifies point counts,
        and saves to disk.
        """
        # 1. Load Metadata
        train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
        if not os.path.exists(train_meta_path):
            raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

        df = pd.read_csv(train_meta_path)

        all_db_infos = []
        all_points_list = []
        current_point_offset = 0

        print(f"Processing {len(df)} samples for GT Database...")

        for idx, row in df.iterrows():
            sample_token = row["sample_token"]
            lidar_rel_path = row["lidar_path"]
            label_str = row["label"]

            if pd.isna(label_str) or label_str == "":
                continue

            # Load Lidar
            lidar_path = os.path.join(Config.DATA_DIR, lidar_rel_path)
            if not os.path.exists(lidar_path):
                continue

            try:
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            except ValueError:
                continue  # Skip corrupt files

            # Parse Labels
            parts = str(label_str).strip().split()
            if len(parts) % 8 != 0:
                continue

            num_objs = len(parts) // 8
            boxes = []
            names = []

            for i in range(num_objs):
                offset = i * 8
                # x, y, z, w, l, h, yaw
                box = [float(parts[offset + j]) for j in range(7)]
                name = parts[offset + 7]
                boxes.append(box)
                names.append(name)

            boxes = np.array(boxes, dtype=np.float32)

            # Crop Points
            # points_in_boxes_cpu returns indices of points inside each box
            point_indices = points_in_boxes_cpu(points[:, :3], boxes)

            for i in range(num_objs):
                indices = point_indices[i]
                num_points = len(indices)

                # Geometric Unit Test: Skip empty or sparse boxes
                if num_points < Config.MIN_POINTS_IN_GT:
                    continue

                # Extract points
                obj_points = points[indices]

                # Transform to local coordinates (center at 0, align with box)
                # Box: x, y, z, w, l, h, yaw
                box = boxes[i]
                center = box[0:3]
                yaw = box[6]

                # Translate
                obj_points[:, :3] -= center

                # Rotate (-yaw) to align with axes
                c = np.cos(-yaw)
                s = np.sin(-yaw)

                x_local = obj_points[:, 0] * c - obj_points[:, 1] * s
                y_local = obj_points[:, 0] * s + obj_points[:, 1] * c

                obj_points[:, 0] = x_local
                obj_points[:, 1] = y_local
                # z is already relative to center after translation

                # Store Data
                info = {
                    "sample_token": sample_token,
                    "class_name": names[i],
                    "box_params": box.tolist(),  # Original world params
                    "num_points": num_points,
                    "point_offset": current_point_offset,
                    "point_count": num_points * 4,  # Storing flattened float32
                }

                all_db_infos.append(info)
                all_points_list.append(obj_points.flatten())
                current_point_offset += num_points * 4

        # Save Metadata
        info_df = pd.DataFrame(all_db_infos)
        info_df.to_parquet(self.db_info_path)

        # Save Points
        if all_points_list:
            all_points_concatenated = np.concatenate(all_points_list).astype(np.float32)
            all_points_concatenated.tofile(self.db_points_path)

        print(f"Database built. {len(info_df)} objects stored.")

    def __call__(self, data_dict):
        """
        Apply sampling to a single training sample.
        data_dict: {
            'points': np.ndarray (N, 4),
            'gt_boxes': np.ndarray (M, 7),
            'gt_names': np.ndarray (M,)
        }
        """
        # If database is empty or not loaded, return as is
        if not self.db_infos or self.db_points_mmap is None:
            return data_dict

        points = data_dict["points"]
        gt_boxes = data_dict["gt_boxes"]
        gt_names = data_dict["gt_names"]

        new_boxes = []
        new_names = []
        new_points_list = []

        # Current boxes for collision detection (BEV)
        # We use a simplified check: center distance < (sum of dims / 2)
        # Or just check if corners overlap.

        # Pre-calculate existing BEV boxes for collision check
        # Format: x, y, w, l (yaw handled loosely or strictly)
        # For speed, we'll use a simple radius check or strict check if needed.
        # Let's use the provided box3d_to_corners to get BEV polygons for strict check
        # but that might be slow inside the loop.
        # Strategy: Try to paste, check overlap with existing + newly pasted.

        existing_boxes = gt_boxes.copy()

        for class_name, sample_count in self.sample_groups.items():
            if class_name not in self.db_infos:
                continue

            # Randomly select candidates
            candidates = np.random.choice(self.db_infos[class_name], sample_count)

            for info in candidates:
                # Retrieve points
                offset = info["point_offset"]
                count = info["point_count"]
                num_pts = info["num_points"]

                # Read from mmap
                # Note: mmap is 1D array of float32
                obj_points_flat = self.db_points_mmap[offset : offset + count]
                obj_points = obj_points_flat.reshape(num_pts, 4).copy()

                # Retrieve original box dims
                # box_params: x, y, z, w, l, h, yaw
                orig_box = np.array(info["box_params"])
                w, l, h = orig_box[3], orig_box[4], orig_box[5]

                # Sample a new location?
                # Standard GT Sampling usually places the object at the SAME location
                # it was found in the original scene (preserving road context),
                # OR samples valid road planes.
                # Without a road plane map, we must rely on "Copy-Paste" to the
                # exact same coordinates but checking for collisions in the *current* scene.
                # If we want to move them, we need road segmentation.
                # Given the task description doesn't provide road planes easily,
                # we will paste at the ORIGINAL coordinates.

                # Check collision with existing boxes in this scene
                # Candidate box
                cand_box = orig_box.copy()  # x,y,z,w,l,h,yaw

                # Simple collision check: BEV Intersection
                # Expand dimensions slightly for safety
                collision = False

                # Vectorized check against existing_boxes
                if len(existing_boxes) > 0:
                    # Distances
                    dists = np.linalg.norm(existing_boxes[:, :2] - cand_box[:2], axis=1)
                    # Max radius approx
                    radii = (
                        np.linalg.norm(existing_boxes[:, 3:5], axis=1) / 2.0
                        + np.linalg.norm(cand_box[3:5]) / 2.0
                    )

                    # If distance < sum of radii, do detailed check
                    potential_collisions = existing_boxes[dists < radii]

                    if len(potential_collisions) > 0:
                        # Detailed check using corners
                        # This is expensive, so we might skip if simple check fails?
                        # Let's just be conservative and skip if close.
                        collision = True

                if not collision:
                    # Paste
                    # Points are local. Need to transform to world using cand_box.
                    # Rotate (yaw)
                    yaw = cand_box[6]
                    c = np.cos(yaw)
                    s = np.sin(yaw)

                    x_world = obj_points[:, 0] * c - obj_points[:, 1] * s
                    y_world = obj_points[:, 0] * s + obj_points[:, 1] * c

                    # Translate (center)
                    x_world += cand_box[0]
                    y_world += cand_box[1]
                    z_world = obj_points[:, 2] + cand_box[2]

                    obj_points[:, 0] = x_world
                    obj_points[:, 1] = y_world
                    obj_points[:, 2] = z_world

                    new_points_list.append(obj_points)
                    new_boxes.append(cand_box)
                    new_names.append(class_name)

                    # Add to existing boxes so we don't paste on top of it next time
                    existing_boxes = np.vstack([existing_boxes, cand_box.reshape(1, 7)])

        # Merge data
        if len(new_boxes) > 0:
            new_boxes = np.array(new_boxes)
            new_names = np.array(new_names)
            new_points = np.concatenate(new_points_list, axis=0)

            data_dict["points"] = np.concatenate([points, new_points], axis=0)
            data_dict["gt_boxes"] = np.concatenate([gt_boxes, new_boxes], axis=0)
            data_dict["gt_names"] = np.concatenate([gt_names, new_names], axis=0)

        return data_dict


class RandomFlip:
    def __init__(self, probability=0.5):
        self.prob = probability

    def __call__(self, data_dict):
        if np.random.rand() < self.prob:
            # Flip X
            data_dict["points"][:, 1] = -data_dict["points"][:, 1]
            data_dict["gt_boxes"][:, 1] = -data_dict["gt_boxes"][:, 1]
            # Flip Yaw: -yaw
            data_dict["gt_boxes"][:, 6] = -data_dict["gt_boxes"][:, 6]

        if np.random.rand() < self.prob:
            # Flip Y
            data_dict["points"][:, 0] = -data_dict["points"][:, 0]
            data_dict["gt_boxes"][:, 0] = -data_dict["gt_boxes"][:, 0]
            # Flip Yaw: pi - yaw
            data_dict["gt_boxes"][:, 6] = np.pi - data_dict["gt_boxes"][:, 6]

        return data_dict


class GlobalRotation:
    def __init__(self, rotation_range=(-np.pi / 4, np.pi / 4)):
        self.range = rotation_range

    def __call__(self, data_dict):
        noise_rotation = np.random.uniform(self.range[0], self.range[1])

        points = data_dict["points"]
        boxes = data_dict["gt_boxes"]

        # Rotate points
        c = np.cos(noise_rotation)
        s = np.sin(noise_rotation)

        x = points[:, 0] * c - points[:, 1] * s
        y = points[:, 0] * s + points[:, 1] * c
        points[:, 0] = x
        points[:, 1] = y

        # Rotate boxes
        bx = boxes[:, 0] * c - boxes[:, 1] * s
        by = boxes[:, 0] * s + boxes[:, 1] * c
        boxes[:, 0] = bx
        boxes[:, 1] = by
        boxes[:, 6] += noise_rotation

        data_dict["points"] = points
        data_dict["gt_boxes"] = boxes

        return data_dict


class GlobalScaling:
    def __init__(self, scale_range=(0.95, 1.05)):
        self.range = scale_range

    def __call__(self, data_dict):
        noise_scale = np.random.uniform(self.range[0], self.range[1])

        data_dict["points"][:, :3] *= noise_scale
        data_dict["gt_boxes"][:, :6] *= noise_scale  # Scale x,y,z,w,l,h

        return data_dict


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
