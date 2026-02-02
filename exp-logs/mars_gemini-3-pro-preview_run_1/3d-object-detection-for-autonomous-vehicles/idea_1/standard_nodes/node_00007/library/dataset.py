import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os
import random
import library.config as config
from library.utils import transform_points
from library.preprocessing import get_bev_cached, generate_target_maps


class BEVDataset(Dataset):
    """
    PyTorch Dataset for 3D Object Detection using Rasterized BEV inputs.
    Handles data loading, coordinate transformation (World -> Sensor),
    and target generation (Heatmaps/Regression Maps).
    """

    def __init__(self, split, data_interface, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            data_interface (DataInterface): Instance for querying transforms.
            load_cached_data (bool): Whether to use cached BEV maps.
        """
        self.split = split
        self.data_interface = data_interface
        self.load_cached_data = load_cached_data

        # Select metadata file
        if split == "train":
            self.metadata_path = config.TRAIN_METADATA
        elif split == "val":
            self.metadata_path = config.VAL_METADATA
        elif split == "test":
            self.metadata_path = config.TEST_METADATA
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Parse complex columns if they exist
        if "annotations" in self.df.columns:
            # Handle potential string serialization from CSV
            self.df["annotations"] = self.df["annotations"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

    def __len__(self):
        return len(self.df)

    def _transform_annotations(self, annotations, sample_token):
        """
        Transforms annotations from World Frame to LiDAR Sensor Frame.

        Args:
            annotations (list): List of dicts (world coordinates).
            sample_token (str): Token to look up ego/sensor pose.

        Returns:
            np.ndarray: (N, 8) array [cx, cy, cz, w, l, h, yaw, class_id] in Sensor Frame.
        """
        if not annotations:
            return np.zeros((0, 8), dtype=np.float32)

        # Get transformation matrix: World -> Sensor
        # This matrix transforms a point P_world to P_sensor
        try:
            world_to_sensor = self.data_interface.get_transform_matrix(sample_token)
        except KeyError:
            # Fallback or empty if pose missing (should not happen in clean data)
            return np.zeros((0, 8), dtype=np.float32)

        # Extract centers and yaws
        centers = []
        dims = []  # w, l, h
        yaws = []
        class_ids = []

        for ann in annotations:
            cls_name = ann["class_name"]
            if cls_name not in config.CLASS_TO_ID:
                continue

            centers.append([ann["center_x"], ann["center_y"], ann["center_z"]])
            dims.append([ann["width"], ann["length"], ann["height"]])
            yaws.append(ann["yaw"])
            class_ids.append(config.CLASS_TO_ID[cls_name])

        if not centers:
            return np.zeros((0, 8), dtype=np.float32)

        centers = np.array(centers, dtype=np.float32)
        dims = np.array(dims, dtype=np.float32)
        yaws = np.array(yaws, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.float32)

        # 1. Transform Centers
        centers_sensor = transform_points(centers, world_to_sensor)

        # 2. Transform Yaws
        # We rotate a unit vector pointing in the yaw direction
        # Vector in World Frame: [cos(yaw), sin(yaw), 0]
        # We only need the rotation part of the matrix
        R = world_to_sensor[:3, :3]

        # Create vectors (N, 3)
        c_yaw = np.cos(yaws)
        s_yaw = np.sin(yaws)
        zeros = np.zeros_like(yaws)
        vectors_world = np.stack([c_yaw, s_yaw, zeros], axis=1)  # (N, 3)

        # Rotate vectors: (R @ v.T).T -> v @ R.T
        vectors_sensor = vectors_world @ R.T

        # Calculate new yaw in sensor XY plane
        yaws_sensor = np.arctan2(vectors_sensor[:, 1], vectors_sensor[:, 0])

        # Stack everything: [cx, cy, cz, w, l, h, yaw, class_id]
        # Note: Dimensions (w, l, h) are intrinsic and do not change with rigid transformation
        boxes_sensor = np.column_stack([centers_sensor, dims, yaws_sensor, class_ids])

        return boxes_sensor.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_token = row["sample_token"]
        lidar_path = row["lidar_path"]

        # 1. Get Input BEV Map (Cached)
        # Shape: (C, H, W) numpy array
        bev_map = get_bev_cached(
            sample_token,
            lidar_path,
            cache_dir=config.CACHE_DIR,
            load_cached=self.load_cached_data,
        )

        # Convert to Tensor
        input_tensor = torch.from_numpy(bev_map).float()

        # 2. Return based on split
        if self.split == "test":
            return {"input": input_tensor, "sample_token": sample_token}

        # 3. Generate Targets (Train/Val)
        annotations = row["annotations"]

        # Transform annotations to Sensor Frame
        boxes_sensor = self._transform_annotations(annotations, sample_token)

        # Geometric Augmentation (Train only) Cite solution_lesson_node_00002
        if self.split == "train":
            # Random Flip X (W-axis in tensor, X-axis in Lidar)
            if random.random() < 0.5:
                input_tensor = torch.flip(input_tensor, [2])
                if len(boxes_sensor) > 0:
                    # x -> -x
                    boxes_sensor[:, 0] = -boxes_sensor[:, 0]
                    # yaw -> pi - yaw
                    boxes_sensor[:, 6] = np.pi - boxes_sensor[:, 6]

            # Random Flip Y (H-axis in tensor, Y-axis in Lidar)
            if random.random() < 0.5:
                input_tensor = torch.flip(input_tensor, [1])
                if len(boxes_sensor) > 0:
                    # y -> -y
                    boxes_sensor[:, 1] = -boxes_sensor[:, 1]
                    # yaw -> -yaw
                    boxes_sensor[:, 6] = -boxes_sensor[:, 6]

        # Generate Heatmap and Regression Maps
        # heatmap: (NumClasses, H_out, W_out)
        # reg_map: (8, H_out, W_out)
        # reg_mask: (1, H_out, W_out)
        hm, reg, mask = generate_target_maps(
            boxes_sensor, input_shape=config.GRID_SIZE, down_ratio=config.DOWN_RATIO
        )

        return {
            "input": input_tensor,
            "hm": torch.from_numpy(hm).float(),
            "reg": torch.from_numpy(reg).float(),
            "reg_mask": torch.from_numpy(mask).float(),
            "sample_token": sample_token,
        }
