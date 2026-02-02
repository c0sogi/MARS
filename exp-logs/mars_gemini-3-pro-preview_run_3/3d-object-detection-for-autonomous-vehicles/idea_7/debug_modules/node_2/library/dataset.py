import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import math

from library.config import (
    DATA_ROOT,
    CLASS_NAMES,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    GRID_SIZE,
    MAX_POINTS_PER_VOXEL,
    MAX_NUMBER_OF_VOXELS_TRAIN,
    ROTATION_RANGE,
    SCALING_RANGE,
    FLIP_PROB,
    NUM_POINT_FEATURES,
)
from library.utils import draw_umich_gaussian, gaussian_radius

# Constants
MAX_OBJECTS = 100  # K
OUTPUT_STRIDE = 4
FEATURE_MAP_SIZE = [
    GRID_SIZE[0] // OUTPUT_STRIDE,
    GRID_SIZE[1] // OUTPUT_STRIDE,
]  # 320x320


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


def transform_matrix(translation, rotation):
    """
    Create 4x4 homogenous transformation matrix.
    rotation: quaternion [w, x, y, z]
    translation: [x, y, z]
    """
    R = quaternion_to_matrix(rotation)
    T = np.array(translation)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


class NuScenesDataset(Dataset):
    def __init__(self, metadata_path, split="train", root_dir=DATA_ROOT):
        self.metadata = pd.read_csv(metadata_path)
        self.split = split
        self.root_dir = root_dir
        self.class_map = {name: i for i, name in enumerate(CLASS_NAMES)}

        # Determine data folder based on split
        # train_metadata and val_metadata use 'train_data'
        # test_metadata uses 'test_data'
        if "test" in metadata_path:
            self.data_folder = "test_data"
        else:
            self.data_folder = "train_data"

        # Load Coordinate Transformation Info
        self._load_transform_data()

    def _load_transform_data(self):
        """
        Load JSONs and build sample_token -> sensor_pose_matrix mapping.
        """
        # Paths
        sample_data_path = os.path.join(
            self.root_dir, self.data_folder, "sample_data.json"
        )
        ego_pose_path = os.path.join(self.root_dir, self.data_folder, "ego_pose.json")
        cal_sensor_path = os.path.join(
            self.root_dir, self.data_folder, "calibrated_sensor.json"
        )

        # Load JSONs
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)
        with open(ego_pose_path, "r") as f:
            ego_pose = {item["token"]: item for item in json.load(f)}
        with open(cal_sensor_path, "r") as f:
            cal_sensor = {item["token"]: item for item in json.load(f)}

        # Build Map: sample_token -> (ego_pose_record, cal_sensor_record)
        # Note: One sample has multiple sensors. We need the LIDAR one.
        # The metadata contains 'lidar_path'. We can match by filename.

        # Create a map from filename (basename) to sample_data record
        filename_to_record = {}
        for item in sample_data:
            if item["filename"].endswith(".bin"):
                fname = os.path.basename(item["filename"])
                filename_to_record[fname] = item

        self.transforms = {}

        # Pre-compute transforms for all samples in metadata
        for idx, row in self.metadata.iterrows():
            s_token = row["sample_token"]
            lidar_path = row["lidar_path"]
            fname = os.path.basename(lidar_path)

            if fname not in filename_to_record:
                continue

            sd_record = filename_to_record[fname]
            ego_token = sd_record["ego_pose_token"]
            cs_token = sd_record["calibrated_sensor_token"]

            ego = ego_pose[ego_token]
            cs = cal_sensor[cs_token]

            # World -> Ego
            M_ego = transform_matrix(ego["translation"], ego["rotation"])
            # Ego -> Sensor
            M_sens = transform_matrix(cs["translation"], cs["rotation"])

            # World -> Sensor = inv(M_sens) @ inv(M_ego)
            # P_sensor = inv(M_sens) @ inv(M_ego) @ P_world

            M_global_to_sensor = np.linalg.inv(M_ego @ M_sens)
            self.transforms[s_token] = M_global_to_sensor

    def parse_labels(self, label_str):
        if pd.isna(label_str) or label_str == "":
            return np.zeros((0, 8)), []

        parts = str(label_str).strip().split()
        stride = 8
        num_objs = len(parts) // stride

        boxes = []
        classes = []

        for i in range(num_objs):
            offset = i * stride
            # x, y, z, w, l, h, yaw, class
            try:
                vals = [float(x) for x in parts[offset : offset + 7]]
                cls_name = parts[offset + 7]
                if cls_name in self.class_map:
                    boxes.append(vals)
                    classes.append(self.class_map[cls_name])
            except ValueError:
                continue

        return np.array(boxes), np.array(classes)

    def augment(self, points, boxes):
        # 1. Global Rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
            rot_cos = np.cos(angle)
            rot_sin = np.sin(angle)
            rot_mat = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
            )

            # Rotate Points
            points[:, :3] = points[:, :3] @ rot_mat.T

            # Rotate Boxes (x, y, z)
            if len(boxes) > 0:
                boxes[:, :3] = boxes[:, :3] @ rot_mat.T
                boxes[:, 6] += angle  # Yaw

        # 2. Global Scaling
        scale = np.random.uniform(SCALING_RANGE[0], SCALING_RANGE[1])
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :3] *= scale  # Position
            boxes[:, 3:6] *= scale  # Dimensions

        # 3. Flip X (Flip along Y axis)
        if np.random.random() < FLIP_PROB:
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]  # Yaw flip: -yaw

        # 4. Flip Y (Flip along X axis)
        if np.random.random() < FLIP_PROB:
            points[:, 0] = -points[:, 0]
            if len(boxes) > 0:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = np.pi - boxes[:, 6]  # Yaw flip: pi - yaw

        return points, boxes

    def generate_targets(self, boxes, classes):
        # Init targets
        heatmap = np.zeros(
            (len(CLASS_NAMES), FEATURE_MAP_SIZE[1], FEATURE_MAP_SIZE[0]),
            dtype=np.float32,
        )

        ind = np.zeros((MAX_OBJECTS), dtype=np.int64)
        mask = np.zeros((MAX_OBJECTS), dtype=np.uint8)

        reg = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
        height = np.zeros((MAX_OBJECTS, 1), dtype=np.float32)
        dim = np.zeros((MAX_OBJECTS, 3), dtype=np.float32)
        rot = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)

        num_objs = min(len(boxes), MAX_OBJECTS)

        for i in range(num_objs):
            box = boxes[i]
            cls_id = int(classes[i])

            # Box: x, y, z, w, l, h, yaw
            x, y, z, w, l, h, yaw = box

            # Filter if outside range
            if (
                x < POINT_CLOUD_RANGE[0]
                or x > POINT_CLOUD_RANGE[3]
                or y < POINT_CLOUD_RANGE[1]
                or y > POINT_CLOUD_RANGE[4]
            ):
                continue

            # Coords in feature map
            # Offset by range min, divide by effective stride
            coor_x = (x - POINT_CLOUD_RANGE[0]) / (VOXEL_SIZE[0] * OUTPUT_STRIDE)
            coor_y = (y - POINT_CLOUD_RANGE[1]) / (VOXEL_SIZE[1] * OUTPUT_STRIDE)

            ct_x = int(coor_x)
            ct_y = int(coor_y)

            if (
                ct_x < 0
                or ct_x >= FEATURE_MAP_SIZE[0]
                or ct_y < 0
                or ct_y >= FEATURE_MAP_SIZE[1]
            ):
                continue

            # Gaussian Radius
            # Use box dimensions in feature map pixels
            # w, l are in meters.
            w_pixel = w / (VOXEL_SIZE[0] * OUTPUT_STRIDE)
            l_pixel = l / (VOXEL_SIZE[1] * OUTPUT_STRIDE)
            radius = gaussian_radius((l_pixel, w_pixel), min_overlap=0.7)
            radius = max(0, int(radius))

            draw_umich_gaussian(heatmap[cls_id], (ct_x, ct_y), radius)

            # Regression Targets
            ind[i] = ct_y * FEATURE_MAP_SIZE[0] + ct_x
            mask[i] = 1

            # Offset
            reg[i] = [coor_x - ct_x, coor_y - ct_y]

            # Height
            height[i] = z

            # Dim (Log space)
            dim[i] = [np.log(l), np.log(w), np.log(h)]  # Order: l, w, h

            # Rot (sin, cos)
            rot[i] = [np.sin(yaw), np.cos(yaw)]

        return {
            "heatmap": heatmap,
            "ind": ind,
            "mask": mask,
            "reg": reg,
            "height": height,
            "dim": dim,
            "rot": rot,
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        s_token = row["sample_token"]
        lidar_path = os.path.join(self.root_dir, row["lidar_path"])

        # 1. Load Points
        # Standard Lidar format: x, y, z, i, r (sometimes). We take first 4.
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]

        # 2. Load Labels & Transform
        if "label" in row and pd.notna(row["label"]):
            boxes_world, classes = self.parse_labels(row["label"])

            # Transform World -> Sensor
            if len(boxes_world) > 0 and s_token in self.transforms:
                M = self.transforms[s_token]
                # Pad boxes for transform
                # Centers: [x, y, z, 1]
                centers = np.hstack(
                    [boxes_world[:, :3], np.ones((len(boxes_world), 1))]
                )
                centers_sens = centers @ M.T
                boxes_world[:, :3] = centers_sens[:, :3]

                # Rotate Yaw
                # Extract rotation from M (top-left 3x3)
                R = M[:3, :3]
                # Rotate orientation vector
                v_yaw = np.stack(
                    [
                        np.cos(boxes_world[:, 6]),
                        np.sin(boxes_world[:, 6]),
                        np.zeros_like(boxes_world[:, 6]),
                    ],
                    axis=1,
                )
                v_new = v_yaw @ R.T
                boxes_world[:, 6] = np.arctan2(v_new[:, 1], v_new[:, 0])

            boxes = boxes_world
        else:
            boxes = np.zeros((0, 8))
            classes = np.zeros((0))

        # 3. Augmentation (Train only)
        if self.split == "train":
            points, boxes = self.augment(points, boxes)

        # 4. Generate Targets
        targets = self.generate_targets(boxes, classes)

        # 5. Metadata
        metadata = {"sample_token": s_token, "lidar_path": lidar_path}

        # Return Dict
        return {
            "points": points,
            "gt_boxes": boxes,  # For eval
            "gt_labels": classes,
            "metadata": metadata,
            "targets": targets,
        }


def custom_collate_fn(batch_list):
    """
    Custom collate function to handle points, metadata, and CenterPoint targets.
    """
    batched_points = []
    batched_gt_boxes = []
    batched_labels = []
    batched_metadata = []

    # Targets
    batched_targets = {
        "heatmap": [],
        "ind": [],
        "mask": [],
        "reg": [],
        "height": [],
        "dim": [],
        "rot": [],
    }

    for i, sample in enumerate(batch_list):
        # Points: Add batch index
        pts = torch.from_numpy(sample["points"])
        batch_idx = torch.full((pts.shape[0], 1), i, dtype=pts.dtype)
        pts_with_idx = torch.cat([batch_idx, pts], dim=1)
        batched_points.append(pts_with_idx)

        # GT Boxes
        batched_gt_boxes.append(torch.from_numpy(sample["gt_boxes"]))
        batched_labels.append(torch.from_numpy(sample["gt_labels"]))

        # Metadata
        batched_metadata.append(sample["metadata"])

        # Targets
        t = sample["targets"]
        for k in batched_targets.keys():
            batched_targets[k].append(torch.from_numpy(t[k]))

    # Stack Points
    batched_points = torch.cat(batched_points, dim=0)

    # Pad GT Boxes
    max_boxes = max([b.shape[0] for b in batched_gt_boxes]) if batched_gt_boxes else 0
    padded_boxes = torch.zeros((len(batch_list), max_boxes, 8))
    padded_labels = torch.zeros((len(batch_list), max_boxes)).long()

    if max_boxes > 0:
        for i, (b, l) in enumerate(zip(batched_gt_boxes, batched_labels)):
            if len(b) > 0:
                padded_boxes[i, : len(b)] = b
                padded_labels[i, : len(l)] = l

    # Stack Targets
    final_targets = {}
    for k, v in batched_targets.items():
        final_targets[k] = torch.stack(v, dim=0)

    return {
        "points": batched_points,
        "gt_boxes": padded_boxes,
        "gt_labels": padded_labels,
        "metadata": batched_metadata,
        "targets": final_targets,
        "batch_size": len(batch_list),
    }
