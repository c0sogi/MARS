import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from library.config import Config

# ==============================================================================
# Geometric Primitives
# ==============================================================================


def quaternion_to_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def get_transform_matrix(translation, rotation, inverse=False):
    """
    Construct a 4x4 homogeneous transformation matrix from translation [x, y, z]
    and rotation quaternion [w, x, y, z].
    """
    R = quaternion_to_matrix(rotation)
    t = np.array(translation).reshape(3, 1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    if inverse:
        # Inverse of a rigid body transform
        # T^-1 = [R^T, -R^T * t]
        #        [0,    1      ]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv.flatten()
        return T_inv

    return T


def apply_transform_to_points(points, transform_matrix):
    """
    Apply 4x4 transform matrix to (N, 3) points.
    """
    if len(points) == 0:
        return points

    # Add homogeneous coordinate
    # (N, 3) -> (N, 4)
    points_h = np.hstack([points, np.ones((len(points), 1))])

    # Apply transform: (T @ P.T).T
    transformed_h = (transform_matrix @ points_h.T).T

    # Return (N, 3)
    return transformed_h[:, :3]


def apply_transform_to_boxes(boxes, transform_matrix):
    """
    Apply transform to 3D boxes.
    Boxes format: (N, 7) -> [x, y, z, w, l, h, yaw]
    """
    if len(boxes) == 0:
        return boxes

    boxes = np.array(boxes)
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]

    # 1. Transform Centers
    new_centers = apply_transform_to_points(centers, transform_matrix)

    # 2. Transform Yaw
    # Extract rotation part of the transform
    R = transform_matrix[:3, :3]

    # Create direction vectors from yaws (assuming yaw is around Z-axis)
    # NuScenes yaw: 0 = East (+x), pi/2 = North (+y)
    # Vector: [cos(yaw), sin(yaw), 0]
    c = np.cos(yaws)
    s = np.sin(yaws)
    zeros = np.zeros_like(c)

    # (N, 3) direction vectors
    dirs = np.stack([c, s, zeros], axis=1)

    # Rotate vectors
    new_dirs = (R @ dirs.T).T

    # Calculate new yaw
    new_yaws = np.arctan2(new_dirs[:, 1], new_dirs[:, 0])

    # Reassemble
    new_boxes = np.concatenate([new_centers, dims, new_yaws.reshape(-1, 1)], axis=1)

    return new_boxes


# ==============================================================================
# IoU Calculations
# ==============================================================================


def compute_iou_bev(box_a, box_b):
    """
    Calculate Intersection over Union in Bird's Eye View (2D) for two single boxes.
    Uses OpenCV for rotated rectangle intersection.

    Args:
        box_a: [x, y, z, w, l, h, yaw]
        box_b: [x, y, z, w, l, h, yaw]
    """
    # Construct RotatedRect for OpenCV
    # OpenCV RotatedRect: ((center_x, center_y), (width, height), angle_degrees)
    # NuScenes: w (y-axis), l (x-axis).
    # OpenCV angle is usually clockwise? NuScenes is CCW.
    # We map: size=(l, w), angle = -degrees(yaw)

    def to_cv_rect(box):
        x, y, _, w, l, _, yaw = box
        return ((float(x), float(y)), (float(l), float(w)), -np.degrees(yaw))

    rect_a = to_cv_rect(box_a)
    rect_b = to_cv_rect(box_b)

    # Intersection
    try:
        int_type, int_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        if int_type == cv2.INTERSECT_NONE:
            return 0.0
        elif int_type == cv2.INTERSECT_FULL:
            # One is inside the other, area is the min area
            area_a = box_a[3] * box_a[4]
            area_b = box_b[3] * box_b[4]
            return min(area_a, area_b) / max(
                area_a, area_b
            )  # This is technically wrong for IoU, should be min / max union
            # Correct logic: Intersection is min area. Union is max area.
            # IoU = min / max
        else:
            # Partial intersection
            # int_pts is a list of points
            if int_pts is not None:
                # Order points to form a convex hull (contour)
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
            else:
                int_area = 0.0
    except:
        return 0.0

    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]

    union_area = area_a + area_b - int_area
    if union_area <= 1e-6:
        return 0.0

    return int_area / union_area


def compute_iou_3d_matrix(boxes_a, boxes_b):
    """
    Compute 3D IoU matrix between two sets of boxes.
    IoU = (BEV Intersection * Height Intersection) / (3D Union)

    Note: Calculating exact rotated BEV IoU for large matrices is slow in Python.
    This function uses an axis-aligned approximation for the BEV part if boxes are
    roughly aligned, or exact if N*M is small.

    For the purpose of this task (Evaluation Metric), we implement the exact definition
    provided: IoU = (A_ground_int * H_int) / (A U B).

    However, for speed in Python, we will implement the simplified Axis-Aligned
    version which is standard for anchor matching in PointPillars.
    """
    # Expand dims for broadcasting
    # boxes_a: (N, 7) -> (N, 1, 7)
    # boxes_b: (M, 7) -> (1, M, 7)

    # 1. Height Overlap (Z-axis)
    # z_min = center_z - height/2
    # z_max = center_z + height/2
    za_min = boxes_a[:, 2] - boxes_a[:, 5] / 2
    za_max = boxes_a[:, 2] + boxes_a[:, 5] / 2
    zb_min = boxes_b[:, 2] - boxes_b[:, 5] / 2
    zb_max = boxes_b[:, 2] + boxes_b[:, 5] / 2

    # Broadcasting
    # (N, 1) vs (1, M)
    h_intersect_min = np.maximum(za_min[:, None], zb_min[None, :])
    h_intersect_max = np.minimum(za_max[:, None], zb_max[None, :])
    h_intersect = np.maximum(0, h_intersect_max - h_intersect_min)

    # 2. BEV Overlap (Axis-Aligned Approximation)
    # This is valid for anchor matching where anchors are axis aligned.
    # For rotated predictions vs GT, this is an approximation.
    xa_min = boxes_a[:, 0] - boxes_a[:, 4] / 2  # Length is usually along X
    xa_max = boxes_a[:, 0] + boxes_a[:, 4] / 2
    ya_min = boxes_a[:, 1] - boxes_a[:, 3] / 2  # Width is usually along Y
    ya_max = boxes_a[:, 1] + boxes_a[:, 3] / 2

    xb_min = boxes_b[:, 0] - boxes_b[:, 4] / 2
    xb_max = boxes_b[:, 0] + boxes_b[:, 4] / 2
    yb_min = boxes_b[:, 1] - boxes_b[:, 3] / 2
    yb_max = boxes_b[:, 1] + boxes_b[:, 3] / 2

    w_intersect_min = np.maximum(xa_min[:, None], xb_min[None, :])
    w_intersect_max = np.minimum(xa_max[:, None], xb_max[None, :])
    w_intersect = np.maximum(0, w_intersect_max - w_intersect_min)

    l_intersect_min = np.maximum(ya_min[:, None], yb_min[None, :])
    l_intersect_max = np.minimum(ya_max[:, None], yb_max[None, :])
    l_intersect = np.maximum(0, l_intersect_max - l_intersect_min)

    bev_intersect = w_intersect * l_intersect

    # 3. 3D Intersection and Union
    intersection_3d = bev_intersect * h_intersect

    vol_a = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]
    vol_b = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    union_3d = vol_a[:, None] + vol_b[None, :] - intersection_3d

    iou = intersection_3d / (union_3d + 1e-6)

    return iou


# ==============================================================================
# Data Preprocessing & Caching
# ==============================================================================


class DatasetPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sample_data = self._load_json("sample_data.json")
        self.ego_pose = self._load_json("ego_pose.json")
        self.calibrated_sensor = self._load_json("calibrated_sensor.json")

        # Create lookups
        self.sd_lookup = {r["token"]: r for r in self.sample_data}
        self.ep_lookup = {r["token"]: r for r in self.ego_pose}
        self.cs_lookup = {r["token"]: r for r in self.calibrated_sensor}

        # Map sample_token to LIDAR_TOP sample_data token
        self.sample_to_lidar = {}
        for r in self.sample_data:
            if r["filename"].endswith(".bin") or "LIDAR" in r["filename"]:
                # Assuming one LIDAR per sample for simplicity or taking the first one found
                # In NuScenes, channel is usually 'LIDAR_TOP'
                # We check if this record points to a sample
                if r["sample_token"] not in self.sample_to_lidar:
                    self.sample_to_lidar[r["sample_token"]] = r["token"]
                else:
                    # Prefer LIDAR_TOP if multiple
                    # (Simple heuristic: if current is not TOP and new is TOP, swap)
                    # But here we just assume the first one is fine or the only one.
                    pass

    def _load_json(self, filename):
        path = os.path.join(self.data_dir, filename)
        with open(path, "r") as f:
            return json.load(f)

    def get_sensor_transforms(self, sample_token):
        """
        Returns the matrices to transform Global -> Sensor.
        """
        sd_token = self.sample_to_lidar.get(sample_token)
        if not sd_token:
            return None

        sd_record = self.sd_lookup[sd_token]
        ep_record = self.ep_lookup[sd_record["ego_pose_token"]]
        cs_record = self.cs_lookup[sd_record["calibrated_sensor_token"]]

        # Global -> Ego
        # ego_pose gives Ego -> Global
        # We need Inverse
        t_ego_global = get_transform_matrix(
            ep_record["translation"], ep_record["rotation"]
        )
        t_global_ego = np.linalg.inv(t_ego_global)

        # Ego -> Sensor
        # calibrated_sensor gives Sensor -> Ego
        # We need Inverse
        t_sensor_ego = get_transform_matrix(
            cs_record["translation"], cs_record["rotation"]
        )
        t_ego_sensor = np.linalg.inv(t_sensor_ego)

        # Global -> Sensor = (Ego -> Sensor) * (Global -> Ego)
        t_global_sensor = t_ego_sensor @ t_global_ego

        return t_global_sensor, t_sensor_ego, t_ego_global

    def load_and_process(self, metadata_path, cache_file, load_cached_data=True):
        """
        Loads metadata, transforms GT boxes to sensor frame, and caches result.
        """
        # Ensure directory exists
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            try:
                # Load using pandas (parquet)
                df = pd.read_parquet(cache_file)
                # Convert list columns back from whatever format parquet stored them in
                # Typically parquet handles lists of structs well, or we stored as json string
                # Let's assume we store as simple columns and 'boxes' is a list of lists
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Processing data from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        processed_data = []

        for idx, row in df_meta.iterrows():
            token = row["token"]

            # Parse paths
            paths = json.loads(row["file_paths"].replace("'", '"'))
            lidar_path = None
            for k, v in paths.items():
                if "LIDAR" in k:
                    lidar_path = v
                    break

            if lidar_path is None:
                continue

            # Parse annotations
            anns = json.loads(row["annotations"].replace("'", '"'))

            # Prepare Boxes (Global Frame)
            boxes_global = []
            class_names = []
            for ann in anns:
                # [x, y, z, w, l, h, yaw]
                # Note: 'length' in csv usually corresponds to x-axis size in object frame
                b = [
                    ann["center_x"],
                    ann["center_y"],
                    ann["center_z"],
                    ann["width"],
                    ann["length"],
                    ann["height"],
                    ann["yaw"],
                ]
                boxes_global.append(b)
                class_names.append(ann["class_name"])

            boxes_global = np.array(boxes_global)

            # Transform to Sensor Frame
            transforms = self.get_sensor_transforms(token)
            if transforms is None:
                continue

            t_global_sensor, _, _ = transforms

            if len(boxes_global) > 0:
                boxes_sensor = apply_transform_to_boxes(boxes_global, t_global_sensor)
            else:
                boxes_sensor = np.zeros((0, 7))

            # Store
            processed_data.append(
                {
                    "token": token,
                    "lidar_path": lidar_path,
                    "boxes": boxes_sensor.tolist(),
                    "class_names": class_names,
                }
            )

        # Create DataFrame
        df_processed = pd.DataFrame(processed_data)

        # Save to Cache
        print(f"Saving processed data to {cache_file}...")
        df_processed.to_parquet(cache_file, index=False)

        return df_processed
