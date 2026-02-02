import os
import numpy as np
import pandas as pd
import torch
import json
import math
from library.config import Config


def read_json(path):
    """
    Loads a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def read_points(file_path, dim=4):
    """
    Reads LIDAR points from a binary file.
    Supports 4 or 5 dimensions (x, y, z, intensity, [dt]).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load raw binary
    points = np.fromfile(file_path, dtype=np.float32)

    # Attempt to reshape based on total elements
    # Priority: 5 dims (temporal), then 4 dims (standard)
    if points.shape[0] % 5 == 0:
        return points.reshape(-1, 5)
    elif points.shape[0] % 4 == 0:
        return points.reshape(-1, 4)
    else:
        # Fallback: Try to reshape to N x dim if specified, else raise error
        if points.shape[0] % dim == 0:
            return points.reshape(-1, dim)
        raise ValueError(
            f"Point cloud data shape {points.shape} not divisible by 4 or 5."
        )


def limit_period(val, offset=0, period=2 * np.pi):
    """
    Limits the value to the range [offset, offset + period].
    Commonly used to normalize yaw to [-pi, pi].
    """
    return val - np.floor(val / period + 0.5) * period


def box3d_to_corners(boxes):
    """
    Converts 3D bounding boxes to 8 corners.
    Args:
        boxes: (N, 7) [x, y, z, w, l, h, yaw]
               We assume l=length (x-axis local/heading), w=width (y-axis local), h=height.
    Returns:
        corners: (N, 8, 3)
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3))

    center = boxes[:, 0:3]
    w = boxes[:, 3:4]
    l = boxes[:, 4:5]
    h = boxes[:, 5:6]
    yaw = boxes[:, 6:7]

    # Create a bounding box at the origin
    # x_corners: [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    # y_corners: [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    # z_corners: [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    x_corners = l / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])

    corners = np.stack((x_corners, y_corners, z_corners), axis=-1)  # (N, 8, 3)

    # Rotate
    c = np.cos(yaw)
    s = np.sin(yaw)

    # Rotation matrix around Z: [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    zeros = np.zeros_like(c)
    ones = np.ones_like(c)

    rot_mat = np.concatenate(
        [c, -s, zeros, s, c, zeros, zeros, zeros, ones], axis=-1
    ).reshape(-1, 3, 3)

    # Apply rotation: (N, 8, 3) @ (N, 3, 3)^T
    corners_rot = np.einsum("nij,nkj->nki", rot_mat, corners)

    # Translate
    corners_final = corners_rot + center[:, np.newaxis, :]

    return corners_final


def points_to_voxel(points, voxel_size, coors_range):
    """
    Converts 3D points to voxel grid coordinates.
    Args:
        points: (N, 3+) [x, y, z, ...]
        voxel_size: [vx, vy, vz]
        coors_range: [min_x, min_y, min_z, max_x, max_y, max_z]
    Returns:
        coords: (M, 3) [z_idx, y_idx, x_idx] (int)
        valid_mask: (N,) bool
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points

    voxel_size = np.array(voxel_size)
    coors_range = np.array(coors_range)

    # Calculate grid size
    grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(
        np.int32
    )

    # Shift points to origin
    shifted_points = points_np[:, :3] - coors_range[:3]

    # Divide by voxel size
    coords = np.floor(shifted_points / voxel_size).astype(np.int32)

    # Check bounds
    valid_x = (coords[:, 0] >= 0) & (coords[:, 0] < grid_size[0])
    valid_y = (coords[:, 1] >= 0) & (coords[:, 1] < grid_size[1])
    valid_z = (coords[:, 2] >= 0) & (coords[:, 2] < grid_size[2])

    valid_mask = valid_x & valid_y & valid_z

    # Return in Z, Y, X order (standard for sparse tensors)
    # Note: coords is X, Y, Z currently
    final_coords = coords[valid_mask][:, ::-1]  # Z, Y, X

    return final_coords, valid_mask


def points_in_boxes_cpu(points, boxes):
    """
    Checks which points are inside which boxes.
    Args:
        points: (N, 3)
        boxes: (M, 7) [x, y, z, w, l, h, yaw]
    Returns:
        point_indices: (N, M) bool (True if point n is in box m)
    """
    # Translate points to box center
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    points_centered = points[:, np.newaxis, :] - boxes[np.newaxis, :, :3]

    # Rotate points to align with box axes
    # We need to rotate by -yaw to align points with axis-aligned box
    yaw = -boxes[:, 6]
    c = np.cos(yaw)
    s = np.sin(yaw)

    # Perform rotation manually for broadcasting
    # x_new = x*c - y*s
    # y_new = x*s + y*c
    x = points_centered[:, :, 0]
    y = points_centered[:, :, 1]
    z = points_centered[:, :, 2]

    x_rot = x * c[np.newaxis, :] - y * s[np.newaxis, :]
    y_rot = x * s[np.newaxis, :] + y * c[np.newaxis, :]
    z_rot = z

    # Check dimensions
    # Box dimensions: w (y), l (x), h (z)
    l = boxes[:, 4]
    w = boxes[:, 3]
    h = boxes[:, 5]

    in_x = np.abs(x_rot) <= l[np.newaxis, :] / 2
    in_y = np.abs(y_rot) <= w[np.newaxis, :] / 2
    in_z = np.abs(z_rot) <= h[np.newaxis, :] / 2

    return in_x & in_y & in_z


def parse_label_string(label_str):
    """
    Parses a label string into a numpy array of boxes and classes.
    Format: center_x center_y center_z width length height yaw class_name
    """
    if pd.isna(label_str) or label_str == "":
        return np.zeros((0, 7)), []

    parts = str(label_str).strip().split()
    stride = 8
    num_objects = len(parts) // stride

    boxes = []
    classes = []

    for i in range(num_objects):
        offset = i * stride
        try:
            # x, y, z, w, l, h, yaw
            box = [float(parts[offset + j]) for j in range(7)]
            cls_name = parts[offset + 7]
            boxes.append(box)
            classes.append(cls_name)
        except ValueError:
            continue

    return np.array(boxes), classes


def format_submission_string(boxes, scores, classes, score_thresh=0.0):
    """
    Formats predictions into the submission string format.
    Format: confidence center_x center_y center_z width length height yaw class_name
    """
    pred_strings = []
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < score_thresh:
            continue

        box = boxes[i]
        cls_name = classes[i]

        # box: [x, y, z, w, l, h, yaw]
        s = f"{score:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {cls_name}"
        pred_strings.append(s)

    return " ".join(pred_strings)


def _get_transform_matrix(translation, rotation):
    """
    Convert translation list and rotation quaternion to 4x4 homogenous matrix.
    Rotation is [w, x, y, z].
    """
    w, x, y, z = rotation
    # Rotation matrix from quaternion
    R = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    T = np.array(translation)

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = T
    return M


def create_gt_database(metadata_path, load_cached_data=True, max_samples=None):
    """
    Generates or loads a Ground Truth Database for augmentation.
    Caches the result as a parquet file and binary point files.
    """
    save_dir = Config.GT_DATABASE_DIR
    db_info_path = os.path.join(save_dir, "gt_database.parquet")

    os.makedirs(save_dir, exist_ok=True)

    # 1. Load Cache
    # Cite debug_lesson_1: Invalidate cache if it might be broken (empty)
    if load_cached_data and os.path.exists(db_info_path):
        try:
            df = pd.read_parquet(db_info_path)
            # Cite debug_lesson_5: Check cache validity (columns and size)
            if "class_name" in df.columns and len(df) > 0:
                print(f"Loading GT Database from {db_info_path}")
                return df
            else:
                print("Cached GT Database is invalid/empty. Regenerating...")
        except Exception:
            print("Error loading cached GT Database. Regenerating...")

    print("Generating GT Database (this may take a while)...")

    # 2. Load NuScenes metadata for coordinate transformation
    print("Loading NuScenes metadata for coordinate transformation...")
    try:
        with open(os.path.join(Config.TRAIN_DATA_DIR, "sample_data.json"), "r") as f:
            sample_data = json.load(f)
        with open(os.path.join(Config.TRAIN_DATA_DIR, "ego_pose.json"), "r") as f:
            ego_pose = json.load(f)
        with open(
            os.path.join(Config.TRAIN_DATA_DIR, "calibrated_sensor.json"), "r"
        ) as f:
            calibrated_sensor = json.load(f)
    except FileNotFoundError:
        print("Warning: NuScenes JSONs not found. Cannot transform coordinates.")
        return pd.DataFrame(
            columns=[
                "sample_token",
                "class_name",
                "box_idx",
                "box_k",
                "num_points",
                "file_path",
            ]
        )

    # Indexing for speed
    sample_to_lidar_sd = {}
    for sd in sample_data:
        if "LIDAR_TOP" in sd["filename"]:
            sample_to_lidar_sd[sd["sample_token"]] = sd

    ep_by_token = {x["token"]: x for x in ego_pose}
    cs_by_token = {x["token"]: x for x in calibrated_sensor}

    # 3. Process
    df = pd.read_csv(metadata_path)
    if max_samples:
        df = df.iloc[:max_samples]

    all_gt_infos = []
    total_samples = len(df)

    for idx, row in df.iterrows():
        sample_token = row["sample_token"]
        lidar_path = os.path.join(Config.INPUT_DIR, row["lidar_path"])
        label_str = row["label"]

        # Load Points (Sensor Frame)
        try:
            points = read_points(lidar_path)
        except FileNotFoundError:
            continue

        # Parse Boxes (Global Frame)
        gt_boxes_global, gt_classes = parse_label_string(label_str)
        if len(gt_boxes_global) == 0:
            continue

        # Get Transformation Matrix: Global -> Sensor
        if sample_token not in sample_to_lidar_sd:
            continue

        sd = sample_to_lidar_sd[sample_token]
        ep = ep_by_token[sd["ego_pose_token"]]
        cs = cs_by_token[sd["calibrated_sensor_token"]]

        # T_global_sensor = T_global_ego @ T_ego_sensor
        T_ego_sensor = _get_transform_matrix(cs["translation"], cs["rotation"])
        T_global_ego = _get_transform_matrix(ep["translation"], ep["rotation"])
        T_global_sensor = T_global_ego @ T_ego_sensor

        # T_sensor_global = inv(T_global_sensor)
        T_sensor_global = np.linalg.inv(T_global_sensor)

        # Transform Boxes to Sensor Frame
        # Center: P_sensor = T_sensor_global @ P_global
        centers_global = gt_boxes_global[:, :3]
        centers_global_hom = np.hstack(
            [centers_global, np.ones((len(centers_global), 1))]
        )
        centers_sensor = (T_sensor_global @ centers_global_hom.T).T[:, :3]

        # Yaw: Rotate vector (cos, sin, 0) by Rotation part of T_sensor_global
        yaws_global = gt_boxes_global[:, 6]
        c = np.cos(yaws_global)
        s = np.sin(yaws_global)
        zeros = np.zeros_like(c)
        # Vectors in global frame
        vecs_global = np.stack([c, s, zeros], axis=1)  # (N, 3)
        # Rotate: R_sensor_global @ vec
        R_sensor_global = T_sensor_global[:3, :3]
        vecs_sensor = (R_sensor_global @ vecs_global.T).T
        yaws_sensor = np.arctan2(vecs_sensor[:, 1], vecs_sensor[:, 0])

        # Construct GT Boxes in Sensor Frame
        # [x, y, z, w, l, h, yaw]
        gt_boxes_sensor = gt_boxes_global.copy()
        gt_boxes_sensor[:, :3] = centers_sensor
        gt_boxes_sensor[:, 6] = yaws_sensor

        # Crop Points (Now both in Sensor Frame)
        point_indices = points_in_boxes_cpu(points[:, :3], gt_boxes_sensor)

        for i in range(len(gt_boxes_sensor)):
            cls_name = gt_classes[i]
            mask = point_indices[:, i]
            box_points = points[mask]

            if box_points.shape[0] < 5:
                continue

            # Save points relative to box center (Sensor Frame)
            box_center = gt_boxes_sensor[i, :3]
            box_points_shifted = box_points.copy()
            box_points_shifted[:, :3] -= box_center

            filename = f"{sample_token}_{cls_name}_{i}.bin"
            filepath = os.path.join(save_dir, filename)
            box_points_shifted.tofile(filepath)

            # Record Info
            # We store the sensor frame box as reference for dimensions/yaw.
            info = {
                "sample_token": sample_token,
                "class_name": cls_name,
                "box_idx": i,
                "box_k": list(gt_boxes_sensor[i]),
                "num_points": box_points.shape[0],
                "file_path": filename,
            }
            all_gt_infos.append(info)

        if idx % 250 == 0:
            print(f"Processed {idx}/{total_samples} samples for GT Database")

    # 4. Save Cache
    # Explicit columns to avoid KeyError if empty
    columns = [
        "sample_token",
        "class_name",
        "box_idx",
        "box_k",
        "num_points",
        "file_path",
    ]
    gt_df = pd.DataFrame(all_gt_infos, columns=columns)

    gt_df.to_parquet(db_info_path)
    print(f"Saved GT Database with {len(gt_df)} objects to {db_info_path}")

    return gt_df
