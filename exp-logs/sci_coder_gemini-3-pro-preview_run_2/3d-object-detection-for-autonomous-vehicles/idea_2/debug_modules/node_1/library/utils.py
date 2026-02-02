import os
import json
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
from library.config import Config


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


def make_transform_matrix(translation, rotation):
    """
    Construct 4x4 homogeneous transformation matrix.
    translation: [x, y, z]
    rotation: [w, x, y, z] (quaternion)
    """
    mat = np.eye(4)
    mat[:3, :3] = quaternion_to_matrix(rotation)
    mat[:3, 3] = translation
    return mat


def load_point_cloud(path):
    """
    Load LiDAR point cloud from binary file.
    Args:
        path (str): Path to .bin file
    Returns:
        np.ndarray: Point cloud array of shape (N, 4) [x, y, z, intensity]
    """
    if not os.path.exists(path):
        return np.zeros((0, 4), dtype=np.float32)

    # Load raw binary
    scan = np.fromfile(path, dtype=np.float32)

    # Reshape based on size (NuScenes/Lyft formats)
    if scan.size % 5 == 0:
        points = scan.reshape(-1, 5)
        points = points[:, :4]  # Keep x,y,z,intensity
    elif scan.size % 4 == 0:
        points = scan.reshape(-1, 4)
    elif scan.size % 3 == 0:
        points = scan.reshape(-1, 3)
        # Pad with intensity 0
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=np.float32)])
    else:
        return np.zeros((0, 4), dtype=np.float32)

    return points


def transform_points(points, matrix):
    """
    Apply 4x4 transformation matrix to points.
    Args:
        points (np.ndarray): (N, 3) or (N, 4+)
        matrix (np.ndarray): (4, 4)
    Returns:
        np.ndarray: Transformed points
    """
    if len(points) == 0:
        return points

    xyz = points[:, :3]
    # Homogeneous coords
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])

    # Transform: (Mat @ vec.T).T
    transformed_xyz = (matrix @ xyz_h.T).T

    # Update points
    points_out = points.copy()
    points_out[:, :3] = transformed_xyz[:, :3]

    return points_out


class CalibrationRegistry:
    """
    Manages loading and caching of calibration data (Sensor <-> Ego <-> Global).
    """

    def __init__(self, input_dir, cache_dir, load_cached=True):
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lookup_table = self._load_data(load_cached)

    def _load_data(self, load_cached):
        cache_path = os.path.join(self.cache_dir, "calibration_lookup.parquet")

        if load_cached and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                return df.set_index(["sample_token", "channel"]).to_dict("index")
            except Exception:
                pass

        # Load and merge JSONs
        data_frames = {}
        for table in ["sample_data", "calibrated_sensor", "ego_pose"]:
            rows = []
            for split in ["train_data", "test_data"]:
                p = os.path.join(self.input_dir, split, f"{table}.json")
                if os.path.exists(p):
                    with open(p, "r") as f:
                        rows.extend(json.load(f))
            if not rows:
                data_frames[table] = pd.DataFrame()
            else:
                data_frames[table] = pd.DataFrame(rows)

        df_sd = data_frames["sample_data"]
        df_cs = data_frames["calibrated_sensor"]
        df_ep = data_frames["ego_pose"]

        if df_sd.empty or df_cs.empty or df_ep.empty:
            return {}

        # Merge tables
        # sample_data -> calibrated_sensor (via calibrated_sensor_token)
        merged = df_sd.merge(
            df_cs,
            left_on="calibrated_sensor_token",
            right_on="token",
            suffixes=("_sd", "_cs"),
        )
        # Merged -> ego_pose (via ego_pose_token)
        merged = merged.merge(
            df_ep, left_on="ego_pose_token", right_on="token", suffixes=("", "_ep")
        )

        # Ensure 'channel' column exists. In NuScenes, it's often in calibrated_sensor,
        # but sometimes implicit. The metadata script assumes it exists or is mapped.
        # We check if 'channel' is in merged columns.
        if "channel" not in merged.columns:
            # Fallback: if not present, we can't build the map correctly by channel.
            # However, standard NuScenes sample_data usually has it or it was added.
            # We will assume 'channel' came from calibrated_sensor or sample_data.
            # If strictly missing, we skip.
            return {}

        out_df = merged[
            [
                "sample_token",
                "channel",
                "translation",
                "rotation",  # Sensor to Ego
                "translation_ep",
                "rotation_ep",  # Ego to Global
            ]
        ].copy()

        out_df.rename(
            columns={
                "translation": "trans_se",
                "rotation": "rot_se",
                "translation_ep": "trans_eg",
                "rotation_ep": "rot_eg",
            },
            inplace=True,
        )

        # Save cache
        out_df.to_parquet(cache_path)

        return out_df.set_index(["sample_token", "channel"]).to_dict("index")

    def get_transform(self, sample_token, channel):
        """
        Returns (sensor_to_ego_mat, ego_to_global_mat)
        """
        key = (sample_token, channel)
        if key not in self.lookup_table:
            return None, None

        record = self.lookup_table[key]

        t_se = np.array(record["trans_se"])
        r_se = np.array(record["rot_se"])
        t_eg = np.array(record["trans_eg"])
        r_eg = np.array(record["rot_eg"])

        mat_se = make_transform_matrix(t_se, r_se)
        mat_eg = make_transform_matrix(t_eg, r_eg)

        return mat_se, mat_eg


def get_sensor_transforms(registry, sample_token, channel):
    """
    Get the Sensor->Global transform matrix.
    """
    mat_se, mat_eg = registry.get_transform(sample_token, channel)
    if mat_se is None:
        return np.eye(4)
    return mat_eg @ mat_se


def box_iou_3d(box_a, box_b):
    """
    Calculate 3D IoU between two boxes.
    Boxes: [x, y, z, w, l, h, yaw]
    """
    # 1. Height Intersection
    za_min = box_a[2] - box_a[5] / 2
    za_max = box_a[2] + box_a[5] / 2
    zb_min = box_b[2] - box_b[5] / 2
    zb_max = box_b[2] + box_b[5] / 2

    inter_h = max(0, min(za_max, zb_max) - max(za_min, zb_min))
    if inter_h == 0:
        return 0.0

    # 2. Ground Intersection (Rotated Rectangle)
    # CV2 angle is clockwise? NuScenes is CCW. Negate yaw.
    # Ensure inputs are floats to avoid OpenCV TypeErrors with integer arrays
    rect_a = (
        (float(box_a[0]), float(box_a[1])),
        (float(box_a[3]), float(box_a[4])),
        float(-np.degrees(box_a[6])),
    )
    rect_b = (
        (float(box_b[0]), float(box_b[1])),
        (float(box_b[3]), float(box_b[4])),
        float(-np.degrees(box_b[6])),
    )

    int_type, int_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)

    if int_type == cv2.INTERSECT_NONE:
        inter_area = 0.0
    elif int_type == cv2.INTERSECT_FULL:
        inter_area = min(box_a[3] * box_a[4], box_b[3] * box_b[4])
    else:
        if int_pts is not None:
            inter_area = cv2.contourArea(int_pts)
        else:
            inter_area = 0.0

    # 3. Union Volume
    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]

    inter_vol = inter_area * inter_h
    union_vol = vol_a + vol_b - inter_vol

    if union_vol <= 1e-6:
        return 0.0

    return inter_vol / union_vol


def convert_box_to_global(box, mat_se, mat_eg):
    """
    Transform box from Sensor frame to Global frame.
    box: [x, y, z, w, l, h, yaw]
    """
    center = np.array([box[0], box[1], box[2]])
    mat_sg = mat_eg @ mat_se

    # Transform center
    center_h = np.append(center, 1.0)
    center_g = (mat_sg @ center_h)[:3]

    # Transform Yaw
    rot_se = mat_se[:3, :3]
    rot_eg = mat_eg[:3, :3]
    rot_sg = rot_eg @ rot_se

    # Box rotation in sensor frame
    r_box = R.from_euler("z", box[6], degrees=False)
    mat_box = r_box.as_matrix()

    # Global box rotation
    mat_box_g = rot_sg @ mat_box

    # Extract new yaw (Euler Z)
    r_box_g = R.from_matrix(mat_box_g)
    yaw_g = r_box_g.as_euler("zxy", degrees=False)[0]

    return np.array(
        [center_g[0], center_g[1], center_g[2], box[3], box[4], box[5], yaw_g]
    )
