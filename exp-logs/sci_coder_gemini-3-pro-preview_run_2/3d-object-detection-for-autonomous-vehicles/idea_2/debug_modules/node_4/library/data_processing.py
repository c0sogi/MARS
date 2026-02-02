import os
import numpy as np
import pandas as pd
import json
import warnings
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation as R

from library.config import Config
from library.utils import (
    load_point_cloud,
    CalibrationRegistry,
    get_sensor_transforms,
    transform_points,
    box_iou_3d,
)

# Suppress warnings from sklearn/numpy
warnings.filterwarnings("ignore")


class PointCloudProcessor:
    """
    Handles geometric processing of LiDAR point clouds: ROI cropping,
    ground removal, and clustering.
    """

    def __init__(self):
        self.roi_x = (Config.X_MIN, Config.X_MAX)
        self.roi_y = (Config.Y_MIN, Config.Y_MAX)
        self.roi_z = (Config.Z_MIN, Config.Z_MAX)

        # RANSAC for ground plane fitting
        self.ransac = RANSACRegressor(
            min_samples=3,
            residual_threshold=Config.RANSAC_DIST_THRESH,
            max_trials=Config.RANSAC_ITERATIONS,
            random_state=Config.SEED,
        )

        # DBSCAN for clustering
        self.dbscan = DBSCAN(
            eps=Config.DBSCAN_EPS, min_samples=Config.DBSCAN_MIN_SAMPLES, n_jobs=-1
        )

    def preprocess(self, points):
        """
        Filter points within the Region of Interest (ROI).
        """
        if len(points) == 0:
            return points

        mask = (
            (points[:, 0] >= self.roi_x[0])
            & (points[:, 0] <= self.roi_x[1])
            & (points[:, 1] >= self.roi_y[0])
            & (points[:, 1] <= self.roi_y[1])
            & (points[:, 2] >= self.roi_z[0])
            & (points[:, 2] <= self.roi_z[1])
        )
        return points[mask]

    def remove_ground(self, points):
        """
        Remove ground points using RANSAC plane fitting.
        """
        if len(points) < 10:
            return points, np.array([])

        # Downsample for faster RANSAC fitting
        # Use every 10th point or max 5000 points
        step = max(1, len(points) // 5000)
        sample_points = points[::step]

        # We assume ground is roughly a plane z = f(x, y)
        X = sample_points[:, :2]
        y = sample_points[:, 2]

        try:
            self.ransac.fit(X, y)

            # Predict for all points
            all_X = points[:, :2]
            all_y = points[:, 2]
            predicted_z = self.ransac.predict(all_X)

            # Calculate residuals
            residuals = np.abs(all_y - predicted_z)

            # Inliers are ground, outliers are obstacles
            mask_ground = residuals < Config.RANSAC_DIST_THRESH

            # Return non-ground points
            return points[~mask_ground]

        except Exception:
            # Fallback if RANSAC fails (e.g., collinear points)
            return points

    def cluster_points(self, points):
        """
        Cluster points using DBSCAN. Returns a list of point arrays (clusters).
        """
        if len(points) < Config.MIN_CLUSTER_POINTS:
            return []

        # Use only XYZ for clustering
        xyz = points[:, :3]
        labels = self.dbscan.fit_predict(xyz)

        clusters = []
        unique_labels = set(labels)

        for lbl in unique_labels:
            if lbl == -1:
                continue  # Noise

            cluster_mask = labels == lbl
            cluster_pts = points[cluster_mask]

            if len(cluster_pts) >= Config.MIN_CLUSTER_POINTS:
                clusters.append(cluster_pts)

        return clusters


class FeatureExtractor:
    """
    Computes geometric features for a cluster of points.
    """

    def extract(self, cluster):
        if len(cluster) == 0:
            return None

        xyz = cluster[:, :3]
        N = len(xyz)

        # 1. Spatial Statistics
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        mean_xyz = np.mean(xyz, axis=0)
        std_xyz = np.std(xyz, axis=0)

        # 2. Dimensions (Axis Aligned Bounding Box)
        dims = max_xyz - min_xyz
        width, length, height = dims[0], dims[1], dims[2]

        # 3. Covariance / Eigenvalues (PCA)
        # Handle cases with coplanar/collinear points
        try:
            cov_mat = np.cov(xyz, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov_mat)
            # Sort descending
            eigenvalues = eigenvalues[::-1]
            # Normalize sum to 1 for shape descriptors
            eig_sum = np.sum(eigenvalues) + 1e-9
            norm_eigs = eigenvalues / eig_sum
        except:
            eigenvalues = np.zeros(3)
            norm_eigs = np.zeros(3)

        # 4. Construct Feature Vector
        features = {
            "point_count": N,
            "x_min": min_xyz[0],
            "y_min": min_xyz[1],
            "z_min": min_xyz[2],
            "x_max": max_xyz[0],
            "y_max": max_xyz[1],
            "z_max": max_xyz[2],
            "x_mean": mean_xyz[0],
            "y_mean": mean_xyz[1],
            "z_mean": mean_xyz[2],
            "x_std": std_xyz[0],
            "y_std": std_xyz[1],
            "z_std": std_xyz[2],
            "cluster_width": width,
            "cluster_length": length,
            "cluster_height": height,
            "cluster_volume": width * length * height,
            "eig_1": eigenvalues[0],
            "eig_2": eigenvalues[1],
            "eig_3": eigenvalues[2],
            "norm_eig_1": norm_eigs[0],
            "norm_eig_2": norm_eigs[1],
            "norm_eig_3": norm_eigs[2],
            "density": N / (width * length * height + 1e-6),
        }

        return features


def transform_box_global_to_sensor(box_global, mat_sg):
    """
    Transform a global bounding box to the sensor frame.
    box_global: dict with center_x, center_y, center_z, width, length, height, yaw
    mat_sg: Sensor to Global transformation matrix (4x4)
    """
    # Compute Global to Sensor Matrix
    try:
        mat_gs = np.linalg.inv(mat_sg)
    except np.linalg.LinAlgError:
        return None

    # Transform Center
    center_g = np.array(
        [box_global["center_x"], box_global["center_y"], box_global["center_z"], 1.0]
    )
    center_s = mat_gs @ center_g

    # Transform Yaw
    # Create a unit vector pointing in the direction of yaw in global frame
    yaw_g = box_global["yaw"]
    vec_g = np.array([np.cos(yaw_g), np.sin(yaw_g), 0.0, 0.0])  # Direction vector

    # Rotate vector to sensor frame
    vec_s = mat_gs @ vec_g

    # Compute new yaw
    yaw_s = np.arctan2(vec_s[1], vec_s[0])

    return {
        "center_x": center_s[0],
        "center_y": center_s[1],
        "center_z": center_s[2],
        "width": box_global["width"],
        "length": box_global["length"],
        "height": box_global["height"],
        "yaw": yaw_s,
        "class_name": box_global["class_name"],
    }


def create_training_dataset(metadata_path, load_cached_data=True):
    """
    Generates a tabular dataset for training the XGBoost model.
    Iterates through samples, processes point clouds, and matches clusters to GT.
    """
    cache_file = os.path.join(Config.WORKING_DIR, "train_features.parquet")

    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached training features from {cache_file}...")
        return pd.read_parquet(cache_file)

    print("Generating training dataset from scratch...")

    # 2. Load Metadata
    df_meta = pd.read_csv(metadata_path)
    # Parse JSON columns
    df_meta["file_paths"] = df_meta["file_paths"].apply(json.loads)
    df_meta["annotations"] = df_meta["annotations"].apply(json.loads)

    # Debugging: Sample subset if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        df_meta = df_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]
        print(f"Debugging mode: Processing {len(df_meta)} samples.")

    # 3. Initialize Processors
    processor = PointCloudProcessor()
    extractor = FeatureExtractor()
    registry = CalibrationRegistry(Config.INPUT_DIR, Config.WORKING_DIR)

    dataset_rows = []

    # 4. Iterate Samples
    for idx, row in df_meta.iterrows():
        sample_token = row["token"]
        paths = row["file_paths"]
        annotations = row["annotations"]

        # Find LiDAR file
        lidar_path = None
        lidar_channel = None
        for ch, p in paths.items():
            if "LIDAR" in ch or p.endswith(".bin"):
                lidar_path = os.path.join(Config.INPUT_DIR, p)
                lidar_channel = ch
                break

        if not lidar_path or not os.path.exists(lidar_path):
            continue

        # Get Transforms
        mat_se, mat_eg = registry.get_transform(sample_token, lidar_channel)
        if mat_se is None:
            continue
        mat_sg = mat_eg @ mat_se

        # Transform GT Boxes to Sensor Frame
        gt_boxes_s = []
        for ann in annotations:
            if ann["class_name"] not in Config.CLASSES:
                continue
            box_s = transform_box_global_to_sensor(ann, mat_sg)
            if box_s:
                gt_boxes_s.append(box_s)

        # Load and Process Point Cloud
        points = load_point_cloud(lidar_path)
        points = processor.preprocess(points)
        points_no_ground = processor.remove_ground(points)
        clusters = processor.cluster_points(points_no_ground)

        # Process Clusters
        for cluster in clusters:
            # Extract Features
            feats = extractor.extract(cluster)
            if feats is None:
                continue

            # Match with GT
            cluster_center = np.array(
                [feats["x_mean"], feats["y_mean"], feats["z_mean"]]
            )

            best_match = None
            min_dist = float("inf")

            for box in gt_boxes_s:
                box_center = np.array(
                    [box["center_x"], box["center_y"], box["center_z"]]
                )
                dist = np.linalg.norm(cluster_center - box_center)

                if dist < min_dist:
                    min_dist = dist
                    best_match = box

            # Label Assignment
            if best_match and min_dist < Config.MATCH_DIST_THRESHOLD:
                # Positive Sample
                feats["target_class"] = Config.CLASS_TO_ID.get(
                    best_match["class_name"], -1
                )
                feats["target_center_x"] = best_match["center_x"]
                feats["target_center_y"] = best_match["center_y"]
                feats["target_center_z"] = best_match["center_z"]
                feats["target_width"] = best_match["width"]
                feats["target_length"] = best_match["length"]
                feats["target_height"] = best_match["height"]
                feats["target_yaw"] = best_match["yaw"]
                feats["is_background"] = 0
            else:
                # Background Sample
                feats["target_class"] = -1
                # Fill targets with dummy values (won't be used for regression training on bg)
                feats["target_center_x"] = 0.0
                feats["target_center_y"] = 0.0
                feats["target_center_z"] = 0.0
                feats["target_width"] = 0.0
                feats["target_length"] = 0.0
                feats["target_height"] = 0.0
                feats["target_yaw"] = 0.0
                feats["is_background"] = 1

            feats["sample_token"] = sample_token
            dataset_rows.append(feats)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df_meta)} samples...")

    # 5. Save Dataset
    if not dataset_rows:
        print("Warning: No clusters generated. Returning empty DataFrame.")
        return pd.DataFrame()

    df_dataset = pd.DataFrame(dataset_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    print(f"Saving {len(df_dataset)} rows to {cache_file}...")
    df_dataset.to_parquet(cache_file)

    return df_dataset
