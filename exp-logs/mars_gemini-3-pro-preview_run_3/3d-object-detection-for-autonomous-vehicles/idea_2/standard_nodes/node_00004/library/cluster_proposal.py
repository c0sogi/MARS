import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import load_lidar


class GeometricProposalGenerator:
    """
    Implements Stage 1 of the pipeline: Unsupervised Geometric Proposal Generation.
    Includes Ground Plane Removal (RANSAC), Clustering (DBSCAN), and Bounding Box Estimation (PCA).
    """

    def __init__(self):
        self.ransac_dist_thresh = Config.RANSAC_DIST_THRESH
        self.ransac_iterations = Config.RANSAC_ITERATIONS
        self.dbscan_eps = Config.DBSCAN_EPS
        self.dbscan_min_samples = Config.DBSCAN_MIN_SAMPLES
        self.min_cluster_points = Config.MIN_CLUSTER_POINTS
        self.max_cluster_points = Config.MAX_CLUSTER_POINTS

    def remove_ground_plane(self, points):
        """
        Applies RANSAC to fit and remove the ground plane.

        Args:
            points (np.ndarray): (N, 4) array of points (x, y, z, intensity).

        Returns:
            np.ndarray: Filtered points (outliers to the ground plane).
        """
        if len(points) < 10:
            return points

        # We use a subset of points for RANSAC fitting to speed up the process
        n_points = points.shape[0]
        sample_size = min(
            n_points, 2000
        )  # 2000 points is usually sufficient for plane fitting

        # Randomly sample points
        rng = np.random.default_rng(Config.RANDOM_SEED)
        indices = rng.choice(n_points, sample_size, replace=False)
        sample_points = points[indices, :3]  # Use x, y, z

        best_inliers_count = -1
        best_plane = None  # (normal, d)

        # RANSAC Loop
        for _ in range(self.ransac_iterations):
            # 1. Pick 3 random points
            # We pick from the sample_points
            idx = rng.choice(sample_size, 3, replace=False)
            p1, p2, p3 = sample_points[idx]

            # 2. Compute Normal Vector
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_mag = np.linalg.norm(normal)

            if norm_mag < 1e-6:
                continue

            normal = normal / norm_mag

            # Constraint: Ground is roughly horizontal. Normal should be close to (0, 0, 1).
            # We check the Z component of the normal.
            if abs(normal[2]) < 0.8:
                continue

            # 3. Compute d
            # Plane: ax + by + cz + d = 0  =>  n . p + d = 0  =>  d = - n . p
            d = -np.dot(normal, p1)

            # 4. Count Inliers in the sample set
            # Distance = | n . p + d |
            dists = np.abs(np.dot(sample_points, normal) + d)
            inliers_count = np.sum(dists < self.ransac_dist_thresh)

            if inliers_count > best_inliers_count:
                best_inliers_count = inliers_count
                best_plane = (normal, d)

        # If no good plane found, return all points (or fallback to height threshold)
        if best_plane is None:
            # Fallback: simple height threshold assuming sensor at ~1.8m
            # Ground usually < -1.0 depending on coordinate system
            return points

        # Filter ALL points using the best plane
        normal, d = best_plane
        all_dists = np.abs(np.dot(points[:, :3], normal) + d)

        # Keep points that are NOT ground (distance > threshold)
        non_ground_mask = all_dists >= self.ransac_dist_thresh
        return points[non_ground_mask]

    def cluster_points(self, points):
        """
        Groups points into clusters using DBSCAN.

        Args:
            points (np.ndarray): (N, 4) array of non-ground points.

        Returns:
            list: List of np.ndarray, each containing points for a cluster.
        """
        if len(points) == 0:
            return []

        # Clustering on spatial coordinates (x, y, z)
        spatial_points = points[:, :3]

        # Run DBSCAN
        # n_jobs=-1 uses all available cores
        clustering = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, n_jobs=-1
        ).fit(spatial_points)

        labels = clustering.labels_
        unique_labels = set(labels)

        clusters = []
        for label in unique_labels:
            if label == -1:
                continue  # Noise points

            # Extract points for this cluster
            mask = labels == label
            cluster_pts = points[mask]

            # Filter by point count
            if self.min_cluster_points <= len(cluster_pts) <= self.max_cluster_points:
                clusters.append(cluster_pts)

        return clusters

    def compute_pca_box(self, points):
        """
        Computes the oriented bounding box using PCA on the XY plane.

        Args:
            points (np.ndarray): (N, 4) array of cluster points.

        Returns:
            np.ndarray: [x, y, z, w, l, h, yaw]
        """
        if len(points) < 3:
            return None

        # Project to XY for orientation
        points_xy = points[:, :2]

        # PCA
        try:
            pca = PCA(n_components=2)
            pca.fit(points_xy)
        except Exception:
            return None

        # Principal axes
        # components_[0] is the axis of max variance (Length)
        # components_[1] is the axis of second max variance (Width)
        v_len = pca.components_[0]

        # Yaw (angle of length axis w.r.t X-axis)
        yaw = np.arctan2(v_len[1], v_len[0])

        # Transform points to local PCA frame to get dimensions
        points_local = pca.transform(points_xy)

        min_xy = np.min(points_local, axis=0)
        max_xy = np.max(points_local, axis=0)

        length = max_xy[0] - min_xy[0]
        width = max_xy[1] - min_xy[1]

        # Center in local frame
        center_local = (min_xy + max_xy) / 2.0

        # Transform center back to world XY
        center_world_xy = pca.inverse_transform(center_local)

        # Z dimension (Height) - Axis Aligned
        z_vals = points[:, 2]
        z_min = np.min(z_vals)
        z_max = np.max(z_vals)
        height = z_max - z_min
        center_z = z_min + height / 2.0

        # Construct box
        # Format: center_x, center_y, center_z, width, length, height, yaw
        box = np.array(
            [
                center_world_xy[0],
                center_world_xy[1],
                center_z,
                width,
                length,
                height,
                yaw,
            ],
            dtype=np.float32,
        )

        return box

    def process_lidar_file(self, lidar_path, transform_matrix=None):
        """
        Runs the proposal generation pipeline on a single LIDAR file.

        Args:
            lidar_path (str): Relative path to the .bin file.
            transform_matrix (np.ndarray, optional): 4x4 matrix to transform points to World Frame.

        Returns:
            list: List of dictionaries {'box': np.array, 'points': np.array}
        """
        # 1. Load Data
        points = load_lidar(lidar_path)
        if len(points) == 0:
            return []

        # 1b. Apply Transformation (Sensor -> World)
        if transform_matrix is not None:
            from library.utils import transform_points

            points = transform_points(points, transform_matrix)

        # 2. Remove Ground
        non_ground_points = self.remove_ground_plane(points)

        # 3. Cluster
        cluster_list = self.cluster_points(non_ground_points)

        # 4. Compute Boxes
        proposals = []
        for pts in cluster_list:
            box = self.compute_pca_box(pts)
            if box is not None:
                proposals.append({"box": box, "points": pts})

        return proposals
