import os
import numpy as np
import pandas as pd
import json
from library.config import Config
from library.utils import read_points


class DataProcessor:
    """
    Handles data loading, temporal aggregation (multi-sweep), and coordinate transformations
    for the NuScenes-like dataset.
    """

    def __init__(self, load_cached_data=True):
        self.cache_dir = Config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "sweeps_metadata.parquet")

        # Load or Generate Metadata
        if load_cached_data and os.path.exists(self.cache_path):
            self.sweeps_df = pd.read_parquet(self.cache_path)
        else:
            self.sweeps_df = self._generate_metadata()
            self.sweeps_df.to_parquet(self.cache_path)

        # Optimize access: Convert to dictionary for O(1) retrieval
        # Key: sample_token, Value: list of records (dicts)
        # Sort by sweep_idx to ensure 0 (current) is first
        self.sweeps_df = self.sweeps_df.sort_values(["sample_token", "sweep_idx"])
        self.data_map = {
            k: g.to_dict("records") for k, g in self.sweeps_df.groupby("sample_token")
        }

    def _get_transform_matrix(self, translation, rotation):
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

    def _process_split(self, data_dir, lidar_dir_name):
        """
        Parses JSON tables for a specific data split (train/test) and generates sweep records.
        """
        # Load JSON tables
        # We assume files exist based on dataset description
        with open(os.path.join(data_dir, "sample.json"), "r") as f:
            samples = json.load(f)
        with open(os.path.join(data_dir, "sample_data.json"), "r") as f:
            sample_data = json.load(f)
        with open(os.path.join(data_dir, "ego_pose.json"), "r") as f:
            ego_pose = json.load(f)
        with open(os.path.join(data_dir, "calibrated_sensor.json"), "r") as f:
            calibrated_sensor = json.load(f)

        # Create lookup indices
        sd_by_token = {x["token"]: x for x in sample_data}
        ep_by_token = {x["token"]: x for x in ego_pose}
        cs_by_token = {x["token"]: x for x in calibrated_sensor}

        # Map sample_token -> LIDAR_TOP sample_data
        # We filter for LIDAR_TOP to ensure we get the primary point cloud
        sample_to_lidar = {}
        for sd in sample_data:
            if "LIDAR_TOP" in sd["filename"]:
                sample_to_lidar[sd["sample_token"]] = sd

        records = []

        # Iterate over all samples in this split
        for s in samples:
            s_token = s["token"]
            if s_token not in sample_to_lidar:
                continue

            curr_sd = sample_to_lidar[s_token]

            # Traverse backwards to find previous sweeps
            next_sd = curr_sd

            for i in range(Config.NUM_SWEEPS):
                # Retrieve Pose Information
                # Note: sample_data points to calibrated_sensor and ego_pose
                if next_sd["calibrated_sensor_token"] not in cs_by_token:
                    break
                if next_sd["ego_pose_token"] not in ep_by_token:
                    break

                cs = cs_by_token[next_sd["calibrated_sensor_token"]]
                ep = ep_by_token[next_sd["ego_pose_token"]]

                # Calculate Global Pose of the Sensor at this timestamp
                # T_global_sensor = T_global_ego @ T_ego_sensor
                T_ego_sensor = self._get_transform_matrix(
                    cs["translation"], cs["rotation"]
                )
                T_global_ego = self._get_transform_matrix(
                    ep["translation"], ep["rotation"]
                )
                T_global_sensor = T_global_ego @ T_ego_sensor

                # Construct File Path
                # Filename in JSON is like 'samples/LIDAR_TOP/file.bin'
                # We map this to 'train_lidar/file.bin' or 'test_lidar/file.bin'
                fname = os.path.basename(next_sd["filename"])
                full_path = os.path.join(lidar_dir_name, fname)

                records.append(
                    {
                        "sample_token": s_token,
                        "sweep_idx": i,
                        "lidar_path": full_path,
                        "timestamp": next_sd["timestamp"],
                        "pose": T_global_sensor.flatten().tolist(),
                    }
                )

                # Move to previous sweep
                prev_token = next_sd["prev"]
                if not prev_token or prev_token not in sd_by_token:
                    break
                next_sd = sd_by_token[prev_token]

        return records

    def _generate_metadata(self):
        """
        Generates the full metadata dataframe for both train and test sets.
        """
        train_records = self._process_split(Config.TRAIN_DATA_DIR, "train_lidar")
        test_records = self._process_split(Config.TEST_DATA_DIR, "test_lidar")

        df = pd.DataFrame(train_records + test_records)
        return df

    def get_lidar_data(self, sample_token):
        """
        Retrieves and aggregates Lidar data for a given sample token.
        Performs multi-sweep aggregation and coordinate transformation.

        Args:
            sample_token (str): The sample identifier.

        Returns:
            points (np.ndarray): (N, 5) array [x, y, z, intensity, time_lag]
        """
        if sample_token not in self.data_map:
            # Fallback or error
            return np.zeros((0, 5), dtype=np.float32)

        sweeps = self.data_map[sample_token]
        # sweeps[0] is the current frame (sweep_idx=0)

        ref_sweep = sweeps[0]
        ref_pose = np.array(ref_sweep["pose"]).reshape(4, 4)
        ref_time = ref_sweep["timestamp"]

        # Compute inverse of reference pose to transform global points back to current frame
        ref_pose_inv = np.linalg.inv(ref_pose)

        all_points = []

        for sweep in sweeps:
            # Construct full path
            file_path = os.path.join(Config.INPUT_DIR, sweep["lidar_path"])

            try:
                # Load points (N, 4) or (N, 5)
                points = read_points(file_path)

                # Extract XYZ and Intensity
                xyz = points[:, :3]
                intensity = points[:, 3:4]  # Keep as (N, 1)

                # Transform points to Reference Frame (Current)
                # P_ref = T_ref_inv @ T_sweep @ P_sweep
                curr_pose = np.array(sweep["pose"]).reshape(4, 4)

                # Combined transform matrix
                T_rel = ref_pose_inv @ curr_pose

                # Homogeneous coordinates
                xyz1 = np.hstack([xyz, np.ones((xyz.shape[0], 1))])

                # Apply transform
                xyz_trans = (T_rel @ xyz1.T).T
                xyz_final = xyz_trans[:, :3]

                # Calculate Time Lag (seconds)
                # Timestamps are in microseconds
                time_lag = (ref_time - sweep["timestamp"]) / 1e6
                time_lag_arr = np.full((xyz_final.shape[0], 1), time_lag)

                # Concatenate features: [x, y, z, intensity, dt]
                points_aug = np.hstack([xyz_final, intensity, time_lag_arr])
                all_points.append(points_aug)

            except FileNotFoundError:
                # Skip missing files (robustness)
                continue

        if not all_points:
            return np.zeros((0, 5), dtype=np.float32)

        # Stack all sweeps
        aggregated_points = np.vstack(all_points).astype(np.float32)

        return aggregated_points
