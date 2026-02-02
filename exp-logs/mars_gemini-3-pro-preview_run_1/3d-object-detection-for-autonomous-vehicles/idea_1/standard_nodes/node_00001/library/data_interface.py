import os
import json
import numpy as np
import library.config as config
from library.utils import load_or_compute, get_transform_matrix as build_matrix


class DataInterface:
    """
    Manages dataset relationships and coordinate transformations.
    Handles loading of interlocking JSON tables and computing World-to-LiDAR matrices.
    """

    def __init__(self, load_cached_data=True):
        """
        Initialize the DataInterface.

        Args:
            load_cached_data (bool): If True, attempts to load lookup tables from cache.
                                     If False or cache missing, recomputes tables.
        """
        self.cache_path = os.path.join(config.CACHE_DIR, "data_interface_cache.npy")

        # Load or compute the lookup tables using the utility function
        data = load_or_compute(
            self.cache_path,
            self._build_lookup_tables,
            load_cached_data=load_cached_data,
            use_parquet=False,
        )

        # Unpack the data into instance attributes for O(1) access
        self.sample_to_lidar_sd = data["sample_to_lidar_sd"]
        self.ego_pose = data["ego_pose"]
        self.calibrated_sensor = data["calibrated_sensor"]

    def _load_json_table(self, filename):
        """
        Helper to load and merge a specific JSON table from both train and test directories.
        """
        combined_data = []
        # Directories defined in config
        dirs = [config.TRAIN_DATA_JSON_DIR, config.TEST_DATA_JSON_DIR]

        for d in dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    combined_data.extend(data)
        return combined_data

    def _build_lookup_tables(self):
        """
        Parses raw JSON files and builds optimized index maps.
        Returns a dictionary containing the maps.
        """
        # Load raw data from all sources
        sample_data_list = self._load_json_table("sample_data.json")
        ego_pose_list = self._load_json_table("ego_pose.json")
        calibrated_sensor_list = self._load_json_table("calibrated_sensor.json")

        # 1. Ego Pose Map: token -> record
        ego_pose_map = {r["token"]: r for r in ego_pose_list}

        # 2. Calibrated Sensor Map: token -> record
        cs_map = {r["token"]: r for r in calibrated_sensor_list}

        # 3. Sample Token -> LiDAR Sample Data Record
        # We need to map the sample_token (time step) to the specific sensor record for LiDAR.
        sample_to_lidar = {}
        for sd in sample_data_list:
            # Identify LiDAR records by file extension
            if sd["filename"].lower().endswith((".bin", ".pcd")):
                s_tok = sd["sample_token"]
                # If multiple LiDAR records exist for a sample, this will take the last one encountered.
                # In this dataset structure, there is typically one main LiDAR scan per sample.
                sample_to_lidar[s_tok] = sd

        return {
            "sample_to_lidar_sd": sample_to_lidar,
            "ego_pose": ego_pose_map,
            "calibrated_sensor": cs_map,
        }

    def get_transform_matrix(self, sample_token):
        """
        Computes the World -> LiDAR transformation matrix for a given sample.

        Args:
            sample_token (str): The unique identifier for the sample.

        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix converting
                        World Coordinates to Local LiDAR Coordinates.
        """
        # 1. Retrieve the LiDAR sample_data record associated with this sample
        sd_record = self.sample_to_lidar_sd.get(sample_token)
        if sd_record is None:
            raise KeyError(f"Sample token {sample_token} not found in LiDAR map.")

        # 2. Retrieve associated tokens
        ep_token = sd_record["ego_pose_token"]
        cs_token = sd_record["calibrated_sensor_token"]

        # 3. Retrieve pose and calibration records
        ep_record = self.ego_pose[ep_token]
        cs_record = self.calibrated_sensor[cs_token]

        # 4. Construct Component Matrices
        # T_ego_to_world: Transforms points from Ego frame to World frame
        ego_to_world = build_matrix(ep_record["translation"], ep_record["rotation"])

        # T_sensor_to_ego: Transforms points from Sensor frame to Ego frame
        sensor_to_ego = build_matrix(cs_record["translation"], cs_record["rotation"])

        # 5. Compute Chain
        # T_sensor_to_world = T_ego_to_world @ T_sensor_to_ego
        sensor_to_world = ego_to_world @ sensor_to_ego

        # 6. Invert to get T_world_to_sensor
        world_to_sensor = np.linalg.inv(sensor_to_world)

        return world_to_sensor
