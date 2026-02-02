import numpy as np
import pandas as pd
from library.utils import (
    clamp_coordinates,
    haversine_array,
    manhattan_distance,
    rotate_coordinates,
)


class FeatureEngineer:
    """
    Handles local, row-level feature engineering for the taxi fare prediction task.

    This class transforms raw input data (timestamps and coordinates) into
    machine-learning-ready features, including:
    1. Temporal features: Hour, Year, Day of Week.
    2. Spatial features: Haversine distance, Manhattan distance, Coordinate differences.
    3. Rotated coordinates: To help tree-based models capture diagonal boundaries.
    """

    def __init__(self, rotation_angle=45, clamp_input=False):
        """
        Initialize the FeatureEngineer.

        Args:
            rotation_angle (float): The angle (in degrees) to rotate coordinates.
                                    Default is 45 degrees.
            clamp_input (bool): If True, clamps coordinates to the NYC bounding box
                                defined in the configuration before processing.
        """
        self.rotation_angle = rotation_angle
        self.clamp_input = clamp_input

    def transform(self, df):
        """
        Applies feature engineering transformations to the input DataFrame.

        Args:
            df (pd.DataFrame): Input dataframe containing raw features
                               (pickup_datetime, pickup/dropoff coordinates).

        Returns:
            pd.DataFrame: A new DataFrame with additional engineered features.
        """
        # Operate on a copy to avoid modifying the original dataframe
        df = df.copy()

        # 1. Coordinate Clamping (Optional)
        # Useful for removing outliers or restricting data to the ROI
        if self.clamp_input:
            df = clamp_coordinates(df)

        # 2. Temporal Feature Extraction
        if "pickup_datetime" in df.columns:
            # Ensure the column is in datetime format
            if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
                df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

            # Extract components
            dt = df["pickup_datetime"].dt
            df["hour"] = dt.hour
            df["year"] = dt.year
            df["dayofweek"] = dt.dayofweek

        # 3. Geometric Feature Generation
        required_coords = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
        ]

        # Only proceed if all coordinate columns are present
        if all(col in df.columns for col in required_coords):
            # Haversine Distance (Great Circle distance in km)
            df["dist_km"] = haversine_array(
                df["pickup_latitude"],
                df["pickup_longitude"],
                df["dropoff_latitude"],
                df["dropoff_longitude"],
            )

            # Manhattan Distance (L1 norm in degrees)
            # Useful for grid-like street networks
            df["dist_manhattan"] = manhattan_distance(
                df["pickup_latitude"],
                df["pickup_longitude"],
                df["dropoff_latitude"],
                df["dropoff_longitude"],
            )

            # Absolute Coordinate Differences
            df["abs_diff_lon"] = (
                df["dropoff_longitude"] - df["pickup_longitude"]
            ).abs()
            df["abs_diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

            # Rotated Coordinates
            # Adds columns like 'pickup_lon_rot', 'pickup_lat_rot', etc.
            df = rotate_coordinates(df, angle=self.rotation_angle)

        return df
