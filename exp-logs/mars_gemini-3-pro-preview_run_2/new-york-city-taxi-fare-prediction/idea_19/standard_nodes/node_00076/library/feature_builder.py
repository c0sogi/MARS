import pandas as pd
import numpy as np
from library.config import ProjectConfig
from library.utils import (
    clamp_coordinates,
    haversine_distance,
    manhattan_distance,
    bearing,
    rotate_coordinates,
    calculate_geohash,
)
from library.stats_manager import StatsManager


class FeatureBuilder:
    def __init__(self):
        self.config = ProjectConfig
        self.stats_manager = StatsManager()

    def add_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes and adds geometric and temporal features to the dataframe.

        Features added:
        - Temporal: Hour, Year, Month, Day, Weekday
        - Spatial: Haversine Distance, Manhattan Distance, Bearing
        - Rotated Coordinates: Aligned with NYC street grid
        - Hierarchical Spatial Indices: Geohashes (L5, L6, L7)
        """
        # Work on a copy to prevent side effects on the original dataframe
        df = df.copy()

        # 1. Temporal Features
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        dt = df["pickup_datetime"].dt
        df["hour"] = dt.hour.astype("int32")
        df["year"] = dt.year.astype("int32")
        df["month"] = dt.month.astype("int32")
        df["day"] = dt.day.astype("int32")
        df["weekday"] = dt.dayofweek.astype("int32")

        # 2. Coordinate Extraction (Ensure float precision)
        plat = df["pickup_latitude"].values.astype(np.float32)
        plon = df["pickup_longitude"].values.astype(np.float32)
        dlat = df["dropoff_latitude"].values.astype(np.float32)
        dlon = df["dropoff_longitude"].values.astype(np.float32)

        # 3. Distance Features
        df["distance_haversine"] = haversine_distance(plat, plon, dlat, dlon).astype(
            "float32"
        )
        df["distance_manhattan"] = manhattan_distance(plat, plon, dlat, dlon).astype(
            "float32"
        )

        # 4. Bearing
        df["bearing"] = bearing(plat, plon, dlat, dlon).astype("float32")

        # 5. Rotated Coordinates (Grid Alignment)
        plat_rot, plon_rot = rotate_coordinates(plat, plon)
        dlat_rot, dlon_rot = rotate_coordinates(dlat, dlon)

        df["pickup_latitude_rot"] = plat_rot.astype("float32")
        df["pickup_longitude_rot"] = plon_rot.astype("float32")
        df["dropoff_latitude_rot"] = dlat_rot.astype("float32")
        df["dropoff_longitude_rot"] = dlon_rot.astype("float32")

        # 6. Geohashes (Hierarchical Spatial Indexing)
        for l in self.config.GEOHASH_LEVELS:
            df[f"geohash_{l}"] = calculate_geohash(plat, plon, l)

        return df

    def enrich_with_stats(self, df: pd.DataFrame, stats: dict = None) -> pd.DataFrame:
        """
        Enriches the dataframe with Hierarchical Distributional Priors (Mean, Std, Count).

        This method delegates to StatsManager to ensure strict Dual-Hygiene:
        - Training (Learner Set): Uses K-Fold Vectorized Subtraction (Global - Fold) to prevent leakage.
        - Inference (Test/Val): Uses Global Wisdom Stats directly.

        Args:
            df (pd.DataFrame): Input dataframe.
            stats (dict, optional): Pre-computed global statistics. If None, they are loaded from cache.

        Returns:
            pd.DataFrame: Dataframe enriched with statistical features.
        """
        # If stats are not provided, load them from cache or compute them from the Wisdom Set
        if stats is None:
            stats = self.stats_manager.compute_global_moments(load_cached=True)

        # Delegate to StatsManager.compute_kfold_moments
        # This function automatically detects if 'fare_amount' and 'fold' are present
        # to switch between Training mode (Subtraction) and Inference mode (Global).
        return self.stats_manager.compute_kfold_moments(df, stats)
