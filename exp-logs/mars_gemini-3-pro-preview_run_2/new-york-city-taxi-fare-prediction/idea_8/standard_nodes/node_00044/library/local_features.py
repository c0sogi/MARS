import os
import numpy as np
import pandas as pd
from library.config import CACHE_DIR
from library.utils import haversine_distance, manhattan_distance


class LocalFeatureGenerator:
    """
    Generates row-level local features including temporal and geometric attributes.
    """

    def __init__(self):
        pass

    def _extract_time_features(self, df):
        """
        Extracts temporal features from pickup_datetime.
        """
        # Ensure datetime format.
        # The data format is like "2014-03-30 12:14:00 UTC".
        # We strip UTC to speed up parsing if necessary, or let pandas handle it.
        # Coercing errors to NaT to handle potential garbage.

        # Working on a copy of the series to avoid SettingWithCopy warnings on the original DF if it's a view
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            # Fast parsing for standard format
            temp_dt = pd.to_datetime(
                df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC", errors="coerce"
            )

            # Fallback for mixed formats if fast parsing fails significantly (though metadata suggests consistency)
            if temp_dt.isnull().all() and not df["pickup_datetime"].empty:
                temp_dt = pd.to_datetime(
                    df["pickup_datetime"], infer_datetime_format=True, errors="coerce"
                )
        else:
            temp_dt = df["pickup_datetime"]

        df["hour"] = temp_dt.dt.hour
        df["year"] = temp_dt.dt.year
        df["month"] = temp_dt.dt.month
        df["day"] = temp_dt.dt.day
        df["weekday"] = temp_dt.dt.dayofweek

        return df

    def _extract_geo_features(self, df):
        """
        Calculates geometric features based on pickup and dropoff coordinates.
        """
        # Basic distances
        df["dist_haversine"] = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        df["dist_manhattan"] = manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )

        # Coordinate differences
        df["abs_diff_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
        df["abs_diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

        # Rotated coordinates (approximate 45 degree rotation for NYC grid)
        # u = x + y, v = x - y
        df["pickup_rot_sum"] = df["pickup_latitude"] + df["pickup_longitude"]
        df["pickup_rot_diff"] = df["pickup_latitude"] - df["pickup_longitude"]
        df["dropoff_rot_sum"] = df["dropoff_latitude"] + df["dropoff_longitude"]
        df["dropoff_rot_diff"] = df["dropoff_latitude"] - df["dropoff_longitude"]

        return df

    def process(self, df):
        """
        Main processing method to generate all local features.

        Args:
            df (pd.DataFrame): Input dataframe with raw features.

        Returns:
            pd.DataFrame: Dataframe with added local features.
        """
        # Create a copy to avoid modifying the original dataframe in place
        df_out = df.copy()

        # Extract features
        df_out = self._extract_time_features(df_out)
        df_out = self._extract_geo_features(df_out)

        # Select relevant columns to return
        # We keep the key for joining, and the new numeric features.
        # We also keep the original coordinates as they are useful for the model.
        # We drop pickup_datetime as it is now decomposed.

        cols_to_keep = [
            "key",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "hour",
            "year",
            "month",
            "day",
            "weekday",
            "dist_haversine",
            "dist_manhattan",
            "abs_diff_lon",
            "abs_diff_lat",
            "pickup_rot_sum",
            "pickup_rot_diff",
            "dropoff_rot_sum",
            "dropoff_rot_diff",
        ]

        # If target exists, keep it
        if "fare_amount" in df_out.columns:
            cols_to_keep.append("fare_amount")

        return df_out[cols_to_keep]


def generate_local_features(df, dataset_name, load_cached_data=True):
    """
    Orchestrates the generation and caching of local features.

    Args:
        df (pd.DataFrame): Input dataframe.
        dataset_name (str): Name of the dataset (e.g., 'train', 'val', 'test') for cache file naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Dataframe containing local features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(CACHE_DIR, f"local_features_{dataset_name}.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached local features for {dataset_name} from {cache_path}...")
        try:
            cached_df = pd.read_parquet(cache_path)
            # Basic validation: check if length matches
            if len(cached_df) == len(df):
                return cached_df
            else:
                print(
                    f"Cache mismatch for {dataset_name} (Expected {len(df)} rows, got {len(cached_df)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache for {dataset_name}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Computing local features for {dataset_name}...")
    generator = LocalFeatureGenerator()
    processed_df = generator.process(df)

    # 3. Save to cache
    print(f"Saving local features for {dataset_name} to {cache_path}...")
    processed_df.to_parquet(cache_path, index=False)

    return processed_df
