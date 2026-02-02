import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from library.config import Config
from library.utils import set_seed, haversine_distance, rotate_coordinates


class SpatialClusterer:
    """
    Wraps MiniBatchKMeans to learn spatial neighborhoods from coordinate data.
    Assigns discrete cluster IDs to pickup and dropoff locations.
    """

    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        # n_init='auto' is default in newer sklearn, explicit here for clarity if needed
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=10000,
            n_init="auto",
        )

    def fit(self, df):
        """
        Fits the KMeans model on stacked pickup and dropoff coordinates
        to learn a shared map of neighborhoods.
        """
        # Stack coordinates to treat pickup and dropoff locations as points in the same space
        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        # Remove NaNs if any (though dataset analysis showed 0 NaNs in lat/lon for train)
        # Using a safe approach just in case
        X = np.vstack([pickup_coords, dropoff_coords])
        X = X[~np.isnan(X).any(axis=1)]

        # Optimization: Sample data for KMeans fitting to save time on massive datasets
        # We don't need 88M points to find 100 centroids.
        MAX_SAMPLES = 500_000
        if len(X) > MAX_SAMPLES:
            indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
            X = X[indices]

        self.model.fit(X)
        return self

    def transform(self, df):
        """
        Predicts cluster IDs for pickup and dropoff locations.
        """
        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].fillna(0).values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].fillna(0).values

        df["pickup_cluster"] = self.model.predict(pickup_coords)
        df["dropoff_cluster"] = self.model.predict(dropoff_coords)
        return df

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self


def add_time_features(df):
    """
    Extracts temporal features from pickup_datetime.
    """
    if not np.issubdtype(df[Config.DATETIME_COL].dtype, np.datetime64):
        df[Config.DATETIME_COL] = pd.to_datetime(df[Config.DATETIME_COL], utc=True)

    dt = df[Config.DATETIME_COL].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["year"] = dt.year
    return df


def add_rotated_features(df):
    """
    Adds coordinates rotated by the specified angle to align with Manhattan grid.
    """
    p_rot_x, p_rot_y = rotate_coordinates(
        df["pickup_longitude"], df["pickup_latitude"], Config.ROTATION_ANGLE
    )
    d_rot_x, d_rot_y = rotate_coordinates(
        df["dropoff_longitude"], df["dropoff_latitude"], Config.ROTATION_ANGLE
    )

    df["pickup_rot_x"] = p_rot_x
    df["pickup_rot_y"] = p_rot_y
    df["dropoff_rot_x"] = d_rot_x
    df["dropoff_rot_y"] = d_rot_y
    return df


def add_landmark_features(df):
    """
    Adds Haversine distances to specific landmarks defined in Config.
    """
    for name, (lat, lon) in Config.LANDMARKS.items():
        df[f"dist_pickup_{name}"] = haversine_distance(
            df["pickup_latitude"], df["pickup_longitude"], lat, lon
        )
        df[f"dist_dropoff_{name}"] = haversine_distance(
            df["dropoff_latitude"], df["dropoff_longitude"], lat, lon
        )
    return df


def process_data(mode="train", load_cached_data=True):
    """
    Main pipeline for data processing.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    set_seed()

    # Define paths based on mode
    if mode == "train":
        input_path = Config.TRAIN_PATH
        output_path = Config.TRAIN_PROCESSED_PATH
    elif mode == "val":
        input_path = Config.VAL_PATH
        output_path = Config.VAL_PROCESSED_PATH
    elif mode == "test":
        input_path = Config.TEST_PATH
        output_path = Config.TEST_PROCESSED_PATH
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached {mode} data from {output_path}")
        return pd.read_parquet(output_path)

    # 2. Process from Scratch
    print(f"Processing {mode} data from {input_path}...")

    # Load raw data
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Failed to read parquet, trying CSV fallback if needed (Error: {e})")
        # Fallback logic usually not needed given metadata generation, but good for safety
        raise e

    # Debug Sampling
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        if len(df) > Config.DEBUG_SAMPLE_SIZE:
            df = df.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

    # Filter Training Data
    if mode == "train":
        initial_count = len(df)
        df = df[
            (df[Config.TARGET_COL] >= Config.FARE_MIN)
            & (df[Config.TARGET_COL] <= Config.FARE_MAX)
        ]
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            print(
                f"Filtered {dropped_count} rows with fare_amount outside [{Config.FARE_MIN}, {Config.FARE_MAX}]"
            )

    # Feature Engineering
    # A. Time
    df = add_time_features(df)

    # B. Basic Distance
    df["haversine_dist"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # C. Rotated Coordinates
    df = add_rotated_features(df)

    # D. Landmark Distances
    df = add_landmark_features(df)

    # E. Spatial Clustering
    clusterer = SpatialClusterer(n_clusters=Config.N_CLUSTERS, random_state=Config.SEED)

    if mode == "train":
        print("Fitting Spatial Clusterer on training data...")
        clusterer.fit(df)
        print(f"Saving Spatial Clusterer to {Config.KMEANS_MODEL_PATH}")
        clusterer.save(Config.KMEANS_MODEL_PATH)
    else:
        # For val/test, we must use the pre-trained clusterer
        if not os.path.exists(Config.KMEANS_MODEL_PATH):
            raise FileNotFoundError(
                f"Clusterer model not found at {Config.KMEANS_MODEL_PATH}. "
                "Please process 'train' data first to generate the model."
            )
        print("Loading Spatial Clusterer...")
        clusterer.load(Config.KMEANS_MODEL_PATH)

    print("Assigning Cluster IDs...")
    df = clusterer.transform(df)

    # Save to Cache
    print(f"Saving processed data to {output_path}")
    df.to_parquet(output_path, index=False)

    return df
