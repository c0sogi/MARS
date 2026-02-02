import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from library.config import Config
from library.utils import log_transform


class SpatialClusterer:
    """
    Wrapper for MiniBatchKMeans to handle spatial clustering of coordinates.
    Saves/loads cluster centers via NPY to avoid pickle constraints.
    """

    def __init__(self, n_clusters=100, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Fit KMeans on coordinates.
        X: numpy array of shape (N, 2) representing [lat, lon]
        """
        # Using MiniBatchKMeans for efficiency on large datasets
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=10000,
            n_init=3,
        )
        kmeans.fit(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        return self

    def predict(self, X):
        """
        Assign cluster IDs to coordinates based on nearest center.
        X: numpy array of shape (N, 2)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted or loaded.")
        # Efficient nearest neighbor search using sklearn metrics to avoid full model loading
        labels = pairwise_distances_argmin(X, self.cluster_centers_)
        return labels

    def save(self, filepath):
        if self.cluster_centers_ is None:
            raise ValueError("No cluster centers to save.")
        np.save(filepath, self.cluster_centers_)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cluster centers file not found: {filepath}")
        self.cluster_centers_ = np.load(filepath)


def clamp_coordinates(df):
    """
    Clamp coordinate columns to the bounding box defined in Config.
    This prevents linear extrapolation errors from extreme outliers.
    """
    df["pickup_latitude"] = df["pickup_latitude"].clip(
        Config.BB_LAT_MIN, Config.BB_LAT_MAX
    )
    df["pickup_longitude"] = df["pickup_longitude"].clip(
        Config.BB_LON_MIN, Config.BB_LON_MAX
    )
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(
        Config.BB_LAT_MIN, Config.BB_LAT_MAX
    )
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(
        Config.BB_LON_MIN, Config.BB_LON_MAX
    )
    return df


def calculate_haversine(df):
    """
    Calculate Haversine distance between pickup and dropoff.
    """
    R = 6371  # Earth radius in km

    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_manhattan(df):
    """
    Calculate Manhattan distance (L1 norm) approximation.
    Useful for city grid navigation.
    """
    R = 6371
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(lon2 - lon1)

    # Approximation: average latitude for longitude distance
    avg_lat = (lat1 + lat2) / 2

    dist_lat = dlat * R
    dist_lon = dlon * R * np.cos(avg_lat)

    return dist_lat + dist_lon


def calculate_bearing(df):
    """
    Calculate bearing (direction) from pickup to dropoff.
    """
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlon = lon2 - lon1

    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360


def extract_time_features(df):
    """
    Extract temporal features from pickup_datetime.
    """
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        # Clean potential ' UTC' suffix if present (common in this dataset)
        if df["pickup_datetime"].dtype == "object":
            df["pickup_datetime"] = (
                df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
            )
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.dayofweek

    # Drop original datetime to save memory
    df = df.drop(columns=["pickup_datetime"])
    return df


def process_data(load_cached_data=True):
    """
    Main function to load, process, and cache data.
    Implements strict caching logic using Parquet and NPY.
    """
    # Define cache paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    cluster_cache = os.path.join(Config.WORKING_DIR, "cluster_centers.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(cluster_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading cached processed data...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Processing data from scratch...")

    # Load raw metadata
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # 1. Coordinate Clamping
    print("Clamping coordinates...")
    train_df = clamp_coordinates(train_df)
    val_df = clamp_coordinates(val_df)
    test_df = clamp_coordinates(test_df)

    # 2. Spatial Clustering
    print("Fitting spatial clusterer...")
    clusterer = SpatialClusterer(
        n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_SEED
    )

    # Prepare training coordinates for fitting (stack pickup and dropoff to learn map)
    coords_train = np.vstack(
        [
            train_df[["pickup_latitude", "pickup_longitude"]].values,
            train_df[["dropoff_latitude", "dropoff_longitude"]].values,
        ]
    )

    clusterer.fit(coords_train)
    clusterer.save(cluster_cache)

    print("Applying spatial clustering...")
    for df in [train_df, val_df, test_df]:
        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        df["pickup_cluster"] = clusterer.predict(pickup_coords)
        df["dropoff_cluster"] = clusterer.predict(dropoff_coords)

    # 3. Physics Features
    print("Calculating physics features...")
    for df in [train_df, val_df, test_df]:
        df["haversine_dist"] = calculate_haversine(df)
        df["manhattan_dist"] = calculate_manhattan(df)
        df["bearing"] = calculate_bearing(df)

    # 4. Time Features
    print("Extracting time features...")
    train_df = extract_time_features(train_df)
    val_df = extract_time_features(val_df)
    test_df = extract_time_features(test_df)

    # 5. Target Transformation
    if Config.USE_LOG_TARGET:
        print("Applying log transform to target...")
        if "fare_amount" in train_df.columns:
            train_df["fare_amount"] = log_transform(train_df["fare_amount"].values)
        if "fare_amount" in val_df.columns:
            val_df["fare_amount"] = log_transform(val_df["fare_amount"].values)

    # 6. Save to Cache
    print("Saving processed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df
