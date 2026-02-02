import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering including physics-informed augmentation,
    standardization, and manifold cluster augmentation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = MiniBatchKMeans(
            n_clusters=Config.N_CLUSTERS,
            random_state=Config.SEED,
            batch_size=4096,
            n_init="auto",
        )
        self.continuous_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        # Will be populated during fit
        self.all_continuous_cols = []
        self.binary_cols = []

    def _compute_physics_features(self, df):
        """
        Computes physics-informed features.
        Returns a DataFrame with original continuous + new physics features.
        """
        # Working on a copy to avoid SettingWithCopy warnings on original df
        df_eng = df[self.continuous_cols].copy()

        # 1. Cyclical Aspect
        # Convert degrees to radians
        aspect_rad = df["Aspect"] * np.pi / 180.0
        df_eng["Aspect_Sin"] = np.sin(aspect_rad)
        df_eng["Aspect_Cos"] = np.cos(aspect_rad)

        # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
        # sqrt(H^2 + V^2)
        df_eng["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

        # 3. Directional Preservation (Absolute Hydrology Elevation)
        # Elevation - Vertical_Distance
        df_eng["Abs_Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

        # 4. Global Context (Mean Distance to Amenities)
        # Mean of (Hydrology, Roadways, Fire Points)
        # Note: Using Horizontal distance for Hydrology
        df_eng["Mean_Distance_To_Amenities"] = (
            df["Horizontal_Distance_To_Hydrology"]
            + df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Fire_Points"]
        ) / 3.0

        return df_eng

    def fit(self, train_df):
        """
        Fits the scaler and KMeans model on the training data.
        """
        # Identify binary columns (Soil Types and Wilderness Areas)
        # We assume any column starting with Soil_Type or Wilderness_Area is binary
        self.binary_cols = [
            c
            for c in train_df.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]

        # Compute physics features for training set
        train_cont = self._compute_physics_features(train_df)
        self.all_continuous_cols = train_cont.columns.tolist()

        # Fit Scaler
        print("Fitting StandardScaler...")
        self.scaler.fit(train_cont)

        # Transform for KMeans fitting
        train_scaled = self.scaler.transform(train_cont)

        # Fit KMeans (Manifold Cluster Augmentation)
        print(f"Fitting MiniBatchKMeans (K={Config.N_CLUSTERS})...")
        self.kmeans.fit(train_scaled)

    def transform(self, df):
        """
        Applies transformations to the data.
        Returns a numpy array of shape (N, Features).
        """
        # 1. Compute Physics Features
        df_cont = self._compute_physics_features(df)

        # 2. Standardize
        X_scaled = self.scaler.transform(df_cont)

        # 3. Manifold Cluster Augmentation (Distances to Centroids)
        # transform() returns distance to each cluster center
        X_cluster_dists = self.kmeans.transform(X_scaled)

        # 4. Get Binary Features
        X_binary = df[self.binary_cols].values.astype(np.float32)

        # 5. Concatenate: [Scaled Continuous, Cluster Distances, Binary]
        X_final = np.hstack([X_scaled, X_cluster_dists, X_binary])

        return X_final.astype(np.float32)


class CoverTypeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def process_data(load_cached_data=True):
    """
    Orchestrates data loading, feature engineering, and caching.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_X = np.load(files["train_X"])
        train_y = np.load(files["train_y"])
        val_X = np.load(files["val_X"])
        val_y = np.load(files["val_y"])
        test_X = np.load(files["test_X"])
        test_ids = np.load(files["test_ids"])
        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load Parquet Metadata
    print(f"Loading {Config.TRAIN_DATA_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    print(f"Loading {Config.VAL_DATA_PATH}...")
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    print(f"Loading {Config.TEST_DATA_PATH}...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Initialize and Fit Feature Engineer
    fe = FeatureEngineer()
    fe.fit(df_train)

    # Transform Data
    print("Transforming Train...")
    train_X = fe.transform(df_train)
    print("Transforming Val...")
    val_X = fe.transform(df_val)
    print("Transforming Test...")
    test_X = fe.transform(df_test)

    # Extract Targets and IDs
    # Targets: Shift from 1-7 to 0-6
    train_y = (df_train["Cover_Type"].values - 1).astype(np.int64)
    val_y = (df_val["Cover_Type"].values - 1).astype(np.int64)
    test_ids = df_test["Id"].values.astype(np.int64)

    # Save to Cache
    print(f"Saving to {cache_dir}...")
    np.save(files["train_X"], train_X)
    np.save(files["train_y"], train_y)
    np.save(files["val_X"], val_X)
    np.save(files["val_y"], val_y)
    np.save(files["test_X"], test_X)
    np.save(files["test_ids"], test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test sets.
    """
    # Load processed data
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(load_cached_data)

    # Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X)  # No targets for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Return input dimension for model initialization
    input_dim = train_X.shape[1]

    print(f"Data Loaders Ready.")
    print(f"  Input Features: {input_dim}")
    print(f"  Train Size: {len(train_dataset)}")
    print(f"  Val Size: {len(val_dataset)}")
    print(f"  Test Size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, test_ids, input_dim
