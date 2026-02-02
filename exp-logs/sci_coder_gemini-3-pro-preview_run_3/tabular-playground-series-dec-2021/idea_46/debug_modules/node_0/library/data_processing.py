import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.utils import load_data

# Define feature groups based on dataset description
CONTINUOUS_COLS = [
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

# Binary columns are Wilderness_Area (4) and Soil_Type (40)
BINARY_PREFIXES = ["Wilderness_Area", "Soil_Type"]


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def feature_engineering(df):
    """
    Applies Augmented Physics-Informed Engineering.
    """
    # Avoid modifying original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation (Retaining raw Aspect as per strategy)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)

    # 2. Geometric Magnitude
    # Euclidean distance to hydrology
    h_dist = df["Horizontal_Distance_To_Hydrology"]
    v_dist = df["Vertical_Distance_To_Hydrology"]
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation
    # Absolute Hydrology Elevation
    df["Absolute_Hydrology_Elevation"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )

    # 4. Global Context
    # Mean Distance to Amenities
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0

    return df


def get_dataloaders(batch_size=4096, load_cached_data=True, num_workers=4):
    """
    Loads data, performs feature engineering/preprocessing, and returns DataLoaders.
    Implements caching strategy using .npy files.
    """
    CACHE_DIR = "./working/idea_46/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached processed data from ./working/idea_46/...")
        train_X = np.load(files["train_X"])
        train_y = np.load(files["train_y"])
        val_X = np.load(files["val_X"])
        val_y = np.load(files["val_y"])
        test_X = np.load(files["test_X"])
        test_ids = np.load(files["test_ids"])

    else:
        print("Processing data from scratch...")

        # Load Raw Data
        train_df = load_data("train")
        val_df = load_data("val")
        test_df = load_data("test")

        # Extract Test IDs before processing
        test_ids = test_df["Id"].values

        # Apply Feature Engineering
        train_df = feature_engineering(train_df)
        val_df = feature_engineering(val_df)
        test_df = feature_engineering(test_df)

        # Define Feature Columns
        # Identify binary columns dynamically
        all_cols = train_df.columns
        binary_cols = [
            c for c in all_cols if any(c.startswith(p) for p in BINARY_PREFIXES)
        ]

        # Identify continuous columns (Original + New)
        # We exclude Id, Cover_Type and Binary columns
        exclude_cols = set(["Id", "Cover_Type"] + binary_cols)
        continuous_cols = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Continuous Features: {len(continuous_cols)}")
        print(f"Binary Features: {len(binary_cols)}")

        # Standardization
        # Fit scaler ONLY on Train
        scaler = StandardScaler()
        train_df[continuous_cols] = scaler.fit_transform(
            train_df[continuous_cols].astype(np.float32)
        )

        # Transform Val and Test
        val_df[continuous_cols] = scaler.transform(
            val_df[continuous_cols].astype(np.float32)
        )
        test_df[continuous_cols] = scaler.transform(
            test_df[continuous_cols].astype(np.float32)
        )

        # Prepare Final Arrays
        feature_cols = continuous_cols + binary_cols

        train_X = train_df[feature_cols].values.astype(np.float32)
        val_X = val_df[feature_cols].values.astype(np.float32)
        test_X = test_df[feature_cols].values.astype(np.float32)

        # Prepare Targets (Shift by -1 for 0-indexed CrossEntropy)
        # Cover_Type is 1-7, we need 0-6
        train_y = (train_df["Cover_Type"].values - 1).astype(np.int64)
        val_y = (val_df["Cover_Type"].values - 1).astype(np.int64)

        # Save to Cache
        print("Saving processed data to cache...")
        np.save(files["train_X"], train_X)
        np.save(files["train_y"], train_y)
        np.save(files["val_X"], val_X)
        np.save(files["val_y"], val_y)
        np.save(files["test_X"], test_X)
        np.save(files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X, None)  # No targets for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
