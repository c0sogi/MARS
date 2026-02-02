import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class ForestDataset(Dataset):
    def __init__(self, X, y=None):
        """
        Custom Dataset for Forest Cover Type data.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def engineer_features(df):
    """
    Applies Augmented Physics-Informed Engineering.
    1. Cyclical Augmentation (Aspect Sin/Cos)
    2. Geometric Magnitude (Euclidean Dist to Hydrology)
    3. Directional Preservation (Abs Hydrology Elevation)
    4. Global Context (Mean Dist to Amenities)
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation
    # Convert Aspect (degrees) to radians for trig functions
    # We retain the raw 'Aspect' column as per Lesson 00034
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude
    # Euclidean Distance to Hydrology = sqrt(H^2 + V^2)
    df["Hydrology_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation
    # Absolute Hydrology Elevation = Elevation - Vertical_Dist
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context
    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    amenity_cols = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Amenities"] = df[amenity_cols].mean(axis=1)

    return df


def preprocess_data(load_cached_data=True):
    """
    Loads data, engineers features, standardizes continuous columns,
    and caches the result.
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

    # Check cache
    if load_cached_data:
        if all(os.path.exists(f) for f in files.values()):
            print(f"Loading cached data from {cache_dir}...")
            X_train = np.load(files["train_X"])
            y_train = np.load(files["train_y"])
            X_val = np.load(files["val_X"])
            y_val = np.load(files["val_y"])
            X_test = np.load(files["test_X"])
            test_ids = np.load(files["test_ids"])
            return X_train, y_train, X_val, y_val, X_test, test_ids
        else:
            print("Cache missing or incomplete. Processing from scratch...")

    # Load Metadata Parquet Files
    print("Loading metadata...")
    train_df = pd.read_parquet(Config.TRAIN_META)
    val_df = pd.read_parquet(Config.VAL_META)
    test_df = pd.read_parquet(Config.TEST_META)

    # Apply Feature Engineering
    print("Engineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Identify Columns
    target_col = Config.TARGET_COL
    id_col = Config.ID_COL

    # Extract Test IDs
    test_ids = test_df[id_col].values

    # Separate Features
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

    # Identify Binary vs Continuous
    # Binary features in this dataset start with Soil_Type or Wilderness_Area
    binary_cols = [
        c
        for c in feature_cols
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in feature_cols if c not in binary_cols]

    print(f"Features: {len(continuous_cols)} continuous, {len(binary_cols)} binary.")

    # Extract values
    X_train_cont = train_df[continuous_cols].values.astype(np.float32)
    X_val_cont = val_df[continuous_cols].values.astype(np.float32)
    X_test_cont = test_df[continuous_cols].values.astype(np.float32)

    X_train_bin = train_df[binary_cols].values.astype(np.float32)
    X_val_bin = val_df[binary_cols].values.astype(np.float32)
    X_test_bin = test_df[binary_cols].values.astype(np.float32)

    y_train = train_df[target_col].values.astype(np.int64)
    y_val = val_df[target_col].values.astype(np.int64)

    # Standardize Continuous Features
    # IMPORTANT: Fit scaler ONLY on training data to prevent leakage
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Concatenate Continuous and Binary features
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # Adjust Targets for PyTorch (1-7 -> 0-6)
    if y_train.min() == 1:
        y_train = y_train - 1
        y_val = y_val - 1

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(files["train_X"], X_train)
    np.save(files["train_y"], y_train)
    np.save(files["val_X"], X_val)
    np.save(files["val_y"], y_val)
    np.save(files["test_X"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Returns PyTorch DataLoaders for train, val, and test sets.
    """
    X_train, y_train, X_val, y_val, X_test, test_ids = preprocess_data(load_cached_data)

    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test, None)

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
