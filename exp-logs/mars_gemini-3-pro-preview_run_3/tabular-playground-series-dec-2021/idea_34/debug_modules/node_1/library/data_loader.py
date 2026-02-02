import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class ForestDataset(Dataset):
    """
    PyTorch Dataset for Forest Cover Type classification.
    Handles feature conversion to tensors and label adjustment.
    """

    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing features and labels/IDs.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode
        self.df = df.copy()

        # Extract IDs
        self.ids = self.df[Config.ID_COL].values

        # Identify feature columns (exclude ID and Target)
        drop_cols = [Config.ID_COL]
        if Config.TARGET_COL in self.df.columns:
            drop_cols.append(Config.TARGET_COL)

        # Convert features to float32 numpy array first for efficiency
        self.features = self.df.drop(columns=drop_cols).values.astype(np.float32)

        # Handle Targets
        if self.mode != "test":
            if Config.TARGET_COL not in self.df.columns:
                raise ValueError(
                    f"Target column '{Config.TARGET_COL}' missing in {mode} set."
                )
            # Convert 1-based class labels (1-7) to 0-based (0-6) for PyTorch CrossEntropy
            self.targets = self.df[Config.TARGET_COL].values.astype(np.int64) - 1
        else:
            self.targets = np.zeros(len(self.df), dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        id_val = torch.tensor(self.ids[idx], dtype=torch.long)

        return x, y, id_val


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Augmented Physics-Informed Engineering.
    Adds cyclical aspect, geometric distances, and global context features.
    """
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Convert degrees to radians for sin/cos
    aspect_rad = np.radians(df[Config.COL_ASPECT])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(H^2 + V^2)
    h_dist = df[Config.COL_HORZ_DIST_HYDRO]
    v_dist = df[Config.COL_VERT_DIST_HYDRO]
    df["Hydro_Euclidean"] = np.sqrt(h_dist**2 + v_dist**2)

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance (preserves flow direction logic)
    df["Hydro_Elevation"] = df[Config.COL_ELEVATION] - df[Config.COL_VERT_DIST_HYDRO]

    # 4. Global Context (Mean Distance to Amenities)
    # Mean of distances to Hydro, Road, Fire
    d_hydro = df[Config.COL_HORZ_DIST_HYDRO]
    d_road = df[Config.COL_HORZ_DIST_ROAD]
    d_fire = df[Config.COL_HORZ_DIST_FIRE]
    df["Mean_Amenities"] = (d_hydro + d_road + d_fire) / 3.0

    return df


def process_data(load_cached_data: bool = True):
    """
    Loads data, applies feature engineering and preprocessing (scaling).
    Implements caching mechanism using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = Config.TRAIN_PROCESSED_PATH
    val_cache = Config.VAL_PROCESSED_PATH
    test_cache = Config.TEST_PROCESSED_PATH

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("Processing data from scratch...")

    # Load raw metadata
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    val_df = pd.read_parquet(Config.VAL_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)

    # Apply Feature Engineering
    print("Applying feature engineering...")
    train_df = apply_feature_engineering(train_df)
    val_df = apply_feature_engineering(val_df)
    test_df = apply_feature_engineering(test_df)

    # Identify Continuous vs Binary Columns
    # Binary columns are Soil_Type* and Wilderness_Area*
    # We assume all others (except Id and Target) are continuous
    exclude_cols = [Config.ID_COL, Config.TARGET_COL]
    all_cols = [c for c in train_df.columns if c not in exclude_cols]

    binary_cols = [
        c
        for c in all_cols
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    print(
        f"Identified {len(continuous_cols)} continuous features and {len(binary_cols)} binary features."
    )

    # Standardize Continuous Features
    # Fit ONLY on Train
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

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Save to cache
    print("Saving processed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data: bool = True):
    """
    Orchestrates data processing and DataLoader creation.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
    """
    # 1. Process Data
    train_df, val_df, test_df = process_data(load_cached_data=load_cached_data)

    # 2. Create Datasets
    train_dataset = ForestDataset(train_df, mode="train")
    val_dataset = ForestDataset(val_df, mode="val")
    test_dataset = ForestDataset(test_df, mode="test")

    # Calculate input dimension (number of features)
    # Get one sample to check shape [0] is features
    input_dim = train_dataset[0][0].shape[0]
    print(f"Input Feature Dimension: {input_dim}")

    # 3. Create DataLoaders
    # Use num_workers from Config, pin_memory for GPU efficiency
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for stability in training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, input_dim
