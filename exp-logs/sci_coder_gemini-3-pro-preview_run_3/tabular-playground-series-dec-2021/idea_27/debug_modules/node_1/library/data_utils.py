import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config

# Ensure reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type data.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def apply_feature_engineering(df):
    """
    Applies physics-informed feature engineering.
    """
    # Avoid modifying the original dataframe
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Aspect is in degrees 0-360
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(h_dist^2 + v_dist^2)
    df["Euclidean_Dist_Hydro"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation of the water source = Cell Elevation - Vertical Distance to Hydro
    df["Abs_Hydro_Elev"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Amenities: Hydrology, Roadways, Fire Points
    amenities = [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    df["Mean_Dist_Amenities"] = df[amenities].mean(axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    Returns: X_train, y_train, X_val, y_val, X_test, test_ids
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    paths = {
        "X_train": os.path.join(cache_dir, "X_train.npy"),
        "y_train": os.path.join(cache_dir, "y_train.npy"),
        "X_val": os.path.join(cache_dir, "X_val.npy"),
        "y_val": os.path.join(cache_dir, "y_val.npy"),
        "X_test": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_cached:
        print("Loading cached data from", cache_dir)
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_val = np.load(paths["X_val"])
        y_val = np.load(paths["y_val"])
        X_test = np.load(paths["X_test"])
        test_ids = np.load(paths["test_ids"])
        return X_train, y_train, X_val, y_val, X_test, test_ids

    print("Processing data from scratch...")

    # Load metadata parquets
    train_df = pd.read_parquet(Config.TRAIN_DATA_PATH)
    val_df = pd.read_parquet(Config.VAL_DATA_PATH)
    test_df = pd.read_parquet(Config.TEST_DATA_PATH)

    # Apply Feature Engineering
    train_df = apply_feature_engineering(train_df)
    val_df = apply_feature_engineering(val_df)
    test_df = apply_feature_engineering(test_df)

    # Identify columns
    # Continuous features: Original numericals + Engineered
    # We exclude Id and Cover_Type
    exclude_cols = ["Id", "Cover_Type"]

    # Identify binary columns (Soil_Type and Wilderness_Area)
    binary_cols = [
        c
        for c in train_df.columns
        if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
    ]

    # Continuous are the rest
    continuous_cols = [
        c for c in train_df.columns if c not in binary_cols + exclude_cols
    ]

    print(f"Features: {len(continuous_cols)} continuous, {len(binary_cols)} binary.")

    # Preprocessing
    # 1. Standardize Continuous Features (Fit on Train, Transform All)
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(
        train_df[continuous_cols].values.astype(np.float32)
    )
    X_val_cont = scaler.transform(val_df[continuous_cols].values.astype(np.float32))
    X_test_cont = scaler.transform(test_df[continuous_cols].values.astype(np.float32))

    # 2. Get Raw Binary Features
    X_train_bin = train_df[binary_cols].values.astype(np.float32)
    X_val_bin = val_df[binary_cols].values.astype(np.float32)
    X_test_bin = test_df[binary_cols].values.astype(np.float32)

    # 3. Concatenate
    X_train = np.hstack([X_train_cont, X_train_bin])
    X_val = np.hstack([X_val_cont, X_val_bin])
    X_test = np.hstack([X_test_cont, X_test_bin])

    # 4. Process Targets (Shift 1-7 to 0-6)
    y_train = (train_df["Cover_Type"].values - 1).astype(np.int64)
    y_val = (val_df["Cover_Type"].values - 1).astype(np.int64)

    # 5. Extract Test IDs
    test_ids = test_df["Id"].values

    # Cache results
    print("Saving processed data to cache...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    Returns: train_loader, val_loader, test_loader, input_dim, test_ids
    """
    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        load_cached_data=load_cached_data
    )

    # Create Datasets
    train_dataset = ForestDataset(X_train, y_train)
    val_dataset = ForestDataset(X_val, y_val)
    test_dataset = ForestDataset(X_test, None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim, test_ids
