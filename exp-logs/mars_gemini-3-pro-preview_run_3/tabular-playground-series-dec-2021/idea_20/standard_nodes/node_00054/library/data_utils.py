import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


# --------------------------------------------------------------------------
# Dataset Class
# --------------------------------------------------------------------------
class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type data.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Targets are 1-7, convert to 0-6 for PyTorch CrossEntropyLoss
            self.y = torch.tensor(y - 1, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# --------------------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------------------
def engineer_features(df):
    """
    Applies feature engineering based on Config flags.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation
    if Config.USE_CYCLICAL_ASPECT:
        # Convert degrees to radians
        aspect_rad = df["Aspect"] * np.pi / 180.0
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)
        # Note: Raw 'Aspect' is retained as per instructions

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    if Config.USE_EUCLIDEAN_HYDRO:
        df["Hydro_Euclidean"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    if Config.USE_ABS_HYDRO_ELEV:
        df["Hydro_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Amenities)
    if Config.USE_MEAN_AMENITIES:
        amenities = [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
        # Ensure columns exist before calculating mean
        available_amenities = [col for col in amenities if col in df.columns]
        if available_amenities:
            df["Mean_Amenities"] = df[available_amenities].mean(axis=1)

    return df


def get_feature_columns(df):
    """
    Identifies continuous and binary columns after engineering.
    """
    # Base continuous columns from Config
    cont_cols = list(Config.CONTINUOUS_COLS)

    # Add engineered continuous columns
    if Config.USE_CYCLICAL_ASPECT:
        cont_cols.extend(["Aspect_Sin", "Aspect_Cos"])
    if Config.USE_EUCLIDEAN_HYDRO:
        cont_cols.append("Hydro_Euclidean")
    if Config.USE_ABS_HYDRO_ELEV:
        cont_cols.append("Hydro_Elevation")
    if Config.USE_MEAN_AMENITIES:
        cont_cols.append("Mean_Amenities")

    # Binary columns are fixed
    bin_cols = list(Config.BINARY_COLS)

    # Verify columns exist in df
    cont_cols = [c for c in cont_cols if c in df.columns]
    bin_cols = [c for c in bin_cols if c in df.columns]

    return cont_cols, bin_cols


# --------------------------------------------------------------------------
# Data Processing Pipeline
# --------------------------------------------------------------------------
def process_data(load_cached_data=True):
    """
    Loads, engineers, scales, and caches data.
    Returns numpy arrays: train_X, train_y, val_X, val_y, test_X, test_ids
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_X": os.path.join(Config.WORKING_DIR, "train_X.npy"),
        "train_y": os.path.join(Config.WORKING_DIR, "train_y.npy"),
        "val_X": os.path.join(Config.WORKING_DIR, "val_X.npy"),
        "val_y": os.path.join(Config.WORKING_DIR, "val_y.npy"),
        "test_X": os.path.join(Config.WORKING_DIR, "test_X.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
    }

    # Check if cache exists
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading cached data from {Config.WORKING_DIR}")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        test_X = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load Parquet Metadata
    # Using pandas read_parquet as requested
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # Extract IDs and Targets
    train_y = df_train[Config.TARGET_COL].values
    val_y = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Feature Engineering
    print("Engineering features...")
    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Identify Columns
    cont_cols, bin_cols = get_feature_columns(df_train)

    # Standardization (Fit on Train, Transform All)
    print("Standardizing continuous features...")
    scaler = StandardScaler()

    # Fit on training data only
    train_cont = scaler.fit_transform(df_train[cont_cols].values.astype(np.float32))
    val_cont = scaler.transform(df_val[cont_cols].values.astype(np.float32))
    test_cont = scaler.transform(df_test[cont_cols].values.astype(np.float32))

    # Get Binary Features (No scaling, just cast to float for tensor compatibility)
    train_bin = df_train[bin_cols].values.astype(np.float32)
    val_bin = df_val[bin_cols].values.astype(np.float32)
    test_bin = df_test[bin_cols].values.astype(np.float32)

    # Concatenate Continuous and Binary features
    train_X = np.hstack([train_cont, train_bin])
    val_X = np.hstack([val_cont, val_bin])
    test_X = np.hstack([test_cont, test_bin])

    # Cache results
    print(f"Caching processed data to {Config.WORKING_DIR}...")
    np.save(cache_files["train_X"], train_X)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["val_X"], val_X)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["test_X"], test_X)
    np.save(cache_files["test_ids"], test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


# --------------------------------------------------------------------------
# DataLoader Generation
# --------------------------------------------------------------------------
def get_dataloaders(batch_size=None, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.
    Returns: train_loader, val_loader, test_loader, input_dim, test_ids
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(load_cached_data)

    # Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X, y=None)

    # Create DataLoaders
    # num_workers=4 is safe for 12 vCPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    input_dim = train_X.shape[1]

    return train_loader, val_loader, test_loader, input_dim, test_ids
