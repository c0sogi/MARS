import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type prediction task.
    Handles the separation of continuous features and categorical indices for the DCN model.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature matrix of shape (N, F).
                            The last 2 columns are expected to be the categorical indices
                            (Soil_Type, Wilderness_Area).
                            The preceding columns are continuous features.
            y (np.ndarray, optional): Target labels of shape (N,). Defaults to None.
        """
        # Convert to tensors
        # X is float32 to accommodate both standardized floats and integer indices (stored as float)
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            # Shift targets from 1-7 (original) to 0-6 (for PyTorch CrossEntropy)
            self.y = torch.tensor(y - 1, dtype=torch.long)
        else:
            self.y = None

        # Number of categorical columns at the end of the feature matrix
        self.n_cat = 2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Extract all features for the index
        features = self.X[idx]

        # Split into continuous and categorical parts
        # Continuous: All except last 2
        x_cont = features[: -self.n_cat]

        # Categorical: Last 2 (Soil, Wilderness)
        # Convert back to long (int64) for embedding lookup
        x_cat = features[-self.n_cat :].long()

        if self.y is not None:
            return x_cont, x_cat, self.y[idx]
        else:
            return x_cont, x_cat


def _engineer_features(df):
    """
    Internal helper to perform feature engineering on a dataframe.

    1. Reconstructs categorical indices from one-hot encoded columns.
    2. Generates composite continuous features (gradient shortcuts).

    Returns:
        df_cont (pd.DataFrame): The continuous features (original + new).
        X_cat (np.ndarray): The reconstructed categorical indices (N, 2).
    """
    # 1. Reconstruct Categorical Indices
    # Identify one-hot columns based on prefixes defined in Config
    soil_cols = [c for c in df.columns if c.startswith(Config.SOIL_PREFIX)]
    wild_cols = [c for c in df.columns if c.startswith(Config.WILDERNESS_PREFIX)]

    # Use argmax to get the index of the active column (0-based)
    # Soil: 0-39, Wilderness: 0-3
    soil_indices = np.argmax(df[soil_cols].values, axis=1)
    wild_indices = np.argmax(df[wild_cols].values, axis=1)

    # Stack into a (N, 2) array
    X_cat = np.stack([soil_indices, wild_indices], axis=1)

    # 2. Process Continuous Features
    # Exclude ID, Target, and the one-hot columns we just compressed
    exclude_cols = set(soil_cols + wild_cols + [Config.ID_COL, Config.TARGET_COL])
    cont_cols = [c for c in df.columns if c not in exclude_cols]

    df_cont = df[cont_cols].copy()

    # 3. Create Composite Features (Domain Knowledge / Gradient Shortcuts)
    # Note: Column names are based on the provided dataset analysis

    # Euclidean Distance to Hydrology
    # Combines Horizontal and Vertical distance into a true straight-line distance
    h_dist_hydro = df_cont["Horizontal_Distance_To_Hydrology"]
    v_dist_hydro = df_cont["Vertical_Distance_To_Hydrolog"]
    df_cont["Dist_Hydro_Euclidean"] = np.sqrt(h_dist_hydro**2 + v_dist_hydro**2)

    # Mean Distance to Amenities (Hydrology, Roadways, Fire Points)
    # Aggregates general "remoteness"
    h_dist_road = df_cont["Horizontal_Distance_To_Roadways"]
    h_dist_fire = df_cont["Horizontal_Distance_To_Fire_Points"]
    df_cont["Mean_Dist_Amenities"] = (h_dist_hydro + h_dist_road + h_dist_fire) / 3.0

    # Elevation relative to Hydrology
    # Absolute elevation of the water source
    elevation = df_cont["Elevation"]
    df_cont["Elev_Minus_Vert_Hydro"] = elevation - v_dist_hydro
    df_cont["Elev_Plus_Vert_Hydro"] = elevation + v_dist_hydro

    return df_cont, X_cat


def process_data(load_cached_data=True):
    """
    Main data processing pipeline.

    1. Checks for cached numpy arrays.
    2. If not found, loads Parquet metadata.
    3. Performs feature engineering and standardization.
    4. Caches the results.

    Args:
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (train_X, train_y, val_X, val_y, test_X, test_ids)
    """
    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if all cache files exist
    cache_files = [
        Config.CACHE_TRAIN_X,
        Config.CACHE_TRAIN_Y,
        Config.CACHE_VAL_X,
        Config.CACHE_VAL_Y,
        Config.CACHE_TEST_X,
        Config.CACHE_TEST_IDS,
    ]

    if load_cached_data and all(os.path.exists(f) for f in cache_files):
        print("Loading cached data from", Config.WORKING_DIR)
        train_X = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_X = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
        test_X = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS)
        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load Dataframes
    print(f"Loading {Config.TRAIN_DATA_PATH}...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    print(f"Loading {Config.VAL_DATA_PATH}...")
    df_val = pd.read_parquet(Config.VAL_DATA_PATH)
    print(f"Loading {Config.TEST_DATA_PATH}...")
    df_test = pd.read_parquet(Config.TEST_DATA_PATH)

    # Extract Targets and IDs
    train_y = df_train[Config.TARGET_COL].values
    val_y = df_val[Config.TARGET_COL].values
    test_ids = df_test[Config.ID_COL].values

    # Feature Engineering
    print("Engineering features...")
    train_cont, train_cat = _engineer_features(df_train)
    val_cont, val_cat = _engineer_features(df_val)
    test_cont, test_cat = _engineer_features(df_test)

    # Standardization
    # Fit only on training data to avoid leakage
    print("Standardizing continuous features...")
    scaler = StandardScaler()
    train_cont_scaled = scaler.fit_transform(train_cont).astype(np.float32)
    val_cont_scaled = scaler.transform(val_cont).astype(np.float32)
    test_cont_scaled = scaler.transform(test_cont).astype(np.float32)

    # Concatenate Continuous and Categorical
    # We store everything in a single float32 array for efficiency.
    # Categorical indices are integers, but can be stored as floats losslessly.
    # Structure: [Continuous Features ..., Soil_Idx, Wild_Idx]
    print("Combining features...")
    train_X = np.hstack([train_cont_scaled, train_cat.astype(np.float32)])
    val_X = np.hstack([val_cont_scaled, val_cat.astype(np.float32)])
    test_X = np.hstack([test_cont_scaled, test_cat.astype(np.float32)])

    # Cache results
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    np.save(Config.CACHE_TRAIN_X, train_X)
    np.save(Config.CACHE_TRAIN_Y, train_y)
    np.save(Config.CACHE_VAL_X, val_X)
    np.save(Config.CACHE_VAL_Y, val_y)
    np.save(Config.CACHE_TEST_X, test_X)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(load_cached_data=True, debug_sample_size=None):
    """
    Factory function to generate DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.
        debug_sample_size (int, optional): If set, subsamples the dataset for debugging.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    # 1. Process/Load Data
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(load_cached_data)

    # 2. Debug Subsampling
    if debug_sample_size is not None:
        print(f"DEBUG: Subsampling data to {debug_sample_size} rows.")
        train_X = train_X[:debug_sample_size]
        train_y = train_y[:debug_sample_size]
        val_X = val_X[:debug_sample_size]
        val_y = val_y[:debug_sample_size]
        test_X = test_X[:debug_sample_size]
        test_ids = test_ids[:debug_sample_size]

    # 3. Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X, None)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, test_ids
