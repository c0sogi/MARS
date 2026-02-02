import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


def load_and_preprocess(load_cached_data=True):
    """
    Loads data from metadata parquet files, performs feature engineering and scaling,
    and returns processed numpy arrays. Implements caching to save time.

    Args:
        load_cached_data (bool): If True, attempts to load from cached .npy files.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, test_ids)
    """
    seed_everything()

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {
        "X_train": os.path.join(cache_dir, "train_X.npy"),
        "y_train": os.path.join(cache_dir, "train_y.npy"),
        "X_val": os.path.join(cache_dir, "val_X.npy"),
        "y_val": os.path.join(cache_dir, "val_y.npy"),
        "X_test": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # 1. Try Loading from Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print("Loading data from cache...")
            X_train = np.load(paths["X_train"])
            y_train = np.load(paths["y_train"])
            X_val = np.load(paths["X_val"])
            y_val = np.load(paths["y_val"])
            X_test = np.load(paths["X_test"])
            test_ids = np.load(paths["test_ids"])
            return X_train, y_train, X_val, y_val, X_test, test_ids
        else:
            print("Cache miss. Processing data from scratch...")
    else:
        print("Force processing data from scratch...")

    # 2. Load Raw Data
    print("Loading parquet files...")
    df_train = pd.read_parquet(Config.TRAIN_PATH)
    df_val = pd.read_parquet(Config.VAL_PATH)
    df_test = pd.read_parquet(Config.TEST_PATH)

    # 3. Feature Engineering
    print("Performing feature engineering...")

    # Define feature groups
    cont_cols = Config.CONTINUOUS_COLS
    binary_cols = Config.BINARY_COLS

    # Helper to create engineered features
    # Cite solution_lesson_node_00016: Explicit Geometric Feature Engineering Outperforms Generic Aggregations
    def engineer_features(df):
        # Euclidean Distance to Hydrology (Hypotenuse)
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

        # Hydrology Elevation (Absolute elevation of the water source)
        df["Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )
        return df

    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Update continuous columns list to include new features
    new_features = ["Euclidean_Distance_To_Hydrology", "Hydrology_Elevation"]
    all_cont_cols = cont_cols + new_features

    # 4. Standardization
    print("Standardizing continuous features...")
    scaler = StandardScaler()

    # Fit on Train only
    scaler.fit(df_train[all_cont_cols])

    # Transform all sets
    train_cont = scaler.transform(df_train[all_cont_cols]).astype(np.float32)
    val_cont = scaler.transform(df_val[all_cont_cols]).astype(np.float32)
    test_cont = scaler.transform(df_test[all_cont_cols]).astype(np.float32)

    # 5. Assemble Final Arrays
    print("Assembling final arrays...")

    # Get binary features
    train_bin = df_train[binary_cols].values.astype(np.float32)
    val_bin = df_val[binary_cols].values.astype(np.float32)
    test_bin = df_test[binary_cols].values.astype(np.float32)

    # Concatenate: [Continuous, Binary]
    X_train = np.hstack([train_cont, train_bin])
    X_val = np.hstack([val_cont, val_bin])
    X_test = np.hstack([test_cont, test_bin])

    # Process Targets (Shift 1-7 to 0-6)
    y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
    y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

    # Process IDs
    test_ids = df_test[Config.ID_COL].values.astype(np.int64)

    # 6. Save to Cache
    print("Saving processed data to cache...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["X_test"], X_test)
    np.save(paths["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


class CoverTypeDataset(Dataset):
    """
    Standard Dataset for Supervised Training and Inference.
    """

    def __init__(self, features, targets=None):
        """
        Args:
            features (np.ndarray): Feature matrix.
            targets (np.ndarray, optional): Target labels.
        """
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).long() if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.targets is not None:
            y = self.targets[idx]
            return x, y
        return x
