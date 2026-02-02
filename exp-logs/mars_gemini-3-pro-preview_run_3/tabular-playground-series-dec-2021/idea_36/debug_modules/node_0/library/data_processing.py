import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type data.
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


def engineer_features(df, config: Config):
    """
    Applies physics-informed feature engineering.
    """
    # Ensure working with a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation (Lesson 00034)
    # We keep raw Aspect as well
    if config.use_aspect_trig:
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Lesson 00016)
    if config.use_dist_hydro_euclidean:
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrol"] ** 2
            + df["Vertical_Distance_To_Hydrolog"] ** 2
        )

    # 3. Directional Preservation (Lesson 00019)
    if config.use_abs_hydro_elevation:
        df["Abs_Hydro_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrolog"]
        )

    # 4. Global Context (Lesson 00009)
    if config.use_mean_dist_amenities:
        df["Mean_Dist_Amenities"] = df[
            [
                "Horizontal_Distance_To_Hydrol",
                "Horizontal_Distance_To_Roadwa",
                "Horizontal_Distance_To_Fire_P",
            ]
        ].mean(axis=1)

    return df


def process_data(config: Config, load_cached_data: bool = True):
    """
    Loads, engineers, scales, and caches data.
    Returns processed numpy arrays.
    """
    cache_dir = config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),  # Save IDs for submission
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(f) for f in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from", cache_dir)
        data = {k: np.load(v) for k, v in files.items()}
        return (
            data["train_X"],
            data["train_y"],
            data["val_X"],
            data["val_y"],
            data["test_X"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load raw data from metadata
    train_df = pd.read_parquet(config.train_path)
    val_df = pd.read_parquet(config.val_path)
    test_df = pd.read_parquet(config.test_path)

    # Separate Target and IDs
    target_col = "Cover_Type"
    id_col = "Id"

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    test_ids = test_df[id_col].values

    # Adjust targets to 0-indexed for PyTorch (Classes are 1-7 -> 0-6)
    y_train = y_train - 1
    y_val = y_val - 1

    # Drop Target and ID from features
    X_train_df = train_df.drop(columns=[target_col, id_col], errors="ignore")
    X_val_df = val_df.drop(columns=[target_col, id_col], errors="ignore")
    X_test_df = test_df.drop(columns=[id_col], errors="ignore")

    # Feature Engineering
    print("Engineering features...")
    X_train_df = engineer_features(X_train_df, config)
    X_val_df = engineer_features(X_val_df, config)
    X_test_df = engineer_features(X_test_df, config)

    # Identify Continuous vs Binary Features
    # Heuristic: Binary features usually start with "Soil_Type" or "Wilderness_Area"
    # or have only 2 unique values. Given the dataset description, we use name matching.
    binary_prefixes = ["Soil_Type", "Wilderness_Area"]

    all_cols = X_train_df.columns.tolist()
    binary_cols = [c for c in all_cols if any(c.startswith(p) for p in binary_prefixes)]
    continuous_cols = [c for c in all_cols if c not in binary_cols]

    print(f"Continuous features: {len(continuous_cols)}")
    print(f"Binary features: {len(binary_cols)}")

    # Scaling
    # We only scale continuous features. Binary features remain 0/1.
    print("Scaling continuous features...")
    scaler = StandardScaler()

    # Fit on Train, Transform all
    train_cont = scaler.fit_transform(
        X_train_df[continuous_cols].values.astype(np.float32)
    )
    val_cont = scaler.transform(X_val_df[continuous_cols].values.astype(np.float32))
    test_cont = scaler.transform(X_test_df[continuous_cols].values.astype(np.float32))

    # Get binary parts
    train_bin = X_train_df[binary_cols].values.astype(np.float32)
    val_bin = X_val_df[binary_cols].values.astype(np.float32)
    test_bin = X_test_df[binary_cols].values.astype(np.float32)

    # Concatenate
    X_train = np.hstack([train_cont, train_bin])
    X_val = np.hstack([val_cont, val_bin])
    X_test = np.hstack([test_cont, test_bin])

    # Save to cache
    print("Saving data to cache...")
    np.save(files["train_X"], X_train)
    np.save(files["train_y"], y_train)
    np.save(files["val_X"], X_val)
    np.save(files["val_y"], y_val)
    np.save(files["test_X"], X_test)
    np.save(files["test_ids"], test_ids)

    return X_train, y_train, X_val, y_val, X_test, test_ids


def get_dataloaders(config: Config, load_cached_data: bool = True):
    """
    Main entry point for data loading.
    Returns: train_loader, val_loader, test_loader, input_dim
    """
    seed_everything(config.seed)

    X_train, y_train, X_val, y_val, X_test, test_ids = process_data(
        config, load_cached_data
    )

    # Debug mode: subsample data
    if config.debug:
        print(f"DEBUG MODE: Subsampling data to {config.batch_size * 2} rows.")
        limit = config.batch_size * 2
        X_train, y_train = X_train[:limit], y_train[:limit]
        X_val, y_val = X_val[:limit], y_val[:limit]
        # We don't subsample test usually to ensure pipeline works, but for speed in debug:
        X_test, test_ids = X_test[:limit], test_ids[:limit]

    # Create Datasets
    train_dataset = CoverTypeDataset(X_train, y_train)
    val_dataset = CoverTypeDataset(X_val, y_val)
    test_dataset = CoverTypeDataset(X_test, None)  # No targets for test

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, test_ids, input_dim
