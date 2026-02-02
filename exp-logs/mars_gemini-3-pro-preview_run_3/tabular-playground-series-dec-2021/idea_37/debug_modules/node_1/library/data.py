import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    CACHE_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG_SAMPLE_SIZE,
    SEED,
    USE_ASPECT_TRIG,
    USE_HYDRO_DIST,
    USE_HYDRO_ELEV,
    USE_AMENITIES_MEAN,
)


class ForestDataset(Dataset):
    def __init__(self, X_cont, X_bin, y=None, is_test=False, ids=None):
        self.X_cont = torch.FloatTensor(X_cont)
        self.X_bin = torch.FloatTensor(X_bin)
        self.is_test = is_test

        if self.is_test:
            self.ids = ids
        else:
            # Shift labels from 1-7 to 0-6 for CrossEntropyLoss
            self.y = torch.LongTensor(y) - 1

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X_cont[idx], self.X_bin[idx], self.ids[idx]
        else:
            return self.X_cont[idx], self.X_bin[idx], self.y[idx]


def _apply_feature_engineering(df):
    """
    Applies physics-informed feature engineering.
    """
    # 1. Cyclical Aspect
    if USE_ASPECT_TRIG:
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Euclidean Distance to Hydrology
    if USE_HYDRO_DIST:
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # 3. Absolute Hydrology Elevation
    if USE_HYDRO_ELEV:
        # Vertical_Dist = Elev - Hydro_Elev -> Hydro_Elev = Elev - Vertical_Dist
        df["Absolute_Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

    # 4. Mean Distance to Amenities
    if USE_AMENITIES_MEAN:
        df["Mean_Distance_To_Amenities"] = (
            df["Horizontal_Distance_To_Hydrology"]
            + df["Horizontal_Distance_To_Roadways"]
            + df["Horizontal_Distance_To_Fire_Points"]
        ) / 3.0

    return df


def _split_features(df):
    """
    Separates continuous and binary features.
    Binary features are Wilderness_Area* and Soil_Type*.
    """
    # Identify binary columns based on name patterns
    bin_cols = [
        c
        for c in df.columns
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
    ]
    # Continuous columns are everything else except Id and Cover_Type
    exclude = ["Id", "Cover_Type"] + bin_cols
    cont_cols = [c for c in df.columns if c not in exclude]

    return df[cont_cols].values.astype(np.float32), df[bin_cols].values.astype(
        np.float32
    )


def get_dataloaders(load_cached_data=True):
    """
    Loads data, performs feature engineering, standardization, and returns DataLoaders.
    Implements caching to speed up subsequent runs.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "train_X_cont": os.path.join(CACHE_DIR, "train_X_cont.npy"),
        "train_X_bin": os.path.join(CACHE_DIR, "train_X_bin.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X_cont": os.path.join(CACHE_DIR, "val_X_cont.npy"),
        "val_X_bin": os.path.join(CACHE_DIR, "val_X_bin.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X_cont": os.path.join(CACHE_DIR, "test_X_cont.npy"),
        "test_X_bin": os.path.join(CACHE_DIR, "test_X_bin.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data = {k: np.load(v) for k, v in cache_files.items()}

        train_X_cont, train_X_bin, train_y = (
            data["train_X_cont"],
            data["train_X_bin"],
            data["train_y"],
        )
        val_X_cont, val_X_bin, val_y = (
            data["val_X_cont"],
            data["val_X_bin"],
            data["val_y"],
        )
        test_X_cont, test_X_bin, test_ids = (
            data["test_X_cont"],
            data["test_X_bin"],
            data["test_ids"],
        )

    else:
        print("Processing data from scratch...")

        # Load Parquet files
        df_train = pd.read_parquet(TRAIN_PATH)
        df_val = pd.read_parquet(VAL_PATH)
        df_test = pd.read_parquet(TEST_PATH)

        # Subsample for debugging if requested
        if DEBUG_SAMPLE_SIZE is not None:
            print(f"DEBUG: Subsampling {DEBUG_SAMPLE_SIZE} rows.")
            df_train = df_train.iloc[:DEBUG_SAMPLE_SIZE]
            df_val = df_val.iloc[:DEBUG_SAMPLE_SIZE]
            df_test = df_test.iloc[:DEBUG_SAMPLE_SIZE]

        # Extract Targets and IDs
        train_y = df_train["Cover_Type"].values
        val_y = df_val["Cover_Type"].values
        test_ids = df_test["Id"].values

        # Apply Feature Engineering
        print("Applying feature engineering...")
        df_train = _apply_feature_engineering(df_train)
        df_val = _apply_feature_engineering(df_val)
        df_test = _apply_feature_engineering(df_test)

        # Split Continuous vs Binary
        print("Splitting continuous and binary features...")
        train_X_cont, train_X_bin = _split_features(df_train)
        val_X_cont, val_X_bin = _split_features(df_val)
        test_X_cont, test_X_bin = _split_features(df_test)

        # Standardization (Fit on Train, Transform All)
        print("Standardizing continuous features...")
        scaler = StandardScaler()
        train_X_cont = scaler.fit_transform(train_X_cont)
        val_X_cont = scaler.transform(val_X_cont)
        test_X_cont = scaler.transform(test_X_cont)

        # Save to Cache
        print(f"Saving processed data to {CACHE_DIR}...")
        np.save(cache_files["train_X_cont"], train_X_cont)
        np.save(cache_files["train_X_bin"], train_X_bin)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_X_cont"], val_X_cont)
        np.save(cache_files["val_X_bin"], val_X_bin)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_X_cont"], test_X_cont)
        np.save(cache_files["test_X_bin"], test_X_bin)
        np.save(cache_files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = ForestDataset(train_X_cont, train_X_bin, train_y, is_test=False)
    val_dataset = ForestDataset(val_X_cont, val_X_bin, val_y, is_test=False)
    test_dataset = ForestDataset(test_X_cont, test_X_bin, ids=test_ids, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Metadata for model initialization
    input_info = {
        "cont_dim": train_X_cont.shape[1],
        "bin_dim": train_X_bin.shape[1],
        "num_classes": 7,  # Fixed for this dataset
    }

    return train_loader, val_loader, test_loader, input_info
