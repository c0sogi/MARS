import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from library.config import Config
from library.utils import seed_everything


class CoverTypeDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.ids = torch.LongTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        elif self.ids is not None:
            return self.X[idx], self.ids[idx]
        else:
            return self.X[idx]


def feature_engineering(df):
    """
    Applies physics-informed feature engineering.
    """
    # 1. Euclidean Distance to Hydrology
    if Config.USE_HYDRO_DISTANCE:
        df["Hydro_Euclidean"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # 2. Absolute Hydrology Elevation
    if Config.USE_HYDRO_ELEVATION:
        df["Hydro_Elevation_Abs"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

    # 3. Mean Distance to Amenities
    if Config.USE_AMENITIES_MEAN:
        amenities = [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
        ]
        df["Amenities_Mean"] = df[amenities].mean(axis=1)

    # 4. Cyclical Aspect
    if Config.USE_ASPECT_CYCLICAL:
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))
        # We retain the raw 'Aspect' column as per strategy

    return df


class DualViewPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        # Subsample set to 1e5 or larger to speed up fit on large data while maintaining accuracy
        self.quantile = QuantileTransformer(
            output_distribution="normal", subsample=200000, random_state=Config.SEED
        )
        self.cont_cols = []
        self.bin_cols = []

    def fit(self, df):
        # Identify columns
        all_cols = df.columns.tolist()
        # Exclude ID and Target if present (though fit should be called on X)
        exclude = ["Id", "Cover_Type"]

        self.bin_cols = [
            c
            for c in all_cols
            if (c.startswith("Soil_Type") or c.startswith("Wilderness_Area"))
            and c not in exclude
        ]
        self.cont_cols = [
            c for c in all_cols if c not in self.bin_cols and c not in exclude
        ]

        # Fit transformations on continuous columns
        X_cont = df[self.cont_cols].values
        self.scaler.fit(X_cont)
        if Config.USE_QUANTILE_TRANSFORM:
            self.quantile.fit(X_cont)

    def transform(self, df):
        X_cont = df[self.cont_cols].values
        X_bin = df[self.bin_cols].values.astype(np.float32)

        # View 1: Physical (Standardized)
        X_physical = self.scaler.transform(X_cont)

        # View 2: Statistical (Gaussian)
        if Config.USE_QUANTILE_TRANSFORM:
            X_statistical = self.quantile.transform(X_cont)
            # Concatenate: [Physical, Statistical, Binary]
            X_final = np.concatenate([X_physical, X_statistical, X_bin], axis=1)
        else:
            # Concatenate: [Physical, Binary]
            X_final = np.concatenate([X_physical, X_bin], axis=1)

        return X_final.astype(np.float32)


def get_dataloaders(load_cached_data=True):
    """
    Loads data, performs feature engineering and preprocessing, and returns DataLoaders.
    Implements caching to speed up subsequent runs.
    """
    seed_everything(Config.SEED)

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Cache file paths
    files = {
        "train_X": os.path.join(cache_dir, "train_X.npy"),
        "train_y": os.path.join(cache_dir, "train_y.npy"),
        "val_X": os.path.join(cache_dir, "val_X.npy"),
        "val_y": os.path.join(cache_dir, "val_y.npy"),
        "test_X": os.path.join(cache_dir, "test_X.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in files.values())

    if load_cached_data and cache_exists:
        print("Loading cached data from:", cache_dir)
        train_X = np.load(files["train_X"])
        train_y = np.load(files["train_y"])
        val_X = np.load(files["val_X"])
        val_y = np.load(files["val_y"])
        test_X = np.load(files["test_X"])
        test_ids = np.load(files["test_ids"])
    else:
        print("Processing data from scratch...")

        # Load Parquet Metadata
        df_train = pd.read_parquet(Config.TRAIN_PATH)
        df_val = pd.read_parquet(Config.VAL_PATH)
        df_test = pd.read_parquet(Config.TEST_PATH)

        # Extract Targets and IDs
        # Adjust targets to 0-indexed (1-7 -> 0-6)
        train_y = (df_train["Cover_Type"].values - 1).astype(np.int64)
        val_y = (df_val["Cover_Type"].values - 1).astype(np.int64)
        test_ids = df_test["Id"].values.astype(np.int64)

        # Drop Target and Id from features
        df_train = df_train.drop(columns=["Cover_Type", "Id"], errors="ignore")
        df_val = df_val.drop(columns=["Cover_Type", "Id"], errors="ignore")
        df_test = df_test.drop(columns=["Id"], errors="ignore")

        # Feature Engineering
        print("Applying Feature Engineering...")
        df_train = feature_engineering(df_train)
        df_val = feature_engineering(df_val)
        df_test = feature_engineering(df_test)

        # Preprocessing (Dual-View)
        print("Fitting Preprocessor...")
        preprocessor = DualViewPreprocessor()
        preprocessor.fit(df_train)

        print("Transforming Data...")
        train_X = preprocessor.transform(df_train)
        val_X = preprocessor.transform(df_val)
        test_X = preprocessor.transform(df_test)

        # Save to Cache
        print("Saving to cache...")
        np.save(files["train_X"], train_X)
        np.save(files["train_y"], train_y)
        np.save(files["val_X"], val_X)
        np.save(files["val_y"], val_y)
        np.save(files["test_X"], test_X)
        np.save(files["test_ids"], test_ids)

    # Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X, ids=test_ids)

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

    input_dim = train_X.shape[1]
    print(f"Data Loaded. Input Dimension: {input_dim}")
    print(
        f"Train Size: {len(train_dataset)}, Val Size: {len(val_dataset)}, Test Size: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, input_dim
