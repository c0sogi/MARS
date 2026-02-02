import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import config


class FeatureEngineer:
    """
    Handles physics-informed feature engineering and preprocessing.
    Implements the Augmented Physics-Informed Engineering strategy.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.continuous_cols = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrol",
            "Vertical_Distance_To_Hydrolog",
            "Horizontal_Distance_To_Roadwa",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_P",
        ]
        # These will be populated during fit
        self.feature_names = None

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies physics-informed transformations.
        """
        df = df.copy()

        # 1. Cyclical Augmentation (Keep raw Aspect as per Lesson 00034)
        # Convert degrees to radians for sin/cos
        aspect_rad = df["Aspect"] * np.pi / 180.0
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

        # 2. Geometric Magnitude
        # Euclidean distance to hydrology (Hypotenuse)
        df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrol"] ** 2
            + df["Vertical_Distance_To_Hydrolog"] ** 2
        )

        # 3. Directional Preservation
        # Absolute Hydrology Elevation (Elevation - Vertical_Distance)
        # Vertical_Dist = Elev - Hydro_Elev -> Hydro_Elev = Elev - Vertical_Dist
        df["Abs_Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrolog"]
        )

        # 4. Global Context
        # Mean Distance to Amenities
        df["Mean_Amenities"] = (
            df["Horizontal_Distance_To_Hydrol"]
            + df["Horizontal_Distance_To_Roadwa"]
            + df["Horizontal_Distance_To_Fire_P"]
        ) / 3.0

        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits scaler on continuous features and transforms the dataframe.
        """
        # Engineer features first
        df_eng = self._engineer_features(df)

        # Identify columns
        # Binary columns are Soil_Type* and Wilderness_Area*
        # We also need to include the newly engineered continuous columns
        new_continuous = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Euclidean_Distance_To_Hydrology",
            "Abs_Hydrology_Elevation",
            "Mean_Amenities",
        ]
        all_continuous = self.continuous_cols + new_continuous

        # Select binary columns
        binary_cols = [
            c
            for c in df_eng.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]

        # Fit scaler on continuous
        self.scaler.fit(df_eng[all_continuous])

        # Transform continuous
        X_cont = self.scaler.transform(df_eng[all_continuous])

        # Get binary part
        X_bin = df_eng[binary_cols].values

        # Concatenate: Continuous first, then Binary
        X_processed = np.hstack([X_cont, X_bin])

        return X_processed.astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms dataframe using already fitted scaler.
        """
        df_eng = self._engineer_features(df)

        new_continuous = [
            "Aspect_Sin",
            "Aspect_Cos",
            "Euclidean_Distance_To_Hydrology",
            "Abs_Hydrology_Elevation",
            "Mean_Amenities",
        ]
        all_continuous = self.continuous_cols + new_continuous
        binary_cols = [
            c
            for c in df_eng.columns
            if c.startswith("Soil_Type") or c.startswith("Wilderness_Area")
        ]

        X_cont = self.scaler.transform(df_eng[all_continuous])
        X_bin = df_eng[binary_cols].values

        X_processed = np.hstack([X_cont, X_bin])

        return X_processed.astype(np.float32)


class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type data.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        if self.labels is not None:
            # Labels are already 0-indexed in process_data
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y

        return x


def process_data(load_cached_data: bool = True, debug: bool = False):
    """
    Loads data, performs feature engineering, and caches results.

    Args:
        load_cached_data: If True, attempts to load from .npy files.
        debug: If True, subsamples data for faster iteration.

    Returns:
        Tuple of (train_X, train_y, val_X, val_y, test_X, test_ids)
    """
    # Define cache paths
    paths = config.paths

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(paths.train_X_path)
        and os.path.exists(paths.train_y_path)
        and os.path.exists(paths.val_X_path)
        and os.path.exists(paths.val_y_path)
        and os.path.exists(paths.test_X_path)
        and os.path.exists(paths.test_ids_path)
    )

    if load_cached_data and cache_exists:
        print("Loading cached data from working directory...")
        train_X = np.load(paths.train_X_path)
        train_y = np.load(paths.train_y_path)
        val_X = np.load(paths.val_X_path)
        val_y = np.load(paths.val_y_path)
        test_X = np.load(paths.test_X_path)
        test_ids = np.load(paths.test_ids_path)

        if debug:
            # Subsample loaded data
            limit = config.train.debug_sample_size
            return (
                train_X[:limit],
                train_y[:limit],
                val_X[:limit],
                val_y[:limit],
                test_X[:limit],
                test_ids[:limit],
            )

        return train_X, train_y, val_X, val_y, test_X, test_ids

    print("Processing data from scratch...")

    # Load Parquet Metadata
    df_train = pd.read_parquet(paths.train_parquet)
    df_val = pd.read_parquet(paths.val_parquet)
    df_test = pd.read_parquet(paths.test_parquet)

    # Debug Subsampling
    if debug:
        print(f"Debug mode: Subsampling to {config.train.debug_sample_size} rows.")
        df_train = df_train.iloc[: config.train.debug_sample_size]
        df_val = df_val.iloc[: config.train.debug_sample_size]
        df_test = df_test.iloc[: config.train.debug_sample_size]

    # Extract Targets and IDs
    # Map 1-7 to 0-6 for PyTorch CrossEntropyLoss
    train_y = (df_train["Cover_Type"].values - 1).astype(np.int64)
    val_y = (df_val["Cover_Type"].values - 1).astype(np.int64)
    test_ids = df_test["Id"].values

    # Drop non-feature columns
    drop_cols = ["Id", "Cover_Type"]
    X_train_raw = df_train.drop(columns=drop_cols, errors="ignore")
    X_val_raw = df_val.drop(columns=drop_cols, errors="ignore")
    X_test_raw = df_test.drop(columns=["Id"], errors="ignore")

    # Feature Engineering
    engineer = FeatureEngineer()

    print("Engineering features for Training set...")
    train_X = engineer.fit_transform(X_train_raw)

    print("Engineering features for Validation set...")
    val_X = engineer.transform(X_val_raw)

    print("Engineering features for Test set...")
    test_X = engineer.transform(X_test_raw)

    # Cache results (only if not debugging, to avoid overwriting full cache with partial data)
    if not debug:
        print(f"Saving processed data to {paths.working_dir}...")
        np.save(paths.train_X_path, train_X)
        np.save(paths.train_y_path, train_y)
        np.save(paths.val_X_path, val_X)
        np.save(paths.val_y_path, val_y)
        np.save(paths.test_X_path, test_X)
        np.save(paths.test_ids_path, test_ids)

    return train_X, train_y, val_X, val_y, test_X, test_ids


def get_dataloaders(load_cached_data: bool = True, debug: bool = False):
    """
    Orchestrates data processing and DataLoader creation.

    Args:
        load_cached_data: Whether to use cached numpy files.
        debug: Whether to run in debug mode (subsampled data).

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(
        load_cached_data, debug
    )

    print(
        f"Data Shapes - Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}"
    )

    # Create Datasets
    train_dataset = ForestDataset(train_X, train_y)
    val_dataset = ForestDataset(val_X, val_y)
    test_dataset = ForestDataset(test_X, labels=None)

    # Create DataLoaders
    # Pin memory for faster host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
