import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything

# Base continuous columns as per dataset description.
# Binary columns (Soil_Type*, Wilderness_Area*) will be detected dynamically.
BASE_CONTINUOUS_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]


class ForestCoverDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type prediction task.
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


def engineer_features(df):
    """
    Applies Augmented Physics-Informed Engineering.
    Calculates cyclical aspect, geometric distances, and global context features.
    """
    df = df.copy()

    # 1. Cyclical Augmentation for Aspect
    # We retain the raw 'Aspect' column as per Lesson 00034
    if Config.ADD_ASPECT_CYCLICAL:
        aspect_rad = np.deg2rad(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    if Config.ADD_HYDROLOGY_EUCLIDEAN:
        df["Hydrology_Distance"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2
            + df["Vertical_Distance_To_Hydrology"] ** 2
        )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    if Config.ADD_ABSOLUTE_HYDROLOGY:
        df["Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
        )

    # 4. Global Context (Mean Distance to Amenities)
    if Config.ADD_AMENITIES_MEAN:
        df["Amenities_Mean_Dist"] = df[
            [
                "Horizontal_Distance_To_Hydrology",
                "Horizontal_Distance_To_Roadways",
                "Horizontal_Distance_To_Fire_Points",
            ]
        ].mean(axis=1)

    return df


def get_data_loaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Loads data, performs feature engineering, preprocessing, and returns PyTorch DataLoaders.
    Implements strict caching logic using .npy files in the working directory.
    """

    # Ensure cache directory exists
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    train_x_path = os.path.join(cache_dir, "train_X.npy")
    train_y_path = os.path.join(cache_dir, "train_y.npy")
    val_x_path = os.path.join(cache_dir, "val_X.npy")
    val_y_path = os.path.join(cache_dir, "val_y.npy")
    test_x_path = os.path.join(cache_dir, "test_X.npy")

    # Check if all required cache files exist
    cache_exists = (
        os.path.exists(train_x_path)
        and os.path.exists(train_y_path)
        and os.path.exists(val_x_path)
        and os.path.exists(val_y_path)
        and os.path.exists(test_x_path)
    )

    # Logic: Load from cache if requested and available, otherwise process from scratch
    if load_cached_data and cache_exists:
        print(f"Loading cached data from {cache_dir}...")
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path)
        y_val = np.load(val_y_path)
        X_test = np.load(test_x_path)

    else:
        print("Cache not found or reload requested. Processing data from scratch...")

        # Load raw metadata Parquet files
        print(f"Loading train data from {Config.TRAIN_DATA_PATH}...")
        df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
        print(f"Loading val data from {Config.VAL_DATA_PATH}...")
        df_val = pd.read_parquet(Config.VAL_DATA_PATH)
        print(f"Loading test data from {Config.TEST_DATA_PATH}...")
        df_test = pd.read_parquet(Config.TEST_DATA_PATH)

        # Apply Debug Subsampling if configured
        if debug_sample_size is not None:
            print(f"DEBUG MODE: Subsampling datasets to {debug_sample_size} rows.")
            df_train = df_train.iloc[:debug_sample_size]
            df_val = df_val.iloc[:debug_sample_size]
            df_test = df_test.iloc[:debug_sample_size]

        # Extract Targets
        # Subtract 1 to map classes 1-7 to 0-6 for CrossEntropyLoss
        y_train = (df_train[Config.TARGET_COL].values - 1).astype(np.int64)
        y_val = (df_val[Config.TARGET_COL].values - 1).astype(np.int64)

        # Prepare Feature DataFrames (Drop ID and Target)
        drop_cols_train = [Config.TARGET_COL]
        if Config.ID_COL in df_train.columns:
            drop_cols_train.append(Config.ID_COL)

        drop_cols_test = []
        if Config.ID_COL in df_test.columns:
            drop_cols_test.append(Config.ID_COL)

        X_train_df = df_train.drop(columns=drop_cols_train)
        X_val_df = df_val.drop(columns=drop_cols_train)
        X_test_df = df_test.drop(columns=drop_cols_test)

        # Feature Engineering
        print("Applying feature engineering...")
        X_train_df = engineer_features(X_train_df)
        X_val_df = engineer_features(X_val_df)
        X_test_df = engineer_features(X_test_df)

        # Identify Continuous vs Binary Columns
        # Construct list of expected continuous columns including new engineered ones
        engineered_continuous = []
        if Config.ADD_ASPECT_CYCLICAL:
            engineered_continuous.extend(["Aspect_Sin", "Aspect_Cos"])
        if Config.ADD_HYDROLOGY_EUCLIDEAN:
            engineered_continuous.append("Hydrology_Distance")
        if Config.ADD_ABSOLUTE_HYDROLOGY:
            engineered_continuous.append("Hydrology_Elevation")
        if Config.ADD_AMENITIES_MEAN:
            engineered_continuous.append("Amenities_Mean_Dist")

        all_potential_continuous = BASE_CONTINUOUS_COLS + engineered_continuous

        # Filter to ensure we only select columns that exist
        cont_cols = [c for c in all_potential_continuous if c in X_train_df.columns]
        # Binary cols are everything else (Soil Types, Wilderness Areas)
        bin_cols = [c for c in X_train_df.columns if c not in cont_cols]

        print(
            f"Detected {len(cont_cols)} continuous features and {len(bin_cols)} binary features."
        )

        # Standardization
        # Fit scaler ONLY on Training data to avoid leakage
        print("Standardizing continuous features...")
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(X_train_df[cont_cols].values)
        X_val_cont = scaler.transform(X_val_df[cont_cols].values)
        X_test_cont = scaler.transform(X_test_df[cont_cols].values)

        # Retrieve Binary features (no scaling)
        X_train_bin = X_train_df[bin_cols].values
        X_val_bin = X_val_df[bin_cols].values
        X_test_bin = X_test_df[bin_cols].values

        # Concatenate Continuous and Binary features
        # We use float32 for model input
        X_train = np.hstack([X_train_cont, X_train_bin]).astype(np.float32)
        X_val = np.hstack([X_val_cont, X_val_bin]).astype(np.float32)
        X_test = np.hstack([X_test_cont, X_test_bin]).astype(np.float32)

        # Save processed arrays to cache
        print(f"Saving processed data to {cache_dir}...")
        np.save(train_x_path, X_train)
        np.save(train_y_path, y_train)
        np.save(val_x_path, X_val)
        np.save(val_y_path, y_val)
        np.save(test_x_path, X_test)

    # Instantiate Datasets
    train_dataset = ForestCoverDataset(X_train, y_train)
    val_dataset = ForestCoverDataset(X_val, y_val)
    test_dataset = ForestCoverDataset(X_test, None)

    # Create DataLoaders
    # Drop last batch for training to maintain stable statistics if batch size is large
    # but ensure we don't drop if dataset is smaller than batch size (handled by PyTorch usually, but explicit check good)
    drop_last_train = True if len(train_dataset) > batch_size else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
