import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    Configures CuDNN for maximum performance as per strategy requirements.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Per strategy "Cite Lesson 00070": Disable strict determinism for performance
    # This prioritizes throughput on the A100 over bit-exact reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device():
    """
    Automatically selects the available NVIDIA A100 GPU or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _feature_engineering(df):
    """
    Applies the "Augmented Physics-Informed Engineering" strategy.
    """
    # Avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation (Keep raw Aspect)
    # Aspect is in degrees (0-360)
    df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
    df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    # sqrt(H^2 + V^2)
    df["Hydro_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Elevation - Vertical_Distance = Elevation of Hydrology
    df["Hydro_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]

    # 4. Global Context (Mean Distance to Amenities)
    # Average of distances to Hydrology, Roadways, and Fire Points
    df["Mean_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadwa"]
        + df["Horizontal_Distance_To_Fire_P"]
    ) / 3.0

    return df


def get_data(
    load_cached_data=True, cache_dir="./working/idea_39/", metadata_dir="./metadata"
):
    """
    Loads, preprocesses, and caches the dataset.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        cache_dir (str): Directory to store/load cached .npy files.
        metadata_dir (str): Directory containing raw parquet metadata.

    Returns:
        dict: Dictionary containing 'train_X', 'train_y', 'val_X', 'val_y', 'test_X', 'test_ids'.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": "train_X.npy",
        "train_y": "train_y.npy",
        "val_X": "val_X.npy",
        "val_y": "val_y.npy",
        "test_X": "test_X.npy",
        "test_ids": "test_ids.npy",
    }

    cache_paths = {k: os.path.join(cache_dir, v) for k, v in files.items()}

    # Check if all cache files exist
    if load_cached_data and all(os.path.exists(p) for p in cache_paths.values()):
        print(f"Loading cached data from {cache_dir}...")
        data = {}
        for k, p in cache_paths.items():
            data[k] = np.load(p)
        return data

    print("Cache not found or reload requested. Processing data from scratch...")

    # Load raw data from metadata
    train_path = os.path.join(metadata_dir, "train.parquet")
    val_path = os.path.join(metadata_dir, "val.parquet")
    test_path = os.path.join(metadata_dir, "test.parquet")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"Metadata files not found in {metadata_dir}. Please ensure metadata is generated."
        )

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # Apply Feature Engineering
    df_train = _feature_engineering(df_train)
    df_val = _feature_engineering(df_val)
    df_test = _feature_engineering(df_test)

    # Identify Columns
    target_col = "Cover_Type"
    id_col = "Id"

    # Binary columns: Wilderness_Area* and Soil_Type*
    # These should NOT be standardized as per strategy
    binary_cols = [
        c
        for c in df_train.columns
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
    ]

    # Continuous columns: All others except Id and Target
    exclude = [target_col, id_col] + binary_cols
    cont_cols = [c for c in df_train.columns if c not in exclude]

    # Standardization
    # Fit on Train, Transform All
    scaler = StandardScaler()
    df_train[cont_cols] = scaler.fit_transform(df_train[cont_cols].astype(np.float32))
    df_val[cont_cols] = scaler.transform(df_val[cont_cols].astype(np.float32))
    df_test[cont_cols] = scaler.transform(df_test[cont_cols].astype(np.float32))

    # Prepare Output Arrays
    # Concatenate Continuous and Binary features
    feature_cols = cont_cols + binary_cols

    X_train = df_train[feature_cols].values.astype(np.float32)
    # Shift target to 0-indexed (1-7 -> 0-6) for PyTorch CrossEntropyLoss
    y_train = (df_train[target_col].values - 1).astype(np.int64)

    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = (df_val[target_col].values - 1).astype(np.int64)

    X_test = df_test[feature_cols].values.astype(np.float32)
    test_ids = df_test[id_col].values.astype(np.int64)

    # Save to Cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(cache_paths["train_X"], X_train)
    np.save(cache_paths["train_y"], y_train)
    np.save(cache_paths["val_X"], X_val)
    np.save(cache_paths["val_y"], y_val)
    np.save(cache_paths["test_X"], X_test)
    np.save(cache_paths["test_ids"], test_ids)

    return {
        "train_X": X_train,
        "train_y": y_train,
        "val_X": X_val,
        "val_y": y_val,
        "test_X": X_test,
        "test_ids": test_ids,
    }
