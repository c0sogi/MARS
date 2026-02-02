import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across various libraries.

    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the appropriate device (cuda if available, else cpu).

    Returns:
        torch.device: The device to use for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _feature_engineering(df):
    """
    Applies the Augmented Physics-Informed Engineering strategy.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Dataframe with added engineered features.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # 1. Cyclical Augmentation for Aspect
    # Convert degrees to radians for sin/cos
    df["Aspect_Sin"] = np.sin(df["Aspect"] * np.pi / 180.0)
    df["Aspect_Cos"] = np.cos(df["Aspect"] * np.pi / 180.0)
    # Note: Raw 'Aspect' is retained.

    # 2. Geometric Magnitude (Euclidean Distance to Hydrology)
    df["Hydrology_Distance_Euclidean"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrol"] ** 2
        + df["Vertical_Distance_To_Hydrolog"] ** 2
    )

    # 3. Directional Preservation (Absolute Hydrology Elevation)
    # Reconstructs the absolute elevation of the water source
    df["Hydrology_Elevation_Abs"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrolog"]
    )

    # 4. Global Context (Mean Distance to Amenities)
    df["Mean_Distance_Amenities"] = df[
        [
            "Horizontal_Distance_To_Hydrol",
            "Horizontal_Distance_To_Roadwa",
            "Horizontal_Distance_To_Fire_P",
        ]
    ].mean(axis=1)

    return df


def get_data(load_cached_data=True, base_dir="./working/idea_11", sample_size=None):
    """
    Loads data, performs feature engineering, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.
        base_dir (str): Directory to store/load cached files.
        sample_size (int, optional): If provided, subsamples the training/validation sets
                                     for debugging purposes.

    Returns:
        train_X (np.ndarray): Training features.
        train_y (np.ndarray): Training labels.
        val_X (np.ndarray): Validation features.
        val_y (np.ndarray): Validation labels.
        test_X (np.ndarray): Test features.
        test_ids (np.ndarray): Test IDs.
    """
    os.makedirs(base_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(base_dir, "train_X.npy"),
        "train_y": os.path.join(base_dir, "train_y.npy"),
        "val_X": os.path.join(base_dir, "val_X.npy"),
        "val_y": os.path.join(base_dir, "val_y.npy"),
        "test_X": os.path.join(base_dir, "test_X.npy"),
        "test_ids": os.path.join(base_dir, "test_ids.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        train_X = np.load(cache_files["train_X"])
        train_y = np.load(cache_files["train_y"])
        val_X = np.load(cache_files["val_X"])
        val_y = np.load(cache_files["val_y"])
        test_X = np.load(cache_files["test_X"])
        test_ids = np.load(cache_files["test_ids"])
    else:
        print("Processing data from scratch...")

        # Load metadata parquet files
        train_df = pd.read_parquet("./metadata/train.parquet")
        val_df = pd.read_parquet("./metadata/val.parquet")
        test_df = pd.read_parquet("./metadata/test.parquet")

        # Extract Targets and IDs
        train_y = train_df["Cover_Type"].values
        val_y = val_df["Cover_Type"].values
        test_ids = test_df["Id"].values

        # Prepare feature dataframes (drop non-feature columns)
        X_train_raw = train_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        X_val_raw = val_df.drop(columns=["Id", "Cover_Type"], errors="ignore")
        X_test_raw = test_df.drop(columns=["Id", "Cover_Type"], errors="ignore")

        # Apply Feature Engineering
        X_train_eng = _feature_engineering(X_train_raw)
        X_val_eng = _feature_engineering(X_val_raw)
        X_test_eng = _feature_engineering(X_test_raw)

        # Identify Continuous vs Binary columns
        # Binary columns in this dataset start with 'Wilderness_Area' or 'Soil_Type'
        all_cols = X_train_eng.columns.tolist()
        binary_cols = [
            c
            for c in all_cols
            if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
        ]
        continuous_cols = [c for c in all_cols if c not in binary_cols]

        # Standardization (Fit on Train, Transform all)
        # Only standardize continuous columns
        scaler = StandardScaler()
        train_cont = scaler.fit_transform(X_train_eng[continuous_cols])
        val_cont = scaler.transform(X_val_eng[continuous_cols])
        test_cont = scaler.transform(X_test_eng[continuous_cols])

        # Extract binary parts
        train_bin = X_train_eng[binary_cols].values
        val_bin = X_val_eng[binary_cols].values
        test_bin = X_test_eng[binary_cols].values

        # Concatenate: Dense Vector = [Standardized Continuous, Raw Binary]
        train_X = np.hstack([train_cont, train_bin]).astype(np.float32)
        val_X = np.hstack([val_cont, val_bin]).astype(np.float32)
        test_X = np.hstack([test_cont, test_bin]).astype(np.float32)

        # Save to cache
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["val_X"], val_X)
        np.save(cache_files["val_y"], val_y)
        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["test_ids"], test_ids)

        print(f"Data processed and saved to {base_dir}")

    # Apply subsampling if requested (for debugging)
    if sample_size is not None:
        if len(train_X) > sample_size:
            train_X = train_X[:sample_size]
            train_y = train_y[:sample_size]
        if len(val_X) > sample_size:
            val_X = val_X[:sample_size]
            val_y = val_y[:sample_size]

    return train_X, train_y, val_X, val_y, test_X, test_ids


def save_submission(ids, predictions, output_path="./submission/submission.csv"):
    """
    Saves predictions to a CSV file in the required format.

    Args:
        ids (np.ndarray or list): Array of test IDs.
        predictions (np.ndarray or list): Array of predicted class labels.
        output_path (str): Path to save the submission file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"Id": ids, "Cover_Type": predictions})

    # Ensure types are correct
    df["Id"] = df["Id"].astype(int)
    df["Cover_Type"] = df["Cover_Type"].astype(int)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
