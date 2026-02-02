import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# Set fixed random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class CoverTypeDataset(Dataset):
    """
    PyTorch Dataset for the Cover Type classification task.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Ensure targets are LongTensor for CrossEntropyLoss
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads data from metadata parquet files, performs physics-informed feature engineering,
    applies dual-view preprocessing (StandardScaler + QuantileTransformer),
    caches the processed numpy arrays, and returns PyTorch Datasets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed data from cache.

    Returns:
        train_dataset (CoverTypeDataset): Training dataset.
        val_dataset (CoverTypeDataset): Validation dataset.
        test_dataset (CoverTypeDataset): Test dataset.
        input_dim (int): Dimension of the input feature vector.
        num_classes (int): Number of target classes.
        test_ids (np.ndarray): Array of IDs corresponding to the test set.
    """
    CACHE_DIR = "./working/idea_42/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    files = {
        "train_X": os.path.join(CACHE_DIR, "train_X.npy"),
        "train_y": os.path.join(CACHE_DIR, "train_y.npy"),
        "val_X": os.path.join(CACHE_DIR, "val_X.npy"),
        "val_y": os.path.join(CACHE_DIR, "val_y.npy"),
        "test_X": os.path.join(CACHE_DIR, "test_X.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist:
            print(f"Loading cached data from {CACHE_DIR}...")
            train_X = np.load(files["train_X"])
            train_y = np.load(files["train_y"])
            val_X = np.load(files["val_X"])
            val_y = np.load(files["val_y"])
            test_X = np.load(files["test_X"])
            test_ids = np.load(files["test_ids"])

            train_dataset = CoverTypeDataset(train_X, train_y)
            val_dataset = CoverTypeDataset(val_X, val_y)
            test_dataset = CoverTypeDataset(test_X)

            input_dim = train_X.shape[1]
            num_classes = len(np.unique(train_y))

            return (
                train_dataset,
                val_dataset,
                test_dataset,
                input_dim,
                num_classes,
                test_ids,
            )

    # 2. Process from scratch
    print("Processing data from scratch...")

    METADATA_DIR = "./metadata"
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    target_col = "Cover_Type"
    id_col = "Id"

    # Extract Targets and IDs
    # Map 1-7 class labels to 0-6 for PyTorch
    train_y = train_df[target_col].values - 1
    val_y = val_df[target_col].values - 1
    test_ids = test_df[id_col].values

    # Drop non-feature columns
    train_df = train_df.drop(columns=[id_col, target_col], errors="ignore")
    val_df = val_df.drop(columns=[id_col, target_col], errors="ignore")
    test_df = test_df.drop(columns=[id_col], errors="ignore")

    # Define Column Groups based on dataset analysis
    cont_cols = [
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

    # Identify binary columns (Soil Types and Wilderness Areas)
    binary_cols = [c for c in train_df.columns if c not in cont_cols]

    # --- Feature Engineering ---
    def engineer_features(df):
        df = df.copy()

        # Cyclical Augmentation for Aspect
        df["Aspect_Sin"] = np.sin(np.radians(df["Aspect"]))
        df["Aspect_Cos"] = np.cos(np.radians(df["Aspect"]))

        # Geometric Magnitude: Euclidean Distance to Hydrology
        df["Hydrology_Euclidean"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrol"] ** 2
            + df["Vertical_Distance_To_Hydrolog"] ** 2
        )

        # Directional Preservation: Absolute Hydrology Elevation
        df["Hydrology_Elevation"] = (
            df["Elevation"] - df["Vertical_Distance_To_Hydrolog"]
        )

        # Global Context: Mean Distance to Amenities
        df["Mean_Amenities"] = df[
            [
                "Horizontal_Distance_To_Hydrol",
                "Horizontal_Distance_To_Roadwa",
                "Horizontal_Distance_To_Fire_P",
            ]
        ].mean(axis=1)

        return df

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Update Continuous Columns list
    new_features = [
        "Aspect_Sin",
        "Aspect_Cos",
        "Hydrology_Euclidean",
        "Hydrology_Elevation",
        "Mean_Amenities",
    ]
    all_cont_cols = cont_cols + new_features

    # Extract Feature Subsets
    X_train_cont = train_df[all_cont_cols].values.astype(np.float32)
    X_val_cont = val_df[all_cont_cols].values.astype(np.float32)
    X_test_cont = test_df[all_cont_cols].values.astype(np.float32)

    X_train_bin = train_df[binary_cols].values.astype(np.float32)
    X_val_bin = val_df[binary_cols].values.astype(np.float32)
    X_test_bin = test_df[binary_cols].values.astype(np.float32)

    # --- Dual-View Transformation ---

    # View 1: Physical (StandardScaler) - Preserves linear relationships
    scaler = StandardScaler()
    X_train_phys = scaler.fit_transform(X_train_cont)
    X_val_phys = scaler.transform(X_val_cont)
    X_test_phys = scaler.transform(X_test_cont)

    # View 2: Statistical (QuantileTransformer) - Enforces Gaussian distribution
    # Subsample is set to ensure reasonable fit time while capturing distribution
    qt = QuantileTransformer(
        output_distribution="normal", random_state=SEED, subsample=200000
    )
    X_train_stat = qt.fit_transform(X_train_cont)
    X_val_stat = qt.transform(X_val_cont)
    X_test_stat = qt.transform(X_test_cont)

    # Concatenate: [Physical View, Statistical View, Binary Features]
    train_X = np.hstack([X_train_phys, X_train_stat, X_train_bin])
    val_X = np.hstack([X_val_phys, X_val_stat, X_val_bin])
    test_X = np.hstack([X_test_phys, X_test_stat, X_test_bin])

    # Save to cache
    np.save(files["train_X"], train_X)
    np.save(files["train_y"], train_y)
    np.save(files["val_X"], val_X)
    np.save(files["val_y"], val_y)
    np.save(files["test_X"], test_X)
    np.save(files["test_ids"], test_ids)

    print("Data processed and cached.")

    # Create Datasets
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)
    test_dataset = CoverTypeDataset(test_X)

    input_dim = train_X.shape[1]
    num_classes = len(np.unique(train_y))

    return train_dataset, val_dataset, test_dataset, input_dim, num_classes, test_ids
