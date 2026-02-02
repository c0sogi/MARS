import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.feature_engineering import process_data


class ForestDataset(Dataset):
    """
    PyTorch Dataset for the Forest Cover Type prediction task.
    """

    def __init__(self, X, y=None):
        """
        Args:
            X (np.ndarray): Feature matrix (float32).
            y (np.ndarray, optional): Target vector (int64). Defaults to None.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def get_cv_folds(y, n_folds=Config.N_FOLDS, seed=Config.SEED):
    """
    Generates Stratified K-Fold indices.

    Args:
        y (np.ndarray): Target labels.
        n_folds (int): Number of folds.
        seed (int): Random seed.

    Returns:
        list: List of tuples (train_idx, val_idx).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def load_data(load_cached_data=True):
    """
    Wrapper around library.feature_engineering.process_data to load the base dataset.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training targets.
        X_test (pd.DataFrame): Test features.
        test_ids (np.ndarray): Test IDs.
    """
    return process_data(load_cached_data=load_cached_data)


def prepare_nn_data(load_cached_data=True):
    """
    Prepares data specifically for the Neural Network:
    1. Loads base processed data.
    2. Applies QuantileTransformer (GaussRank) to continuous features.
    3. Caches the scaled numpy arrays for efficiency.

    Args:
        load_cached_data (bool): Whether to attempt loading scaled data from cache.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training targets.
        X_test_scaled (np.ndarray): Scaled test features.
        test_ids (np.ndarray): Test IDs.
    """
    # Define cache paths for NN-specific data
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X_train_nn = os.path.join(cache_dir, "X_train_nn.npy")
    path_X_test_nn = os.path.join(cache_dir, "X_test_nn.npy")

    # We rely on process_data caching for y_train and test_ids,
    # but we load them here to return a complete set.

    # Check if NN specific cache exists
    nn_cache_exists = os.path.exists(path_X_train_nn) and os.path.exists(path_X_test_nn)

    if load_cached_data and nn_cache_exists:
        print(f"Loading scaled NN data from cache: {cache_dir}")
        X_train_scaled = np.load(path_X_train_nn)
        X_test_scaled = np.load(path_X_test_nn)

        # Load targets/ids using the standard loader (fast if cached)
        _, y_train, _, test_ids = load_data(load_cached_data=True)

        return X_train_scaled, y_train, X_test_scaled, test_ids

    print(
        "NN Cache not found or reload requested. Processing and scaling data for Neural Network..."
    )

    # 1. Load base data (DataFrames)
    X_train_df, y_train, X_test_df, test_ids = load_data(
        load_cached_data=load_cached_data
    )

    # 2. Identify columns to scale
    # Scale defined numeric columns AND generated interaction columns (containing '_x_')
    cols_to_scale = [
        c for c in X_train_df.columns if c in Config.NUMERIC_COLS or "_x_" in c
    ]

    # Columns that are not scaled (binary/categorical)
    cols_passthrough = [c for c in X_train_df.columns if c not in cols_to_scale]

    print(
        f"Scaling {len(cols_to_scale)} features using QuantileTransformer (Normal distribution)..."
    )

    # 3. Apply QuantileTransformer
    # We convert to float32 to save memory
    scaler = QuantileTransformer(
        output_distribution="normal",
        random_state=Config.SEED,
        subsample=min(
            200000, len(X_train_df)
        ),  # Subsample for speed if dataset is huge
    )

    # Fit on Train
    X_train_scaled_part = scaler.fit_transform(X_train_df[cols_to_scale]).astype(
        np.float32
    )
    X_test_scaled_part = scaler.transform(X_test_df[cols_to_scale]).astype(np.float32)

    # Get passthrough parts
    X_train_pass = X_train_df[cols_passthrough].values.astype(np.float32)
    X_test_pass = X_test_df[cols_passthrough].values.astype(np.float32)

    # Concatenate: We keep the order consistent
    # To maintain column correspondence, we should ideally reconstruct, but for NN
    # the order just needs to be consistent between train and test.
    # We will concatenate [Scaled, Passthrough]
    X_train_final = np.hstack([X_train_scaled_part, X_train_pass])
    X_test_final = np.hstack([X_test_scaled_part, X_test_pass])

    print(f"Saving scaled NN data to cache: {cache_dir}")
    np.save(path_X_train_nn, X_train_final)
    np.save(path_X_test_nn, X_test_final)

    return X_train_final, y_train, X_test_final, test_ids
