import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import library.config as config
from library.config import SEQ_LEN, FEATURE_NAMES
from library.features import prepare_datasets


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Reshapes flat dataframe into sequences of length SEQ_LEN (80).
    """

    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        self.is_test = is_test

        # Verify data integrity
        if len(df) % SEQ_LEN != 0:
            raise ValueError(
                f"Data length {len(df)} is not divisible by SEQ_LEN {SEQ_LEN}"
            )

        self.n_breaths = len(df) // SEQ_LEN

        # 1. Extract and Reshape Features
        # df[FEATURE_NAMES] is already scaled by RobustScaler in prepare_datasets
        X_np = df[FEATURE_NAMES].values.astype(np.float32)
        self.X = torch.from_numpy(
            X_np.reshape(self.n_breaths, SEQ_LEN, len(FEATURE_NAMES))
        )

        # 2. Extract and Reshape u_out (for loss masking)
        # We keep a separate tensor for u_out to easily mask the loss function
        u_out_np = df["u_out"].values.astype(np.float32)
        self.u_out = torch.from_numpy(u_out_np.reshape(self.n_breaths, SEQ_LEN))

        # 3. Extract and Reshape IDs (for submission)
        ids_np = df["id"].values.astype(np.int64)
        self.ids = torch.from_numpy(ids_np.reshape(self.n_breaths, SEQ_LEN))

        # 4. Extract and Reshape Targets (if not test)
        if not self.is_test:
            y_np = df["pressure"].values.astype(np.float32)
            self.y = torch.from_numpy(y_np.reshape(self.n_breaths, SEQ_LEN))
        else:
            self.y = None

    def __len__(self):
        return self.n_breaths

    def __getitem__(self, idx):
        item = {
            "x": self.X[idx],  # Shape: (80, N_features)
            "u_out": self.u_out[idx],  # Shape: (80,)
            "ids": self.ids[idx],  # Shape: (80,)
        }

        if not self.is_test:
            item["y"] = self.y[idx]  # Shape: (80,)

        return item


def get_ventilator_datasets(load_cached_data: bool = True):
    """
    Loads data, performs debug sampling if configured, and returns PyTorch Datasets.

    Args:
        load_cached_data (bool): Whether to attempt loading pre-engineered parquet files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # 1. Load and Engineer Data (handled by library.features with caching)
    train_df, val_df, test_df = prepare_datasets(load_cached_data=load_cached_data)

    # 2. Handle Debug Mode (Sample breaths to reduce size)
    if config.DEBUG:
        print(
            f"DEBUG mode enabled: Sampling {config.DEBUG_SAMPLE_SIZE} breaths per dataset."
        )

        def sample_by_breath(df, n_samples):
            unique_breaths = df["breath_id"].unique()
            if len(unique_breaths) > n_samples:
                # Deterministic slicing for reproducibility
                selected_breaths = unique_breaths[:n_samples]
                return df[df["breath_id"].isin(selected_breaths)].copy()
            return df

        train_df = sample_by_breath(train_df, config.DEBUG_SAMPLE_SIZE)
        val_df = sample_by_breath(val_df, config.DEBUG_SAMPLE_SIZE)
        test_df = sample_by_breath(test_df, config.DEBUG_SAMPLE_SIZE)

    # 3. Instantiate Datasets
    print("Converting DataFrames to PyTorch Tensors...")
    train_dataset = VentilatorDataset(train_df, is_test=False)
    val_dataset = VentilatorDataset(val_df, is_test=False)
    test_dataset = VentilatorDataset(test_df, is_test=True)

    return train_dataset, val_dataset, test_dataset
