import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config
from library.features import FeatureEngineer


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.

    Loads pre-processed data using FeatureEngineer, reshapes it into time-series sequences,
    and provides access to inputs, masks, and targets.
    """

    def __init__(self, split="train", load_cached=True):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            load_cached (bool): Whether to load from parquet cache if available.
        """
        self.split = split

        # Initialize Feature Engineer
        fe = FeatureEngineer()

        # Load Data based on split
        if split in ["train", "val"]:
            train_df, val_df = fe.process_train_val(load_cached=load_cached)
            self.df = train_df if split == "train" else val_df
        elif split == "test":
            self.df = fe.process_test(load_cached=load_cached)
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', or 'test'."
            )

        # Define Feature Columns
        # We include continuous scaled features + binary u_out
        self.feature_cols = [
            "time_step",
            "u_in",
            "u_out",  # Included in input as a feature
            "R",
            "C",
            "volume",
            "R__u_in",
            "vol__C",
        ]

        # Add Lags if enabled in Config
        if Config.use_lags:
            self.feature_cols.extend([f"u_in_lag{i}" for i in Config.lag_steps])

        # Add Diffs if enabled in Config
        if Config.use_diffs:
            self.feature_cols.extend(["u_in_diff1", "u_in_diff2"])

        # Define Special Columns
        self.target_col = "pressure"
        self.mask_col = (
            "u_out_raw"  # Used for loss masking (1=expiratory, 0=inspiratory)
        )

        # Process and reshape into tensors
        self._prepare_tensors()

    def _prepare_tensors(self):
        """
        Converts the flat DataFrame into 3D tensors (N_breaths, Seq_Len, Features).
        """
        num_rows = len(self.df)
        seq_len = Config.seq_len

        if num_rows % seq_len != 0:
            raise ValueError(
                f"Dataset length {num_rows} is not divisible by sequence length {seq_len}."
            )

        self.num_breaths = num_rows // seq_len

        # 1. Prepare Inputs (X)
        # Shape: (N, 80, F)
        x_data = self.df[self.feature_cols].values.astype(np.float32)
        self.x = torch.from_numpy(
            x_data.reshape(self.num_breaths, seq_len, len(self.feature_cols))
        )

        # 2. Prepare Mask (u_out_raw)
        # Shape: (N, 80)
        # Note: u_out_raw is 1 for expiratory phase, 0 for inspiratory.
        # The loss function will likely use (1 - u_out) to mask out expiratory phase.
        u_out_data = self.df[self.mask_col].values.astype(np.float32)
        self.u_out = torch.from_numpy(u_out_data.reshape(self.num_breaths, seq_len))

        # 3. Prepare Targets (y)
        # Shape: (N, 80)
        if self.split != "test":
            y_data = self.df[self.target_col].values.astype(np.float32)
            self.y = torch.from_numpy(y_data.reshape(self.num_breaths, seq_len))
        else:
            # For test set, create dummy targets
            self.y = torch.zeros((self.num_breaths, seq_len), dtype=torch.float32)

        # Clear DataFrame to free memory
        del self.df

    def __len__(self):
        return self.num_breaths

    def __getitem__(self, idx):
        """
        Returns:
            x (Tensor): Input features (Seq_Len, Features)
            u_out (Tensor): Mask signal (Seq_Len)
            y (Tensor): Target pressure (Seq_Len)
        """
        return self.x[idx], self.u_out[idx], self.y[idx]
