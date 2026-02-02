import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.features import (
    load_and_clean_signal,
    generate_dual_spectrograms,
    process_and_cache_features,
)


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for Volcano Eruption Prediction.

    This dataset handles:
    1. Loading and merging metadata with engineered statistical features.
    2. Deterministic management of scalers (StandardScaler) for targets and features.
    3. On-the-fly generation of multi-resolution spectrograms.
    """

    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on the spectrogram.
            load_cached_data (bool): Whether to attempt loading features from cache.
        """
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # 1. Determine File Paths based on Mode
        if self.mode == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.features_path = Config.TRAIN_FEATURES_PATH
        elif self.mode == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.features_path = Config.VAL_FEATURES_PATH
        elif self.mode == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
            self.features_path = Config.TEST_FEATURES_PATH
        else:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'train', 'val', or 'test'."
            )

        # 2. Load or Generate Statistical Features
        # This function handles the caching logic for the features dataframe
        self.features_df = process_and_cache_features(
            self.metadata_path,
            self.features_path,
            load_cached_data=self.load_cached_data,
        )

        # 3. Load Metadata
        # We need this for 'file_path' and 'time_to_eruption' which might not be in the features parquet
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.meta_df = pd.read_csv(self.metadata_path)

        # 4. Merge DataFrames
        # Join on 'segment_id' to combine engineered features with file paths and targets
        self.df = pd.merge(self.features_df, self.meta_df, on="segment_id", how="inner")

        # 5. Debug Mode
        if Config.DEBUG:
            print(
                f"[Dataset] DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows for {self.mode}."
            )
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # 6. Identify Feature Columns
        # Filter out metadata columns to keep only the engineered statistics
        exclude_cols = {"segment_id", "time_to_eruption", "file_path"}
        self.feature_cols = sorted(
            [c for c in self.df.columns if c not in exclude_cols]
        )

        # 7. Manage Scalers (Target and Features)
        self._manage_scalers()

    def _manage_scalers(self):
        """
        Computes scalers on training data and saves them.
        Loads scalers for validation/test data.
        """
        # Ensure working directory exists for scalers
        os.makedirs(os.path.dirname(Config.TARGET_MEAN_PATH), exist_ok=True)

        # --- Target Scaler (StandardScaler) ---
        if self.mode == "train":
            # Compute statistics
            targets = self.df["time_to_eruption"].values.astype(np.float32)
            self.target_mean = np.mean(targets)
            self.target_std = np.std(targets)

            # Save to cache
            np.save(Config.TARGET_MEAN_PATH, self.target_mean)
            np.save(Config.TARGET_STD_PATH, self.target_std)
            print(
                f"[Dataset] Computed and saved target scaler: Mean={self.target_mean:.4f}, Std={self.target_std:.4f}"
            )
        else:
            # Load from cache
            if os.path.exists(Config.TARGET_MEAN_PATH) and os.path.exists(
                Config.TARGET_STD_PATH
            ):
                self.target_mean = np.load(Config.TARGET_MEAN_PATH)
                self.target_std = np.load(Config.TARGET_STD_PATH)
            else:
                # Fallback if scalers missing (e.g. inference without training)
                print(
                    "[Dataset] Warning: Target scaler files not found. Using identity scaling."
                )
                self.target_mean = 0.0
                self.target_std = 1.0

        # --- Feature Scaler (StandardScaler) ---
        if self.mode == "train":
            # Compute statistics
            feats = self.df[self.feature_cols].values.astype(np.float32)
            self.feat_mean = np.mean(feats, axis=0)
            self.feat_scale = np.std(feats, axis=0)

            # Handle constant features (std=0) to avoid division by zero
            self.feat_scale[self.feat_scale == 0] = 1.0

            # Save to cache
            np.save(Config.STATS_SCALER_MEAN_PATH, self.feat_mean)
            np.save(Config.STATS_SCALER_SCALE_PATH, self.feat_scale)
            print(
                f"[Dataset] Computed and saved feature scaler to {Config.STATS_SCALER_MEAN_PATH}"
            )
        else:
            # Load from cache
            if os.path.exists(Config.STATS_SCALER_MEAN_PATH) and os.path.exists(
                Config.STATS_SCALER_SCALE_PATH
            ):
                self.feat_mean = np.load(Config.STATS_SCALER_MEAN_PATH)
                self.feat_scale = np.load(Config.STATS_SCALER_SCALE_PATH)
            else:
                print(
                    "[Dataset] Warning: Feature scaler files not found. Using identity scaling."
                )
                self.feat_mean = np.zeros(len(self.feature_cols))
                self.feat_scale = np.ones(len(self.feature_cols))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------------------------------------------------------
        # 1. Spectrogram Branch (20 Channels)
        # ---------------------------------------------------------
        # Construct absolute file path
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load and clean raw signal (fills NaNs with 0)
        signal_df = load_and_clean_signal(file_path)

        # Generate stacked dual-view spectrograms
        # Returns Tensor of shape (20, 128, Time)
        spectrogram = generate_dual_spectrograms(signal_df)

        # Apply Augmentation (if provided, e.g., SpecAugment)
        if self.transform:
            spectrogram = self.transform(spectrogram)

        # ---------------------------------------------------------
        # 2. Statistical Feature Branch
        # ---------------------------------------------------------
        # Extract features
        feat_vals = row[self.feature_cols].values.astype(np.float32)

        # Normalize features
        feat_vals = (feat_vals - self.feat_mean) / self.feat_scale

        # Convert to Tensor
        features = torch.tensor(feat_vals, dtype=torch.float32)

        # ---------------------------------------------------------
        # 3. Target Variable
        # ---------------------------------------------------------
        if self.mode == "test":
            # Dummy target for test set
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            raw_target = row["time_to_eruption"]

            # Scale target (StandardScaler)
            scaled_target = (raw_target - self.target_mean) / self.target_std
            target = torch.tensor(scaled_target, dtype=torch.float32)

        return spectrogram, features, target
