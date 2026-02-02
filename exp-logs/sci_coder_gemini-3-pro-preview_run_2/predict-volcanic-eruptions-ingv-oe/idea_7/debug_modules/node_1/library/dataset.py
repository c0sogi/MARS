import os
import torch
import pandas as pd
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config
from library.utils import TargetScaler
from library.feature_engineering import process_segment, FeatureEngineer


class VolcanoDataset(Dataset):
    """
    PyTorch Dataset for the Volcano Eruption Prediction task.
    Combines on-the-fly spectrogram generation with pre-computed tabular features.
    """

    def __init__(self, mode="train"):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
        """
        self.mode = mode
        self.config = Config

        # 1. Determine paths based on mode
        if self.mode == "train":
            self.feature_path = self.config.TRAIN_FEATURES_PATH
            self.metadata_path = self.config.TRAIN_METADATA
        elif self.mode == "val":
            self.feature_path = self.config.VAL_FEATURES_PATH
            self.metadata_path = self.config.VAL_METADATA
        elif self.mode == "test":
            self.feature_path = self.config.TEST_FEATURES_PATH
            self.metadata_path = self.config.TEST_METADATA
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # 2. Ensure Features Exist (Caching Mechanism)
        # If the parquet file is missing, trigger the feature engineering pipeline.
        if not os.path.exists(self.feature_path):
            print(
                f"[{self.mode.upper()}] Features not found at {self.feature_path}. Generating..."
            )
            fe = FeatureEngineer()
            fe.run(load_cached_data=True)

        # 3. Load Data
        self.df_features = pd.read_parquet(self.feature_path)

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.df_meta = pd.read_csv(self.metadata_path)

        # 4. Merge Features with Metadata (to get file_paths)
        # Inner join ensures we only have records that exist in both
        self.data = pd.merge(
            self.df_features,
            self.df_meta[["segment_id", "file_path"]],
            on="segment_id",
            how="inner",
        )

        # 5. Handle Debug Mode
        if self.config.DEBUG:
            print(
                f"[{self.mode.upper()}] Debug mode active. Slicing dataset to {self.config.DEBUG_SIZE} samples."
            )
            self.data = self.data.iloc[: self.config.DEBUG_SIZE].reset_index(drop=True)

        # 6. Identify Feature Columns
        # Exclude metadata and target columns
        exclude_cols = ["segment_id", "target", "file_path"]
        self.feature_cols = [c for c in self.data.columns if c not in exclude_cols]

        # 7. Setup Target Scaler
        self.scaler = TargetScaler()
        if self.mode in ["train", "val"]:
            if os.path.exists(self.config.TARGET_MEAN_PATH) and os.path.exists(
                self.config.TARGET_STD_PATH
            ):
                self.scaler.load(
                    self.config.TARGET_MEAN_PATH, self.config.TARGET_STD_PATH
                )
            else:
                # If stats are missing but we are in train/val, we can't scale correctly.
                # However, FeatureEngineer.run() should have generated them.
                print(
                    "Warning: Target scaler stats not found. Targets will not be scaled correctly."
                )

        # 8. Setup SpecAugment (Training Only)
        # Conservative parameters: Mask <15% of dimensions
        # n_mels=128 -> freq_mask=19
        # time_steps~235 -> time_mask=35
        if self.mode == "train":
            self.spec_augment = torch.nn.Sequential(
                T.FrequencyMasking(freq_mask_param=19),
                T.TimeMasking(time_mask_param=35),
            )
        else:
            self.spec_augment = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        segment_id = int(row["segment_id"])

        # ---------------------------------------------------------
        # A. Spectrogram Generation (On-the-fly)
        # ---------------------------------------------------------
        # Metadata path is relative to input dir
        full_path = os.path.join(self.config.INPUT_DIR, row["file_path"])

        # process_segment returns (spec_tensor, stats_dict). We discard stats_dict.
        spec_tensor, _ = process_segment(full_path, return_spec=True)

        # Apply SpecAugment if in training mode
        if self.spec_augment is not None:
            spec_tensor = self.spec_augment(spec_tensor)

        # ---------------------------------------------------------
        # B. Tabular Features (Pre-computed)
        # ---------------------------------------------------------
        # Extract feature vector and convert to tensor
        features_np = row[self.feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features_np, dtype=torch.float32)

        # ---------------------------------------------------------
        # C. Target Variable
        # ---------------------------------------------------------
        target_tensor = torch.tensor(0.0, dtype=torch.float32)  # Default for test

        if self.mode in ["train", "val"]:
            if "target" in row:
                raw_target = row["target"]
                # Apply scaling
                if self.scaler.mean is not None:
                    scaled_target = self.scaler.transform(raw_target)
                else:
                    scaled_target = raw_target

                target_tensor = torch.tensor(scaled_target, dtype=torch.float32)

        return spec_tensor, features_tensor, target_tensor, segment_id
