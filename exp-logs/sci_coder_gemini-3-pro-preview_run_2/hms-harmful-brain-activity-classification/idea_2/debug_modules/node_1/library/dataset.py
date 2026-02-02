import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_full_path
from library.transforms import load_and_process_eeg, process_kaggle_spec, get_transforms


class EEGMultiModalDataset(Dataset):
    """
    Dual-stream dataset for loading raw EEG and Kaggle Spectrograms.
    Stream 1: Raw EEG -> Mel Spectrogram (cached for validation).
    Stream 2: Pre-computed Kaggle Spectrograms (sliced and processed).
    """

    def __init__(self, df, config, mode="train"):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode

        # Caching strategy:
        # Train: Generate on-the-fly (load_cached_data=False) to save space.
        # Val: Cache to disk (load_cached_data=True) to speed up evaluation.
        # Test: Generate on-the-fly.
        self.load_cached_data = mode == "val"

        # Define target columns
        # The metadata script generates *_prob columns which are normalized votes.
        if mode in ["train", "val"]:
            self.target_cols = [f"{c.split('_')[0]}_prob" for c in config.CLASS_NAMES]
            # Verify columns exist
            if not all(col in self.df.columns for col in self.target_cols):
                # Fallback to vote columns if prob columns missing (unlikely with provided metadata)
                self.target_cols = config.CLASS_NAMES

        # Initialize transforms
        self.transform = get_transforms(config, mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------------------------------------------------------------------
        # Stream 1: Raw EEG -> Mel Spectrogram
        # ---------------------------------------------------------------------
        eeg_path = get_full_path(row["eeg_path"])
        eeg_offset = row["eeg_label_offset_seconds"]
        # Unique cache ID: eeg_id + sub_id
        cache_id = f"{int(row['eeg_id'])}_{int(row.get('eeg_sub_id', 0))}"

        eeg_spec = load_and_process_eeg(
            file_path=eeg_path,
            start_time=eeg_offset,
            duration=self.config.EEG_DURATION,
            cache_id=cache_id,
            config=self.config,
            load_cached_data=self.load_cached_data,
        )

        # ---------------------------------------------------------------------
        # Stream 2: Kaggle Spectrogram
        # ---------------------------------------------------------------------
        spec_path = get_full_path(row["spectrogram_path"])
        spec_offset = row["spectrogram_label_offset_seconds"]

        try:
            # Load the full spectrogram parquet
            spec_df = pd.read_parquet(spec_path)

            # Slice the specific 10-minute window (600 seconds)
            # Strategy: Use 'time' column if available, else assume 0.5Hz sampling
            if "time" in spec_df.columns:
                mask = (spec_df["time"] >= spec_offset) & (
                    spec_df["time"] < spec_offset + self.config.SPEC_DURATION
                )
                spec_slice = spec_df[mask].drop(columns=["time"])
            else:
                # Fallback: 2 seconds per row
                start_row = int(spec_offset / 2)
                end_row = start_row + int(self.config.SPEC_DURATION / 2)
                spec_slice = spec_df.iloc[start_row:end_row]

            # Handle empty slice (edge case)
            if spec_slice.empty:
                raise ValueError("Empty spectrogram slice")

            kaggle_spec = process_kaggle_spec(spec_slice, self.config)

        except Exception:
            # Return zero tensor on failure
            kaggle_spec = np.zeros(
                (self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 1), dtype=np.float32
            )

        # ---------------------------------------------------------------------
        # Augmentation & Formatting
        # ---------------------------------------------------------------------
        # Apply Albumentations (expects H, W, C)
        # Returns dictionary with 'image' key containing Tensor (C, H, W)
        aug_eeg = self.transform(image=eeg_spec)["image"]
        aug_kaggle = self.transform(image=kaggle_spec)["image"]

        # Expand channels to match Backbone (e.g., 3 channels for EfficientNet)
        if self.config.IN_CHANNELS == 3:
            aug_eeg = aug_eeg.repeat(3, 1, 1)
            aug_kaggle = aug_kaggle.repeat(3, 1, 1)

        # ---------------------------------------------------------------------
        # Return
        # ---------------------------------------------------------------------
        data = {"eeg_spec": aug_eeg, "kaggle_spec": aug_kaggle}

        if self.mode in ["train", "val"]:
            target = row[self.target_cols].values.astype(np.float32)
            # Normalize to sum to 1 (handling potential float precision issues)
            total = target.sum()
            if total > 0:
                target = target / total
            else:
                # Uniform distribution fallback
                target = np.ones_like(target) / len(target)

            data["target"] = torch.tensor(target, dtype=torch.float32)

        return data
