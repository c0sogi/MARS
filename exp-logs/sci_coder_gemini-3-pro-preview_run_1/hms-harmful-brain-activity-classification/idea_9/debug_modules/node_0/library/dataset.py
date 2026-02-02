import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict

from library.config import Config
from library.utils import seed_everything


class EEGDataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        load_cached_data: bool = True,
        sample_size: Optional[int] = None,
    ):
        """
        Args:
            mode: 'train', 'val', or 'test'.
            load_cached_data: Whether to load processed data from cache.
            sample_size: If set, limits the dataset size (for debugging).
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.config = Config

        # Define target columns (probabilities)
        # Config.CLASS_NAMES contains '_vote', we need '_prob' for targets
        self.target_cols = [
            c.replace("_vote", "_prob") for c in self.config.CLASS_NAMES
        ]

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(self.config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(self.config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(self.config.TEST_CSV)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Debugging: Subsample
        if sample_size is not None:
            self.df = self.df.iloc[:sample_size].reset_index(drop=True)

        # Ensure cache directory exists
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)

        # Augmentations
        self.transform_spec = None
        if mode == "train":
            self.transform_spec = A.Compose(
                [
                    A.XYMasking(
                        num_masks_x=(1, 3),
                        mask_x_length=(10, 40),
                        num_masks_y=(1, 3),
                        mask_y_length=(10, 40),
                        p=0.5,
                    )
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Generate a unique cache ID
        # For train/val: label_id is unique per annotated event
        # For test: eeg_id is unique
        if self.mode == "test":
            cache_id = f"test_{row['eeg_id']}"
        else:
            cache_id = str(row["label_id"])

        cache_path = os.path.join(self.config.CACHE_DIR, f"{cache_id}.npz")

        eeg_data = None
        spec_data = None

        # 1. Try Loading Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                eeg_data = cached["eeg"]
                spec_data = cached["spec"]
            except Exception:
                # Corrupt cache, reprocess
                pass

        # 2. Process if not loaded
        if eeg_data is None or spec_data is None:
            eeg_data, spec_data = self.process_data(row)
            # Save to cache
            np.savez(cache_path, eeg=eeg_data, spec=spec_data)

        # 3. Augmentations (Train only)
        if self.mode == "train":
            # EEG Augmentation: Channel Dropout
            eeg_data = self.apply_channel_dropout(eeg_data)

            # Spectrogram Augmentation: SpecAugment via Albumentations
            # Albumentations expects (H, W, C), our spec is (H, W) or (1, H, W)
            # We treat it as an image
            if self.transform_spec:
                aug = self.transform_spec(image=spec_data)
                spec_data = aug["image"]

        # 4. Prepare Tensors
        # EEG: (Channels, Time) -> FloatTensor
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)

        # Spec: (Freq, Time) -> Add channel dim -> (1, Freq, Time)
        spec_tensor = torch.tensor(spec_data, dtype=torch.float32).unsqueeze(0)

        # 5. Targets
        if self.mode == "test":
            # Dummy target for test
            target_tensor = torch.zeros(len(self.target_cols), dtype=torch.float32)
        else:
            targets = row[self.target_cols].values.astype(np.float32)
            target_tensor = torch.tensor(targets, dtype=torch.float32)

        return {
            "eeg": eeg_tensor,
            "spec": spec_tensor,
            "target": target_tensor,
            "eeg_id": row["eeg_id"] if "eeg_id" in row else 0,
        }

    def process_data(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads raw files and processes them into numpy arrays.
        """
        # --- Process EEG ---
        eeg_rel_path = row["eeg_path"]
        eeg_full_path = os.path.join(self.config.INPUT_DIR, eeg_rel_path)

        try:
            raw_eeg = pd.read_parquet(eeg_full_path)
        except Exception:
            # Fallback for missing files (should not happen with validated metadata)
            raw_eeg = pd.DataFrame(np.zeros((10000, 20)))

        # Determine Offset
        if self.mode == "test":
            offset_sec = 0
        else:
            offset_sec = int(row["eeg_label_offset_seconds"])

        # Sampling parameters
        fs_raw = self.config.EEG_RAW_SAMPLE_RATE  # 200
        fs_target = self.config.EEG_TARGET_SAMPLE_RATE  # 100
        duration = self.config.EEG_DURATION  # 50

        start_idx = int(offset_sec * fs_raw)
        end_idx = start_idx + int(duration * fs_raw)

        # Slice
        if len(raw_eeg) < end_idx:
            # Pad if necessary
            pad_len = end_idx - len(raw_eeg)
            eeg_segment = raw_eeg.iloc[start_idx:].values
            eeg_segment = np.pad(eeg_segment, ((0, pad_len), (0, 0)), mode="constant")
        else:
            eeg_segment = raw_eeg.iloc[start_idx:end_idx].values

        # Select Columns (Exclude non-EEG if any, but keep EKG)
        # The dataset typically has 20 columns: 19 EEG + 1 EKG.
        # We use all available columns assuming they match expected 20.
        # If 'time' is in columns, drop it.
        if "time" in raw_eeg.columns:
            # Find index of 'time' and exclude
            # But usually we just take the values.
            # Let's ensure we take the first 20 feature columns.
            # The competition data usually has named columns.
            cols = [c for c in raw_eeg.columns if c != "time"]
            eeg_segment = raw_eeg.iloc[start_idx:end_idx][cols].values

            # Handle padding again if we used iloc on dataframe
            if len(eeg_segment) < (end_idx - start_idx):
                pad_len = (end_idx - start_idx) - len(eeg_segment)
                eeg_segment = np.pad(
                    eeg_segment, ((0, pad_len), (0, 0)), mode="constant"
                )

        # Downsample (200Hz -> 100Hz)
        # Simple slicing ::2
        step = int(fs_raw / fs_target)
        eeg_segment = eeg_segment[::step, :]

        # Handle NaNs (replace with 0)
        eeg_segment = np.nan_to_num(eeg_segment, nan=0.0)

        # Normalize (Instance Normalization: Channel-wise)
        # Shape: (Time, Channels) -> (Channels, Time) for calculation
        eeg_segment = eeg_segment.T
        mean = np.mean(eeg_segment, axis=1, keepdims=True)
        std = np.std(eeg_segment, axis=1, keepdims=True)
        eeg_segment = (eeg_segment - mean) / (std + 1e-6)

        # Clip outliers
        eeg_segment = np.clip(eeg_segment, -10, 10)

        # --- Process Spectrogram ---
        spec_rel_path = row["spectrogram_path"]
        spec_full_path = os.path.join(self.config.INPUT_DIR, spec_rel_path)

        try:
            raw_spec = pd.read_parquet(spec_full_path)
        except Exception:
            raw_spec = pd.DataFrame(np.zeros((300, 401)))  # Dummy

        # Determine Offset
        # Window is 10 minutes = 600 seconds
        if self.mode == "test":
            # Test spectrograms are exactly 10 mins
            spec_window = raw_spec
        else:
            offset_spec = int(row["spectrogram_label_offset_seconds"])
            # Filter by time column
            if "time" in raw_spec.columns:
                spec_window = raw_spec.loc[
                    (raw_spec["time"] >= offset_spec)
                    & (raw_spec["time"] < offset_spec + self.config.SPEC_DURATION)
                ]
            else:
                # Fallback if time column missing (unlikely for train)
                spec_window = raw_spec

        # Drop time column
        if "time" in spec_window.columns:
            spec_window = spec_window.drop(columns=["time"])

        spec_arr = spec_window.values

        # Fill NaNs
        spec_arr = np.nan_to_num(spec_arr, nan=0.0)

        # Log Transform
        spec_arr = np.log1p(spec_arr)

        # Resize to (512, 512)
        # Current shape: (Time_Steps, Freq_Bins)
        # Resize expects (Width, Height) -> (Time, Freq)
        # We want output (Freq, Time) = (512, 512)
        # So we resize to (512, 512)
        spec_arr = cv2.resize(
            spec_arr, dsize=self.config.SPEC_SIZE, interpolation=cv2.INTER_LINEAR
        )

        # Normalize (Global Standardize for this sample)
        s_mean = spec_arr.mean()
        s_std = spec_arr.std()
        spec_arr = (spec_arr - s_mean) / (s_std + 1e-6)

        # Ensure correct type
        eeg_segment = eeg_segment.astype(np.float32)
        spec_arr = spec_arr.astype(np.float32)

        return eeg_segment, spec_arr

    def apply_channel_dropout(self, eeg_data: np.ndarray, p: float = 0.2) -> np.ndarray:
        """
        Randomly sets some channels to zero.
        Input: (Channels, Time)
        """
        if np.random.rand() < 0.5:  # Apply augmentation with 50% prob
            num_channels = eeg_data.shape[0]
            # Select channels to drop
            mask = np.random.rand(num_channels) > p
            # Ensure at least one channel remains
            if mask.sum() == 0:
                mask[0] = True

            # Apply mask (broadcasting over time dimension)
            eeg_data = eeg_data * mask[:, np.newaxis]

        return eeg_data
