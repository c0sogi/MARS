import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.data_transforms import MultiResolutionSTFT, get_transforms


class EEGDataset(Dataset):
    """
    PyTorch Dataset for Brain Activity Classification.
    Handles loading of EEG and Spectrogram data, caching raw inputs to RAM/Disk,
    and applying multi-resolution preprocessing on-the-fly.
    """

    def __init__(
        self,
        mode="train",
        config=Config,
        load_cached_data=True,
        subset_size=None,
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            config (class): Configuration class.
            load_cached_data (bool): Whether to use cached .npy files.
            subset_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.config = config
        self.subset_size = subset_size

        # Initialize preprocessing modules
        self.multi_res_stft = MultiResolutionSTFT()
        self.transforms_a = get_transforms(mode=mode, data_type="eeg")
        self.transforms_b = get_transforms(mode=mode, data_type="spec")

        # Load Metadata
        self.df = self._load_metadata()
        if self.subset_size:
            self.df = self.df.iloc[: self.subset_size].reset_index(drop=True)

        # Prepare Cache Paths
        self.cache_dir = self.config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.eeg_cache_path = os.path.join(self.cache_dir, f"{mode}_eeg_raw.npy")
        self.spec_cache_path = os.path.join(self.cache_dir, f"{mode}_spec_raw.npy")
        self.target_cache_path = os.path.join(self.cache_dir, f"{mode}_targets.npy")

        # Load or Generate Data
        self.eeg_data, self.spec_data, self.targets = self._load_data(load_cached_data)

    def _load_metadata(self):
        """Loads the appropriate metadata CSV based on mode."""
        if self.mode == "train":
            return pd.read_csv(self.config.TRAIN_CSV)
        elif self.mode == "val":
            return pd.read_csv(self.config.VAL_CSV)
        elif self.mode == "test":
            return pd.read_csv(self.config.TEST_CSV)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _load_data(self, load_cached_data):
        """
        Orchestrates the caching logic:
        1. Try to load from .npy files if load_cached_data=True.
        2. If fail, generate from scratch using parquet files.
        3. Save to .npy for future use.
        """
        if load_cached_data:
            if (
                os.path.exists(self.eeg_cache_path)
                and os.path.exists(self.spec_cache_path)
                and (self.mode == "test" or os.path.exists(self.target_cache_path))
            ):
                print(f"[{self.mode}] Loading cached raw data from {self.cache_dir}...")
                eeg_data = np.load(self.eeg_cache_path, mmap_mode="r")
                spec_data = np.load(self.spec_cache_path, mmap_mode="r")

                if self.mode != "test":
                    targets = np.load(self.target_cache_path)
                else:
                    targets = None

                # Verify length matches metadata (in case cache is stale or subset changed)
                if len(eeg_data) == len(self.df):
                    return eeg_data, spec_data, targets
                else:
                    print(f"[{self.mode}] Cache size mismatch. Regenerating...")

        # Generate data from scratch
        print(f"[{self.mode}] Processing raw data from parquet files...")
        eeg_data, spec_data, targets = self._process_raw_files()

        # Save to cache
        print(f"[{self.mode}] Saving processed data to cache...")
        np.save(self.eeg_cache_path, eeg_data)
        np.save(self.spec_cache_path, spec_data)
        if targets is not None:
            np.save(self.target_cache_path, targets)

        return eeg_data, spec_data, targets

    def _process_raw_files(self):
        """
        Iterates through metadata, reads parquet files, extracts windows,
        and aggregates into numpy arrays.
        """
        n_samples = len(self.df)

        # Pre-allocate arrays to save memory fragmentation
        # EEG: (N, 10000, 19)
        eeg_shape = (n_samples, self.config.TOTAL_SAMPLES, self.config.N_EEG_CHANNELS)
        # Spec: (N, 300, 400) - 10 mins (600s) / 2s per row = 300 rows. 400 freq bins.
        spec_shape = (n_samples, 300, 400)

        eeg_storage = np.zeros(eeg_shape, dtype=np.float32)
        spec_storage = np.zeros(spec_shape, dtype=np.float32)

        if self.mode != "test":
            targets = self.df[self.config.TARGET_COLS].values.astype(np.float32)
        else:
            targets = None

        # Optimization: Cache the last loaded dataframe to avoid re-reading for consecutive rows
        last_eeg_id = None
        last_eeg_df = None
        last_spec_id = None
        last_spec_df = None

        # Paths
        if self.mode == "test":
            eeg_dir = self.config.TEST_EEGS_DIR
            spec_dir = self.config.TEST_SPECS_DIR
        else:
            eeg_dir = self.config.TRAIN_EEGS_DIR
            spec_dir = self.config.TRAIN_SPECS_DIR

        # Column definitions for Spectrograms (LL, RL, LP, RP)
        # We assume standard Kaggle spec columns. We'll parse them dynamically once.
        spec_cols = None

        for i, row in self.df.iterrows():
            if i % 1000 == 0:
                print(f"  Processed {i}/{n_samples} samples...")

            # --- Process EEG ---
            eeg_id = row["eeg_id"]
            if eeg_id != last_eeg_id:
                eeg_path = os.path.join(eeg_dir, f"{eeg_id}.parquet")
                last_eeg_df = pd.read_parquet(
                    eeg_path, columns=self.config.EEG_CHANNELS
                )
                last_eeg_id = eeg_id

            # Calculate offset index (200 Hz)
            eeg_offset_sec = int(row.get("eeg_label_offset_seconds", 0))
            eeg_start_idx = int(eeg_offset_sec * self.config.SR)
            eeg_end_idx = eeg_start_idx + self.config.TOTAL_SAMPLES

            # Extract and handle bounds/padding if necessary
            # Note: Metadata guarantees validity, but safety check is good
            raw_eeg = last_eeg_df.values[eeg_start_idx:eeg_end_idx]

            # Handle edge case where file is shorter than expected (padding)
            if raw_eeg.shape[0] < self.config.TOTAL_SAMPLES:
                pad_len = self.config.TOTAL_SAMPLES - raw_eeg.shape[0]
                raw_eeg = np.pad(raw_eeg, ((0, pad_len), (0, 0)), mode="constant")

            eeg_storage[i] = raw_eeg

            # --- Process Spectrogram ---
            spec_id = row["spectrogram_id"]
            if spec_id != last_spec_id:
                spec_path = os.path.join(spec_dir, f"{spec_id}.parquet")
                last_spec_df = pd.read_parquet(spec_path)
                last_spec_id = spec_id

                # One-time setup for column ordering
                if spec_cols is None:
                    # Identify columns for each region
                    cols = last_spec_df.columns.tolist()
                    # Filter out 'time' if present
                    cols = [c for c in cols if c != "time"]
                    # We need to ensure consistent order: LL, RL, LP, RP
                    # And within region, sort by frequency
                    ll_cols = sorted(
                        [c for c in cols if c.startswith("LL")],
                        key=lambda x: float(x.split("_")[1]),
                    )
                    rl_cols = sorted(
                        [c for c in cols if c.startswith("RL")],
                        key=lambda x: float(x.split("_")[1]),
                    )
                    lp_cols = sorted(
                        [c for c in cols if c.startswith("LP")],
                        key=lambda x: float(x.split("_")[1]),
                    )
                    rp_cols = sorted(
                        [c for c in cols if c.startswith("RP")],
                        key=lambda x: float(x.split("_")[1]),
                    )
                    spec_cols = ll_cols + rl_cols + lp_cols + rp_cols

            # Calculate offset index (Spectrograms are 2s per row usually)
            # Metadata offset is in seconds.
            spec_offset_sec = int(row.get("spectogram_label_offset_seconds", 0))
            # 1 row = 2 seconds
            spec_start_idx = spec_offset_sec // 2
            spec_end_idx = spec_start_idx + 300  # 10 mins = 600s = 300 rows

            raw_spec = last_spec_df[spec_cols].values[spec_start_idx:spec_end_idx]

            # Handle padding
            if raw_spec.shape[0] < 300:
                pad_len = 300 - raw_spec.shape[0]
                raw_spec = np.pad(raw_spec, ((0, pad_len), (0, 0)), mode="constant")

            spec_storage[i] = raw_spec

        return eeg_storage, spec_storage, targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Retrieve Raw Data
        # EEG: (10000, 19)
        eeg_raw = self.eeg_data[idx]
        # Spec: (300, 400) - Flattened regions
        spec_raw = self.spec_data[idx]

        # 2. Process Stream A (EEG -> Multi-Res STFT)
        # Output: (128, 500, 57)
        X_a = self.multi_res_stft(eeg_raw)

        # Apply Albumentations (Normalize, SpecAugment)
        # Albumentations expects (H, W, C). Our STFT output is (Freq, Time, Channels).
        # Normalization and Augmentation happen here.
        # Output becomes Tensor (C, H, W) -> (57, 128, 500)
        aug_a = self.transforms_a(image=X_a)["image"]

        # 3. Process Stream B (Spectrogram)
        # Raw shape (300, 400). Columns are [LL(100) | RL(100) | LP(100) | RP(100)]
        # We need to reshape to (300, 100, 4) -> (Time, Freq, Channels)
        # And then resize to (256, 256)

        # Split into regions
        # Assuming 400 columns are ordered LL, RL, LP, RP
        spec_reshaped = spec_raw.reshape(300, 100, 4)  # (Time, Freq, Ch)

        # Log Transform (Log1p)
        spec_log = np.log1p(np.abs(spec_reshaped))

        # Handle NaNs
        spec_log = np.nan_to_num(spec_log, nan=0.0)

        # Albumentations expects (H, W, C).
        # Currently (Time, Freq, Ch) -> (300, 100, 4).
        # We want to resize to (256, 256).
        # Note: Albumentations Resize takes (Height, Width).
        # We treat Time as Height, Freq as Width? Or vice versa?
        # Standard is usually Freq on Y (Height), Time on X (Width).
        # So we transpose (300, 100, 4) -> (100, 300, 4)
        spec_img = np.transpose(spec_log, (1, 0, 2))  # (Freq, Time, Ch)

        # Resize to target (256, 256)
        # We use a separate resize transform or cv2.
        # Since we have a complex pipeline, let's use cv2 here for resizing before augmentations
        # or rely on Albumentations if we added Resize to get_transforms.
        # The provided get_transforms does NOT have Resize. We must do it here.
        spec_resized = np.zeros(
            (self.config.IMG_SIZE_B[0], self.config.IMG_SIZE_B[1], 4), dtype=np.float32
        )
        for c in range(4):
            # Resize each channel. cv2.resize expects (Width, Height)
            spec_resized[:, :, c] = cv2.resize(
                spec_img[:, :, c],
                (self.config.IMG_SIZE_B[1], self.config.IMG_SIZE_B[0]),
            )

        # Apply Albumentations (Normalize)
        # Output Tensor (C, H, W) -> (4, 256, 256)
        aug_b = self.transforms_b(image=spec_resized)["image"]

        # 4. Prepare Target
        if self.mode != "test":
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (aug_a, aug_b), target
        else:
            return (aug_a, aug_b), torch.zeros(self.config.NUM_CLASSES)


import cv2  # Imported here to ensure visibility in __getitem__ scope if needed
