import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
import library.config as config


class HMSDataset(Dataset):
    def __init__(self, csv_file, mode="train", augment=False, load_cached_data=True):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply data augmentation.
            load_cached_data (bool): Whether to load processed data from cache.
        """
        self.mode = mode
        self.augment = augment
        self.df = pd.read_csv(csv_file)

        # Define cache paths
        self.cache_dir = config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_suffix = f"_{mode}"
        self.eeg_cache_path = os.path.join(
            self.cache_dir, f"eeg_data{self.cache_suffix}.npy"
        )
        self.spec_cache_path = os.path.join(
            self.cache_dir, f"spec_data{self.cache_suffix}.npy"
        )
        self.target_cache_path = os.path.join(
            self.cache_dir, f"targets{self.cache_suffix}.npy"
        )

        # Load or Process Data
        if load_cached_data and self._check_cache_exists():
            print(f"Loading cached data for {mode} from {self.cache_dir}...")
            self.eeg_data = np.load(self.eeg_cache_path, mmap_mode="r")
            self.spec_data = np.load(self.spec_cache_path, mmap_mode="r")
            if mode != "test":
                self.targets = np.load(self.target_cache_path)
        else:
            print(f"Processing data for {mode} (Cache not found or ignored)...")
            self._process_and_cache()
            # Reload in mmap mode to save RAM
            self.eeg_data = np.load(self.eeg_cache_path, mmap_mode="r")
            self.spec_data = np.load(self.spec_cache_path, mmap_mode="r")
            if mode != "test":
                self.targets = np.load(self.target_cache_path)

    def _check_cache_exists(self):
        exists = os.path.exists(self.eeg_cache_path) and os.path.exists(
            self.spec_cache_path
        )
        if self.mode != "test":
            exists = exists and os.path.exists(self.target_cache_path)
        return exists

    def _process_and_cache(self):
        """
        Iterates through the dataframe, loads raw files, preprocesses them,
        and saves the result to .npy files.
        """
        n_samples = len(self.df)

        # Pre-allocate arrays
        # EEG: (N, 19, 2500)
        eeg_array = np.zeros(
            (n_samples, len(config.EEG_CHANNELS), config.EEG_SEQ_LENGTH),
            dtype=np.float32,
        )
        # Spec: (N, 4, 256, 256)
        spec_array = np.zeros(
            (n_samples, config.SPEC_CHANNELS, config.SPEC_HEIGHT, config.SPEC_WIDTH),
            dtype=np.float32,
        )

        if self.mode != "test":
            target_array = np.zeros((n_samples, config.NUM_CLASSES), dtype=np.float32)

        print(f"Starting processing of {n_samples} samples...")

        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{n_samples}")

            # ==========================
            # Process EEG
            # ==========================
            eeg_path = os.path.join(config.INPUT_DIR, row["eeg_path"])
            try:
                # Read specific columns
                eeg_df = pd.read_parquet(eeg_path, columns=config.EEG_CHANNELS)

                if self.mode == "test":
                    # Test files are exactly 50s
                    eeg_raw = eeg_df.values
                else:
                    # Train files use offset
                    offset_sec = row["eeg_label_offset_seconds"]
                    start_idx = int(offset_sec * config.EEG_RAW_SAMPLE_RATE)
                    end_idx = start_idx + int(
                        config.EEG_DURATION_SEC * config.EEG_RAW_SAMPLE_RATE
                    )

                    # Bounds check
                    if start_idx < 0:
                        start_idx = 0
                    eeg_raw = eeg_df.iloc[start_idx:end_idx].values

                    # Pad if short
                    target_len = int(
                        config.EEG_DURATION_SEC * config.EEG_RAW_SAMPLE_RATE
                    )
                    if len(eeg_raw) < target_len:
                        pad_len = target_len - len(eeg_raw)
                        eeg_raw = np.pad(eeg_raw, ((0, pad_len), (0, 0)), "constant")

                # Downsample 200Hz -> 50Hz (::4)
                eeg_sub = eeg_raw[::4, :]  # (2500, 19)

                # Handle NaNs
                eeg_sub = np.nan_to_num(eeg_sub, nan=0.0)

                # Normalize (Z-score per channel)
                eeg_sub = np.clip(eeg_sub, -1024, 1024)
                mean = np.mean(eeg_sub, axis=0)
                std = np.std(eeg_sub, axis=0)
                eeg_norm = (eeg_sub - mean) / (std + 1e-6)

                # Transpose to (Channels, Time)
                eeg_array[idx] = eeg_norm.T

            except Exception as e:
                # print(f"Error processing EEG {eeg_path}: {e}")
                pass  # Leave as zeros

            # ==========================
            # Process Spectrogram
            # ==========================
            spec_path = os.path.join(config.INPUT_DIR, row["spec_path"])
            try:
                spec_df = pd.read_parquet(spec_path)

                if self.mode == "test":
                    spec_slice = spec_df
                else:
                    offset_sec = row["spectogram_label_offset_seconds"]
                    if "time" in spec_df.columns:
                        mask = (spec_df["time"] >= offset_sec) & (
                            spec_df["time"] < offset_sec + config.SPEC_DURATION_SEC
                        )
                        spec_slice = spec_df[mask]
                    else:
                        spec_slice = spec_df  # Fallback

                if len(spec_slice) > 0:
                    # Filter columns for regions
                    cols = [c for c in spec_slice.columns if c != "time"]
                    spec_vals = spec_slice[cols]

                    regions = ["LL", "RL", "LP", "RP"]
                    region_maps = {
                        r: [c for c in cols if c.startswith(r)] for r in regions
                    }

                    processed_regions = []
                    for r in regions:
                        r_cols = region_maps[r]
                        if not r_cols:
                            r_img = np.zeros((config.SPEC_HEIGHT, config.SPEC_WIDTH))
                        else:
                            r_data = spec_vals[r_cols].values  # (Time, Freq)
                            r_data = np.nan_to_num(r_data, nan=0.0)

                            # Log Transform
                            r_data = np.log1p(r_data)

                            # Resize to (256, 256)
                            # Note: cv2.resize takes (W, H). Input is (Time, Freq).
                            # We treat Time as Width.
                            r_img = cv2.resize(
                                r_data,
                                (config.SPEC_WIDTH, config.SPEC_HEIGHT),
                                interpolation=cv2.INTER_LINEAR,
                            )

                            # Min-Max Normalize to [0, 1] per region
                            r_min, r_max = r_img.min(), r_img.max()
                            if r_max > r_min:
                                r_img = (r_img - r_min) / (r_max - r_min)
                            else:
                                r_img = np.zeros_like(r_img)

                        processed_regions.append(r_img)

                    # Stack (4, 256, 256)
                    spec_array[idx] = np.stack(processed_regions, axis=0)

            except Exception as e:
                # print(f"Error processing Spec {spec_path}: {e}")
                pass

            # ==========================
            # Process Targets
            # ==========================
            if self.mode != "test":
                target_vals = row[config.TARGET_COLS].values.astype(np.float32)
                sum_vals = target_vals.sum()
                if sum_vals > 0:
                    target_vals /= sum_vals
                else:
                    target_vals = np.ones_like(target_vals) / len(target_vals)
                target_array[idx] = target_vals

        print("Saving processed arrays to disk...")
        np.save(self.eeg_cache_path, eeg_array)
        np.save(self.spec_cache_path, spec_array)
        if self.mode != "test":
            np.save(self.target_cache_path, target_array)
        print("Caching complete.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load data (copy to allow modification by augmentations)
        eeg = self.eeg_data[idx].copy()
        spec = self.spec_data[idx].copy()

        # Augmentation
        if self.mode == "train" and self.augment:
            eeg = self._augment_eeg(eeg)
            spec = self._augment_spec(spec)

        eeg_tensor = torch.from_numpy(eeg).float()
        spec_tensor = torch.from_numpy(spec).float()

        if self.mode != "test":
            target = self.targets[idx].copy()
            target_tensor = torch.from_numpy(target).float()
            return eeg_tensor, spec_tensor, target_tensor
        else:
            return eeg_tensor, spec_tensor

    def _augment_eeg(self, eeg):
        """Random Channel Dropout"""
        if np.random.rand() < 0.5:
            num_drop = np.random.randint(1, 4)
            channels_idx = np.random.choice(eeg.shape[0], num_drop, replace=False)
            eeg[channels_idx, :] = 0.0
        return eeg

    def _augment_spec(self, spec):
        """SpecAugment (Time/Freq Masking)"""
        # Time Masking
        if np.random.rand() < 0.5:
            mask_width = np.random.randint(10, 50)
            start = np.random.randint(0, spec.shape[2] - mask_width)
            spec[:, :, start : start + mask_width] = 0.0

        # Frequency Masking
        if np.random.rand() < 0.5:
            mask_height = np.random.randint(10, 50)
            start = np.random.randint(0, spec.shape[1] - mask_height)
            spec[:, start : start + mask_height, :] = 0.0

        return spec
