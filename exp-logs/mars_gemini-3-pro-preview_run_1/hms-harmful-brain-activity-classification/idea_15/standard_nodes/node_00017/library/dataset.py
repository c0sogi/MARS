import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset
from library.config import Config


class HMSDataset(Dataset):
    def __init__(self, mode="train", transform=None, use_cache=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform: Optional transform (not used directly, internal augs used).
            use_cache (bool): Whether to save/load processed data from disk.
        """
        self.mode = mode
        self.use_cache = use_cache

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            # Global Random Subsampling for Training
            if len(self.df) > Config.TRAIN_SAMPLE_SIZE:
                self.df = self.df.sample(
                    n=Config.TRAIN_SAMPLE_SIZE, random_state=Config.SEED
                ).reset_index(drop=True)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Pre-calculate column groups for spectrograms
        self.regions = ["LL", "RL", "LP", "RP"]

        # Augmentations
        self.spec_transform = None
        if mode == "train":
            # SpecAugment equivalent using CoarseDropout
            # Masking blocks in Time (height) and Freq (width)
            self.spec_transform = A.Compose(
                [
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=32,
                        max_width=32,
                        min_holes=1,
                        min_height=8,
                        min_width=8,
                        fill_value=0,
                        p=0.5,
                    )
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Unique identifier for caching
        # Test set might not have sub_ids, use eeg_id
        eeg_id = str(row["eeg_id"])
        eeg_sub_id = str(int(row.get("eeg_sub_id", 0)))
        spec_sub_id = str(int(row.get("spectrogram_sub_id", 0)))

        cache_name = f"{eeg_id}_{eeg_sub_id}_{spec_sub_id}.npz"
        cache_path = os.path.join(Config.CACHE_DIR, cache_name)

        # 1. Try Loading from Cache
        if self.use_cache and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                X_eeg = data["eeg"]
                X_spec = data["spec"]

                # Apply Augmentation on cached data if training
                if self.mode == "train":
                    X_eeg = self._augment_eeg(X_eeg)
                    X_spec = self._augment_spec(X_spec)

                # Convert to Tensor
                X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
                X_spec = torch.tensor(X_spec, dtype=torch.float32)

                if self.mode != "test":
                    y = row[Config.TARGET_COLS].values.astype(np.float32)
                    return X_eeg, X_spec, torch.tensor(y, dtype=torch.float32)
                else:
                    return X_eeg, X_spec
            except Exception:
                # If load fails, fall back to processing
                pass

        # 2. Process EEG
        X_eeg = self._process_eeg(row)

        # 3. Process Spectrogram
        X_spec = self._process_spec(row)

        # 4. Save to Cache (before augmentation to keep cache pure)
        if self.use_cache:
            try:
                np.savez_compressed(cache_path, eeg=X_eeg, spec=X_spec)
            except Exception as e:
                # Disk full or write error, ignore
                pass

        # 5. Apply Augmentations (On-the-fly)
        if self.mode == "train":
            X_eeg = self._augment_eeg(X_eeg)
            X_spec = self._augment_spec(X_spec)

        # 6. Return Tensors
        X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
        X_spec = torch.tensor(X_spec, dtype=torch.float32)

        if self.mode != "test":
            y = row[Config.TARGET_COLS].values.astype(np.float32)
            return X_eeg, X_spec, torch.tensor(y, dtype=torch.float32)
        else:
            return X_eeg, X_spec

    def _process_eeg(self, row):
        """Reads, slices, downsamples, and normalizes EEG."""
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])

        try:
            # Load parquet
            eeg_df = pd.read_parquet(eeg_path)

            # Select relevant window
            # Determine offset: Test set defaults to 0, Train/Val uses metadata
            if self.mode == "test":
                offset = 0
            else:
                offset = row["eeg_label_offset_seconds"]

            start_idx = int(offset * Config.EEG_RAW_SAMPLING_RATE)
            end_idx = start_idx + int(
                Config.EEG_DURATION * Config.EEG_RAW_SAMPLING_RATE
            )

            # Handle edge cases where offset might be slightly off
            if start_idx < 0:
                start_idx = 0

            data = eeg_df.iloc[start_idx:end_idx].values

            # Pad if shorter than expected (rare)
            expected_len = int(Config.EEG_DURATION * Config.EEG_RAW_SAMPLING_RATE)
            if len(data) < expected_len:
                pad_len = expected_len - len(data)
                data = np.pad(data, ((0, pad_len), (0, 0)), mode="constant")
            elif len(data) > expected_len:
                data = data[:expected_len]

            # Handle NaNs (common in EEG)
            data = np.nan_to_num(data, nan=0.0)

            # Downsample: 200Hz -> 100Hz (Stride 2)
            # Shape: (Time, Channels) -> (Time/2, Channels)
            data = data[::2, :]

            # Instance Normalization (Channel-wise)
            # (T, C)
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            data = (data - mean) / (std + 1e-6)

            # Transpose to (Channels, Time) for 1D Conv
            data = data.transpose(1, 0)

            # Ensure strictly 20 channels (19 EEG + 1 EKG)
            # If columns mismatch, we might need column selection, but
            # competition data is consistent.

            return data.astype(np.float32)

        except Exception as e:
            # Fallback for corrupt files
            # Return zeros of correct shape (20, 5000)
            return np.zeros((Config.EEG_CHANNELS, Config.EEG_SEQ_LEN), dtype=np.float32)

    def _process_spec(self, row):
        """Reads, slices, resizes, and adds coordinate map to Spectrogram."""
        spec_path = os.path.join(Config.INPUT_DIR, row["spectrogram_path"])

        try:
            spec_df = pd.read_parquet(spec_path)

            # Select relevant window
            if self.mode == "test":
                # Test files are exactly 10 mins
                data_df = spec_df
            else:
                offset = row["spectrogram_label_offset_seconds"]
                # Spectrograms have a 'time' column
                if "time" in spec_df.columns:
                    # 10 minutes = 600 seconds
                    mask = (spec_df["time"] >= offset) & (
                        spec_df["time"] < offset + 600
                    )
                    data_df = spec_df[mask]
                    # If empty (rare alignment issue), take closest
                    if len(data_df) == 0:
                        data_df = spec_df  # Fallback
                else:
                    # Fallback if no time column (should not happen in this dataset)
                    data_df = spec_df

            # Drop time column to get just frequencies
            if "time" in data_df.columns:
                data_df = data_df.drop(columns=["time"])

            # Identify region columns
            # Columns are like 'LL_0.59', 'RL_0.59', etc.
            cols = data_df.columns
            ll_cols = [c for c in cols if c.startswith("LL")]
            rl_cols = [c for c in cols if c.startswith("RL")]
            lp_cols = [c for c in cols if c.startswith("LP")]
            rp_cols = [c for c in cols if c.startswith("RP")]

            # Extract regions
            # Fill NaNs with 0 (or log(small))
            # Log transform: log(x + 1)
            regions_data = []
            for region_cols in [ll_cols, rl_cols, lp_cols, rp_cols]:
                if not region_cols:
                    # Missing region? Create zeros
                    r_img = np.zeros((Config.SPEC_RESIZE_SIZE), dtype=np.float32)
                else:
                    r_vals = data_df[region_cols].values
                    r_vals = np.nan_to_num(r_vals, nan=0.0)
                    r_vals = np.log1p(r_vals)

                    # Resize to (512, 512)
                    # Input shape: (Time, Freq)
                    # Resize expects (Width, Height) -> (Freq, Time)
                    # We want output (512, 512).
                    r_img = cv2.resize(
                        r_vals, Config.SPEC_RESIZE_SIZE, interpolation=cv2.INTER_AREA
                    )

                regions_data.append(r_img)

            # Stack regions: (512, 512, 4)
            spec_img = np.stack(regions_data, axis=-1)

            # Normalize to [0, 1] or standardized range?
            # EfficientNet likes standardized, but raw log spectrograms are usually fine.
            # Let's do Min-Max per sample to keep it in reasonable range [0, 1]
            eps = 1e-6
            img_min = spec_img.min()
            img_max = spec_img.max()
            spec_img = (spec_img - img_min) / (img_max - img_min + eps)

            # Coordinate Injection (Channel 5)
            # Linear gradient along the Time axis (Height usually in this resizing)
            # We resized to (512, 512). Let's assume axis 0 is Time, axis 1 is Freq.
            # Gradient from -1 to 1 along axis 0.
            H, W, _ = spec_img.shape
            # Create (H, W) array
            y_coords = np.linspace(-1, 1, H)
            coord_map = np.tile(y_coords[:, None], (1, W))  # (H, W)
            coord_map = coord_map[..., np.newaxis]  # (H, W, 1)

            # Concatenate: (512, 512, 5)
            spec_final = np.concatenate([spec_img, coord_map], axis=-1)

            # Transpose for PyTorch: (C, H, W) -> (5, 512, 512)
            spec_final = spec_final.transpose(2, 0, 1)

            return spec_final.astype(np.float32)

        except Exception as e:
            # Fallback
            return np.zeros(
                (Config.SPEC_CHANNELS, *Config.SPEC_RESIZE_SIZE), dtype=np.float32
            )

    def _augment_eeg(self, eeg_data):
        """
        Randomly zero out channels.
        Input: (C, T)
        """
        if np.random.rand() < 0.5:
            # Channel Dropout
            c, t = eeg_data.shape
            # Drop 1 to 3 channels
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(c, num_drop, replace=False)
            eeg_data[drop_indices, :] = 0.0
        return eeg_data

    def _augment_spec(self, spec_data):
        """
        Apply Albumentations to the first 4 channels (Spectrograms).
        Keep 5th channel (Coordinate) intact.
        Input: (C, H, W)
        """
        # Transpose to (H, W, C) for Albumentations
        img = spec_data.transpose(1, 2, 0)

        # Split Spec and Coord
        spec_content = img[..., :4]
        coord_content = img[..., 4:]

        # Apply transform
        if self.spec_transform:
            augmented = self.spec_transform(image=spec_content)["image"]
        else:
            augmented = spec_content

        # Recombine
        final_img = np.concatenate([augmented, coord_content], axis=-1)

        # Back to (C, H, W)
        return final_img.transpose(2, 0, 1)
