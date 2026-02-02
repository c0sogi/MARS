import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from library.config import Config
from library.utils import seed_everything


class EEGDataset(Dataset):
    """
    Dual-Stream Dataset for EEG and Spectrogram data.
    Implements Coordinate Injection for Spectrograms and raw waveform loading for EEG.
    """

    def __init__(self, metadata, mode="train", config=Config):
        self.df = metadata
        self.mode = mode
        self.config = config

        # Caching setup
        self.cache_dir = os.path.join(config.OUTPUT_DIR, "cache")
        if self.config.USE_CACHE:
            os.makedirs(self.cache_dir, exist_ok=True)

        # EEG Channel names (Fixed order for consistency)
        self.eeg_cols = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        # Spectrogram Region Prefixes
        self.spec_regions = ["LL", "RL", "LP", "RP"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Determine unique ID for caching
        # Train/Val have label_id, Test has eeg_id
        if "label_id" in row:
            uid = int(row["label_id"])
        else:
            uid = int(row["eeg_id"])

        cache_path = os.path.join(self.cache_dir, f"{uid}.npy")

        # 1. Load Data (Cache vs Compute)
        data_loaded = False
        if self.config.USE_CACHE and os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path, allow_pickle=True).item()
                eeg = cached_data["eeg"]
                spec = cached_data["spec"]
                data_loaded = True
            except Exception as e:
                # Corrupt cache, recompute
                pass

        if not data_loaded:
            eeg, spec = self.process_raw(row)
            if self.config.USE_CACHE:
                try:
                    np.save(cache_path, {"eeg": eeg, "spec": spec})
                except Exception:
                    pass  # Ignore save errors to avoid crashing

        # 2. Augmentations (Train only)
        if self.mode == "train":
            eeg = self.augment_eeg(eeg)
            spec = self.augment_spec(spec)

        # 3. Convert to Tensors
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        spec_tensor = torch.tensor(spec, dtype=torch.float32)

        # 4. Get Targets
        if self.mode != "test":
            # Extract probability columns
            target_cols = [c for c in row.index if c.endswith("_prob")]
            # Fallback for raw votes if probs not present (though metadata script ensures probs)
            if not target_cols:
                target_cols = [c for c in row.index if c.endswith("_vote")]

            target = row[target_cols].values.astype(np.float32)
            # Ensure sum to 1 (handling potential float errors)
            target = target / (target.sum() + 1e-6)
            return eeg_tensor, spec_tensor, torch.tensor(target, dtype=torch.float32)
        else:
            # Dummy target for test
            return eeg_tensor, spec_tensor, torch.zeros(6, dtype=torch.float32)

    def process_raw(self, row):
        """
        Reads raw parquet files and performs deterministic preprocessing.
        """
        # --- Process Spectrogram ---
        spec_path = os.path.join(self.config.INPUT_DIR, row["spectrogram_path"])

        # Handle offset: default to 0 if missing (Test set)
        spec_offset = row.get("spectrogram_label_offset_seconds", 0)
        if np.isnan(spec_offset):
            spec_offset = 0

        try:
            spec_df = pd.read_parquet(spec_path)

            # Locate the 10-minute window
            # The offset points to the start of the subsample in the consolidated file.
            # If 'time' column exists, use it. Otherwise assume index corresponds to time.
            if "time" in spec_df.columns:
                # Find index closest to offset
                start_idx = (spec_df["time"] - spec_offset).abs().idxmin()
            else:
                # Fallback: assume 0.5s per row (2Hz) -> offset * 2
                start_idx = int(spec_offset * 2)

            # 10 minutes = 600 seconds.
            # If 2Hz (standard), that's 1200 rows. If 0.5Hz, 300 rows.
            # We take a fixed number of rows to cover 600s or until end.
            # EDA suggests ~300-900 rows. We'll take a slice based on time if possible.
            if "time" in spec_df.columns:
                # Filter rows within [offset, offset + 600]
                mask = (spec_df["time"] >= spec_offset) & (
                    spec_df["time"] < spec_offset + 600
                )
                sub_df = spec_df[mask]
                if len(sub_df) == 0:  # Fallback if empty
                    sub_df = spec_df.iloc[start_idx : start_idx + 300]
            else:
                # Fallback slice
                sub_df = spec_df.iloc[start_idx : start_idx + 300]

            # Extract and Aggregate Channels
            # Columns are like "LL_0.59", "RP_10.2"
            # We group by region (LL, RL, LP, RP)
            img_layers = []
            for region in self.spec_regions:
                cols = [c for c in sub_df.columns if c.startswith(region)]
                if not cols:
                    # Create dummy zeros if region missing
                    region_img = np.zeros((self.config.IMG_SIZE), dtype=np.float32)
                else:
                    # (Time, Freq)
                    raw_region = sub_df[cols].values
                    # Log transform
                    raw_region = np.log1p(np.clip(raw_region, 0, None))
                    # Handle NaNs
                    raw_region = np.nan_to_num(raw_region, nan=0.0)
                    # Resize to fixed (H, W) = (512, 512)
                    # cv2.resize expects (Width, Height) -> (512, 512)
                    region_img = cv2.resize(
                        raw_region, self.config.IMG_SIZE, interpolation=cv2.INTER_LINEAR
                    )

                img_layers.append(region_img)

            # Stack -> (4, 512, 512)
            # Note: cv2 returns (H, W). We want (C, H, W).
            spec_img = np.stack(img_layers, axis=0)

        except Exception as e:
            # Fallback for read errors
            spec_img = np.zeros(
                (4, self.config.IMG_SIZE[0], self.config.IMG_SIZE[1]), dtype=np.float32
            )

        # --- Coordinate Injection ---
        # Create the 5th channel: Relative Time Map
        # The event is centered in the 10-minute window (300s mark).
        # We create a gradient from -1 to 1 along the Time axis (Height).
        H, W = self.config.IMG_SIZE
        # Linspace along height: -1 (start) to 1 (end)
        # Assuming H is Time.
        coord_vec = np.linspace(
            self.config.COORD_RANGE[0], self.config.COORD_RANGE[1], H, dtype=np.float32
        )
        # Broadcast to (H, W)
        coord_map = np.tile(coord_vec[:, None], (1, W))
        # Expand to (1, H, W)
        coord_map = coord_map[None, :, :]

        # Concatenate: (4, H, W) + (1, H, W) -> (5, H, W)
        spec_final = np.concatenate([spec_img, coord_map], axis=0)

        # --- Process EEG ---
        eeg_path = os.path.join(self.config.INPUT_DIR, row["eeg_path"])
        eeg_offset = row.get("eeg_label_offset_seconds", 0)
        if np.isnan(eeg_offset):
            eeg_offset = 0

        try:
            eeg_df = pd.read_parquet(eeg_path, columns=self.eeg_cols)

            # Calculate indices
            # 200 Hz sampling rate
            start_sample = int(eeg_offset * 200)
            end_sample = start_sample + 10000  # 50 seconds * 200 Hz

            # Slice
            eeg_data = eeg_df.iloc[start_sample:end_sample].values

            # Handle edge cases (padding if too short)
            if eeg_data.shape[0] < 10000:
                pad_len = 10000 - eeg_data.shape[0]
                eeg_data = np.pad(eeg_data, ((0, pad_len), (0, 0)), mode="constant")

            # Handle NaNs
            eeg_data = np.nan_to_num(eeg_data, nan=0.0)

            # Downsample to 100Hz (Target)
            # Simple decimation by 2
            eeg_data = eeg_data[::2, :]  # (5000, 20)

            # Clip outliers (simple artifact removal)
            eeg_data = np.clip(eeg_data, -1024, 1024)

            # Instance Normalization per channel
            # (T, C) -> normalize columns
            mean = eeg_data.mean(axis=0, keepdims=True)
            std = eeg_data.std(axis=0, keepdims=True)
            eeg_data = (eeg_data - mean) / (std + 1e-6)

            # Transpose to (C, T) -> (20, 5000)
            eeg_final = eeg_data.T

        except Exception as e:
            # Fallback
            eeg_final = np.zeros((20, 5000), dtype=np.float32)

        return eeg_final.astype(np.float32), spec_final.astype(np.float32)

    def augment_spec(self, spec):
        """
        Applies SpecAugment-like masking to the spectrogram channels.
        Does NOT mask the coordinate channel (last channel).
        """
        # spec shape: (5, H, W)
        C, H, W = spec.shape

        # Apply only to content channels (0-3)
        content = spec[:4]

        # Frequency Masking (Horizontal strips)
        if np.random.rand() < 0.5:
            width = np.random.randint(10, W // 4)
            x0 = np.random.randint(0, W - width)
            content[:, :, x0 : x0 + width] = 0

        # Time Masking (Vertical strips)
        if np.random.rand() < 0.5:
            height = np.random.randint(10, H // 4)
            y0 = np.random.randint(0, H - height)
            content[:, y0 : y0 + height, :] = 0

        spec[:4] = content
        return spec

    def augment_eeg(self, eeg):
        """
        Applies Channel Dropout to EEG.
        """
        # eeg shape: (20, 5000)
        if np.random.rand() < 0.5:
            # Drop 1 to 3 channels
            num_drop = np.random.randint(1, 4)
            channels_to_drop = np.random.choice(eeg.shape[0], num_drop, replace=False)
            eeg[channels_to_drop, :] = 0
        return eeg


def get_dataloaders(train_csv_path, val_csv_path, test_csv_path, config=Config):
    """
    Factory function to create dataloaders.
    """
    # Load Metadata
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Debugging: Subsample if requested
    if config.DEBUG_SAMPLE_SIZE > 0:
        train_df = train_df.iloc[: config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: config.DEBUG_SAMPLE_SIZE]

    # Create Datasets
    train_dataset = EEGDataset(train_df, mode="train", config=config)
    val_dataset = EEGDataset(val_df, mode="val", config=config)
    test_dataset = EEGDataset(test_df, mode="test", config=config)

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
