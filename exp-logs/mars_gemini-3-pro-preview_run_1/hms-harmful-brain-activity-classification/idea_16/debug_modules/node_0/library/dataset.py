import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Define EEG Channels in order as per 10-20 system + EKG
EEG_CHANNELS = [
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


class EEGSpecDataset(Dataset):
    """
    Dataset class for the Pyramid-Resolution Coordinate-Guided Fusion Network.
    Handles loading of Raw EEG and Spectrograms, applying Coordinate Injection,
    and performing augmentations.
    """

    def __init__(self, metadata_df, mode="train", load_cached_data=True, augment=False):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing file paths and labels.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
            augment (bool): Whether to apply data augmentation.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.augment = augment

        # Global Random Subsampling for Training
        if self.mode == "train" and Config.TRAIN_SUBSAMPLE_SIZE is not None:
            if len(metadata_df) > Config.TRAIN_SUBSAMPLE_SIZE:
                # Deterministic shuffle based on Config.SEED
                metadata_df = metadata_df.sample(
                    n=Config.TRAIN_SUBSAMPLE_SIZE, random_state=Config.SEED
                ).reset_index(drop=True)

        self.metadata = metadata_df.reset_index(drop=True)

        # Setup Cache Directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Spectrogram Augmentations (Albumentations)
        # Applied to the stacked spectrogram image
        self.spec_transform = A.Compose(
            [
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.SPEC_SIZE[0] // 8,
                    max_width=Config.SPEC_SIZE[1] // 8,
                    fill_value=0,
                    p=0.5,
                ),
            ]
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Generate Cache Key
        eeg_id = row["eeg_id"]
        spec_id = row["spectrogram_id"]

        # Get offsets, defaulting to 0 for test set or missing values
        eeg_offset = row.get("eeg_label_offset_seconds", 0)
        spec_offset = row.get("spectrogram_label_offset_seconds", 0)

        # Ensure offsets are integers for filename safety
        if pd.isna(eeg_offset):
            eeg_offset = 0
        if pd.isna(spec_offset):
            spec_offset = 0

        cache_key = f"{eeg_id}_{int(eeg_offset)}_{spec_id}_{int(spec_offset)}.npy"
        cache_path = os.path.join(self.cache_dir, cache_key)

        data_loaded = False
        processed_data = None

        # 1. Try Load Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                processed_data = np.load(cache_path, allow_pickle=True).item()
                data_loaded = True
            except Exception:
                data_loaded = False

        # 2. Process if not loaded
        if not data_loaded:
            processed_data = self._process_item(row)
            # Save to cache
            np.save(cache_path, processed_data)

        # Unpack
        eeg_tensor = processed_data["eeg"]  # (20, 5000)
        spec_tensor = processed_data["spec"]  # (5, 512, 512)

        # 3. Augmentations (Train only)
        if self.mode == "train" and self.augment:
            eeg_tensor = self._augment_eeg(eeg_tensor)
            spec_tensor = self._augment_spec(spec_tensor)

        # 4. Prepare Outputs
        eeg_tensor = torch.tensor(eeg_tensor, dtype=torch.float32)
        spec_tensor = torch.tensor(spec_tensor, dtype=torch.float32)

        if self.mode != "test":
            targets = row[Config.PROB_COLS].values.astype(np.float32)
            # Ensure sum to 1 (numerical stability)
            targets = targets / (targets.sum() + 1e-6)
            return eeg_tensor, spec_tensor, torch.tensor(targets, dtype=torch.float32)
        else:
            return eeg_tensor, spec_tensor

    def _process_item(self, row):
        """
        Reads raw files and processes them into tensors.
        Returns a dictionary: {'eeg': np.array, 'spec': np.array}
        """
        # --- EEG Processing ---
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])
        eeg_offset = row.get("eeg_label_offset_seconds", 0)
        if pd.isna(eeg_offset):
            eeg_offset = 0

        # Load Parquet
        try:
            eeg_df = pd.read_parquet(eeg_path, columns=EEG_CHANNELS)
        except:
            # Fallback for missing columns or full read
            eeg_df = pd.read_parquet(eeg_path)
            # Ensure all channels exist
            for c in EEG_CHANNELS:
                if c not in eeg_df.columns:
                    eeg_df[c] = 0.0
            eeg_df = eeg_df[EEG_CHANNELS]

        # Slice 50 seconds
        fs = Config.EEG_RAW_SAMPLE_RATE  # 200
        start_idx = int(eeg_offset * fs)
        end_idx = start_idx + int(Config.EEG_DURATION_SEC * fs)

        eeg_vals = eeg_df.values

        if self.mode == "test":
            # Test files are exactly 50s
            eeg_segment = eeg_vals
        else:
            # Pad if out of bounds
            if start_idx < 0:
                start_idx = 0

            if end_idx > len(eeg_vals):
                eeg_segment = eeg_vals[start_idx:]
                pad_len = int(Config.EEG_DURATION_SEC * fs) - len(eeg_segment)
                if pad_len > 0:
                    eeg_segment = np.pad(
                        eeg_segment, ((0, pad_len), (0, 0)), "constant"
                    )
            else:
                eeg_segment = eeg_vals[start_idx:end_idx]

        # Downsample 200Hz -> 100Hz
        eeg_segment = eeg_segment[::2, :]  # (5000, 20)

        # Transpose to (Channels, Time) -> (20, 5000)
        eeg_segment = eeg_segment.T

        # Handle NaNs
        eeg_segment = np.nan_to_num(eeg_segment, nan=0.0)

        # Normalize (Instance Norm per channel)
        means = np.mean(eeg_segment, axis=1, keepdims=True)
        stds = np.std(eeg_segment, axis=1, keepdims=True)
        eeg_segment = (eeg_segment - means) / (stds + 1e-6)

        # --- Spectrogram Processing ---
        spec_path = os.path.join(Config.INPUT_DIR, row["spectrogram_path"])
        spec_offset = row.get("spectrogram_label_offset_seconds", 0)
        if pd.isna(spec_offset):
            spec_offset = 0

        spec_df = pd.read_parquet(spec_path)

        # Filter by time if exists
        if "time" in spec_df.columns:
            mask = (spec_df["time"] >= spec_offset) & (
                spec_df["time"] < spec_offset + Config.SPEC_DURATION_SEC
            )
            spec_chunk = spec_df[mask].drop(columns=["time"])
            if len(spec_chunk) == 0:
                spec_chunk = spec_df.drop(columns=["time"]).iloc[:300]  # Fallback
        else:
            spec_chunk = spec_df

        # Identify Region Columns and Stack
        regions = ["LL", "RL", "LP", "RP"]
        region_maps = []
        all_cols = spec_chunk.columns.tolist()

        for region in regions:
            # Find columns starting with region + "_"
            r_cols = [c for c in all_cols if c.startswith(f"{region}_")]
            # Sort by frequency value
            try:
                r_cols.sort(key=lambda x: float(x.split("_")[1]))
            except:
                pass

            if not r_cols:
                r_data = np.zeros((len(spec_chunk), 100))
            else:
                r_data = spec_chunk[r_cols].values  # (Time, Freq)

            r_data = np.nan_to_num(r_data, nan=0.0)
            r_data = np.log1p(r_data)

            # Resize to (512, 512)
            # Input is (Time, Freq). We want output (Freq, Time) = (512, 512).
            # Transpose first to (Freq, Time)
            r_data_T = r_data.T
            # Resize
            r_resized = cv2.resize(
                r_data_T, Config.SPEC_SIZE, interpolation=cv2.INTER_LINEAR
            )
            region_maps.append(r_resized)

        # Stack 4 regions -> (4, 512, 512)
        spec_img = np.stack(region_maps, axis=0)

        # --- Coordinate Map ---
        # 5th Channel: Relative Time [-1, 1]
        # We want the coordinate to vary along the Time axis.
        # Our tensor is (Freq, Time). Time is the second dimension (axis 2).
        time_steps = Config.SPEC_SIZE[1]  # 512 (Time axis)
        coord_line = np.linspace(-1, 1, time_steps).astype(np.float32)
        # Tile across Frequency axis (Height)
        coord_map = np.tile(coord_line, (Config.SPEC_SIZE[0], 1))  # (512, 512)
        coord_map = coord_map[np.newaxis, :, :]  # (1, 512, 512)

        # Concatenate -> (5, 512, 512)
        full_spec = np.concatenate([spec_img, coord_map], axis=0)

        # Normalize Spectrogram Channels (0-3)
        s_mean = np.mean(full_spec[:4], axis=(1, 2), keepdims=True)
        s_std = np.std(full_spec[:4], axis=(1, 2), keepdims=True)
        full_spec[:4] = (full_spec[:4] - s_mean) / (s_std + 1e-6)

        return {
            "eeg": eeg_segment.astype(np.float32),
            "spec": full_spec.astype(np.float32),
        }

    def _augment_eeg(self, eeg):
        # Channel Dropout
        if np.random.rand() < 0.5:
            num_drop = np.random.randint(1, 4)
            channels = np.random.choice(eeg.shape[0], num_drop, replace=False)
            eeg[channels, :] = 0.0
        return eeg

    def _augment_spec(self, spec):
        # Transpose to (H, W, C) for Albumentations
        spec_T = spec.transpose(1, 2, 0)
        res = self.spec_transform(image=spec_T)["image"]
        return res.transpose(2, 0, 1)


def get_train_dataloader(metadata_df):
    ds = EEGSpecDataset(metadata_df, mode="train", load_cached_data=True, augment=True)
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )


def get_val_dataloader(metadata_df):
    ds = EEGSpecDataset(metadata_df, mode="val", load_cached_data=True, augment=False)
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_test_dataloader(metadata_df):
    ds = EEGSpecDataset(metadata_df, mode="test", load_cached_data=True, augment=False)
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
