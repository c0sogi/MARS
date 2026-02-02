import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from typing import Optional, Tuple, Dict

from library.config import Config
from library.utils import seed_everything


class SiameseDualDataset(Dataset):
    """
    Dataset class for the Symmetry-Aware Siamese Dual-Stream Network.
    Handles loading and preprocessing of paired EEG (Left/Right) and Spectrogram data.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        mode: str = "train",
        augment: bool = False,
        load_cached_data: bool = False,
    ):
        self.metadata = metadata
        self.mode = mode
        self.augment = augment
        self.load_cached_data = load_cached_data

        # Channel definitions
        self.left_channels = Config.LEFT_HEMISPHERE_CHANNELS
        self.right_channels = Config.RIGHT_HEMISPHERE_CHANNELS

        # Initialize transforms
        self.spec_transform = (
            self._get_spec_transforms() if augment else self._get_valid_transforms()
        )

        # Ensure cache directory exists if caching is enabled
        if self.load_cached_data:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.metadata)

    def _get_spec_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    height=Config.SPEC_RESIZE_SIZE[0], width=Config.SPEC_RESIZE_SIZE[1]
                ),
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
                ToTensorV2(),
            ]
        )

    def _get_valid_transforms(self):
        return A.Compose(
            [
                A.Resize(
                    height=Config.SPEC_RESIZE_SIZE[0], width=Config.SPEC_RESIZE_SIZE[1]
                ),
                ToTensorV2(),
            ]
        )

    def _process_eeg(self, row) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads EEG parquet, extracts 50s window, splits into Left/Right groups,
        downsamples to 100Hz, and applies instance normalization.
        """
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])

        # 1. Load Data
        try:
            # Read full parquet (efficient enough for these file sizes)
            eeg_df = pd.read_parquet(eeg_path)
        except Exception:
            # Fallback for corrupt/missing files
            shape = (len(self.left_channels), Config.EEG_SEQ_LENGTH)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        # 2. Time Slicing
        # Calculate start/end indices based on offset
        if self.mode == "test":
            start_idx = 0
        else:
            offset_sec = row["eeg_label_offset_seconds"]
            start_idx = int(offset_sec * Config.EEG_SRC_SAMPLING_RATE)

        # Target duration in source samples
        src_duration_samples = int(
            Config.EEG_DURATION_SEC * Config.EEG_SRC_SAMPLING_RATE
        )
        end_idx = start_idx + src_duration_samples

        # Handle boundary conditions
        file_len = len(eeg_df)
        if start_idx >= file_len:
            # Offset beyond file length (rare error case)
            chunk = pd.DataFrame(columns=eeg_df.columns)
        else:
            chunk = eeg_df.iloc[start_idx:end_idx]

        # 3. Channel Selection & Grouping
        def extract_group(channels):
            group_data = []
            for ch in channels:
                if ch in chunk.columns:
                    # Fill NaNs in raw data with 0 before processing
                    vals = chunk[ch].fillna(0.0).values
                else:
                    vals = np.zeros(len(chunk))
                group_data.append(vals)
            return np.array(group_data)  # Shape (C, T)

        left_data = extract_group(self.left_channels)
        right_data = extract_group(self.right_channels)

        # 4. Padding (if chunk is shorter than 50s)
        def pad_signal(data):
            C, T = data.shape
            if T < src_duration_samples:
                pad_width = src_duration_samples - T
                data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
            elif T > src_duration_samples:
                data = data[:, :src_duration_samples]
            return data

        left_data = pad_signal(left_data)
        right_data = pad_signal(right_data)

        # 5. Downsampling (200Hz -> 100Hz)
        # Using simple slicing for integer factor 2 is efficient and sufficient
        if (
            Config.EEG_SRC_SAMPLING_RATE == 200
            and Config.EEG_TARGET_SAMPLING_RATE == 100
        ):
            left_data = left_data[:, ::2]
            right_data = right_data[:, ::2]
        else:
            # Generic resampling
            left_data = resample(left_data, Config.EEG_SEQ_LENGTH, axis=1)
            right_data = resample(right_data, Config.EEG_SEQ_LENGTH, axis=1)

        # 6. Instance Normalization
        # (x - mean) / (std + eps) per channel
        def normalize(data):
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            return (data - mean) / (std + 1e-6)

        left_proc = normalize(left_data).astype(np.float32)
        right_proc = normalize(right_data).astype(np.float32)

        return left_proc, right_proc

    def _process_spec(self, row) -> np.ndarray:
        """
        Loads Spectrogram parquet, extracts 10m window, log-transforms,
        and prepares it as a 3-channel image.
        """
        spec_path = os.path.join(Config.INPUT_DIR, row["spectrogram_path"])

        try:
            spec_df = pd.read_parquet(spec_path)
        except Exception:
            return np.zeros((*Config.SPEC_RESIZE_SIZE, 3), dtype=np.float32)

        # 1. Time Slicing
        # HMS spectrograms typically have time as rows.
        # We need the 10-minute window (600 seconds).
        if self.mode == "test":
            spec_chunk = spec_df
        else:
            offset = row["spectrogram_label_offset_seconds"]
            if "time" in spec_df.columns:
                mask = (spec_df["time"] >= offset) & (spec_df["time"] < offset + 600)
                spec_chunk = spec_df[mask]
            else:
                # Fallback: Assume 0.5Hz resolution (2s per row) common in this dataset
                start_row = int(offset / 2)
                end_row = start_row + 300  # 300 rows * 2s = 600s
                spec_chunk = spec_df.iloc[start_row:end_row]

        # 2. Preprocessing
        # Drop time column if exists
        cols = [c for c in spec_chunk.columns if c != "time"]
        data = spec_chunk[cols].values  # Shape (T, F)

        # Fill NaNs
        data = np.nan_to_num(data, nan=0.0)

        # Log Transform (dB scale)
        data = np.log1p(data)

        # Min-Max Normalize to [0, 1] for image representation
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val > 1e-6:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data)

        # 3. Format as Image (H, W, 3)
        # Treat Time as Height, Freq as Width
        # Replicate to 3 channels for ImageNet-pretrained backbones
        img = np.stack([data, data, data], axis=-1)

        return img.astype(np.float32)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Determine Cache Key
        if "label_id" in row:
            cache_key = str(int(row["label_id"]))
        else:
            cache_key = f"{row['eeg_id']}_{idx}"

        cache_file = os.path.join(Config.CACHE_DIR, f"{cache_key}.npz")

        # --- Caching Logic ---
        loaded_from_cache = False
        data_dict = {}

        if self.load_cached_data and os.path.exists(cache_file):
            try:
                # Load compressed numpy archive
                cached = np.load(cache_file)
                data_dict = {
                    "left_eeg": cached["left_eeg"],
                    "right_eeg": cached["right_eeg"],
                    "spec_img": cached["spec_img"],
                    "label": cached["label"],
                }
                loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if not loaded_from_cache:
            # Compute from scratch
            left_eeg, right_eeg = self._process_eeg(row)
            spec_img = self._process_spec(row)

            if self.mode != "test":
                label = row[Config.TARGET_COLS].values.astype(np.float32)
            else:
                label = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

            data_dict = {
                "left_eeg": left_eeg,
                "right_eeg": right_eeg,
                "spec_img": spec_img,
                "label": label,
            }

            # Save to cache
            if self.load_cached_data:
                np.savez_compressed(cache_file, **data_dict)

        # --- Final Data Preparation ---
        left_tensor = torch.tensor(data_dict["left_eeg"], dtype=torch.float32)
        right_tensor = torch.tensor(data_dict["right_eeg"], dtype=torch.float32)
        label_tensor = torch.tensor(data_dict["label"], dtype=torch.float32)

        # Apply Albumentations to Spectrogram
        # Albumentations expects numpy image (H, W, C)
        spec_img = data_dict["spec_img"]
        if self.spec_transform:
            augmented = self.spec_transform(image=spec_img)
            spec_tensor = augmented["image"]  # (C, H, W)
        else:
            # Fallback manual conversion
            spec_tensor = torch.tensor(spec_img).permute(2, 0, 1)

        return {
            "left_eeg": left_tensor,
            "right_eeg": right_tensor,
            "spectrogram": spec_tensor,
            "label": label_tensor,
            "eeg_id": row["eeg_id"] if "eeg_id" in row else 0,
        }


def get_dataloaders(
    train_csv: str = Config.TRAIN_CSV,
    val_csv: str = Config.VAL_CSV,
    test_csv: str = Config.TEST_CSV,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    load_cached_data: bool = False,
    debug_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """

    # Load Metadata
    # Handle missing files gracefully for local testing environments
    try:
        train_df = pd.read_csv(train_csv)
    except FileNotFoundError:
        train_df = pd.DataFrame()

    try:
        val_df = pd.read_csv(val_csv)
    except FileNotFoundError:
        val_df = pd.DataFrame()

    try:
        test_df = pd.read_csv(test_csv)
    except FileNotFoundError:
        test_df = pd.DataFrame()

    # Apply Debug Sizing
    if debug_size is not None:
        if not train_df.empty:
            train_df = train_df.iloc[:debug_size]
        if not val_df.empty:
            val_df = val_df.iloc[:debug_size]
        if not test_df.empty:
            test_df = test_df.iloc[:debug_size]

    # Instantiate Datasets
    # Train: Augmentation ON
    train_ds = SiameseDualDataset(
        train_df, mode="train", augment=True, load_cached_data=load_cached_data
    )
    # Val: Augmentation OFF
    val_ds = SiameseDualDataset(
        val_df, mode="val", augment=False, load_cached_data=load_cached_data
    )
    # Test: Augmentation OFF
    test_ds = SiameseDualDataset(
        test_df, mode="test", augment=False, load_cached_data=load_cached_data
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if len(train_ds) > batch_size else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
