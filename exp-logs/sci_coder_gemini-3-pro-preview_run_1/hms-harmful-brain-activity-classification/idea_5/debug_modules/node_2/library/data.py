import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Standard ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipelines for Meso and Macro views.
    """
    if mode == "train":
        # SpecAugment-like masking using CoarseDropout
        # We apply this before normalization
        aug = A.Compose(
            [
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        aug = A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return aug


class HMSDataset(Dataset):
    def __init__(self, metadata, transform=None, mode="train", load_cached_data=False):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing paths and offsets.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load/save pre-processed .npy files.
        """
        self.metadata = metadata
        self.transform = transform
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Pre-compute full paths and values for faster access in __getitem__
        self.eeg_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in metadata["eeg_path"]
        ]
        self.spec_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in metadata["spectrogram_path"]
        ]
        if "eeg_label_offset_seconds" in metadata.columns:
            self.eeg_offsets = metadata["eeg_label_offset_seconds"].values
        else:
            self.eeg_offsets = np.zeros(len(metadata))

        if "spectrogram_label_offset_seconds" in metadata.columns:
            self.spec_offsets = metadata["spectrogram_label_offset_seconds"].values
        else:
            self.spec_offsets = np.zeros(len(metadata))

        # Identifiers for caching
        self.eeg_ids = metadata["eeg_id"].values
        # Use sub_id if available (train/val), else index or dummy (test)
        if "eeg_sub_id" in metadata.columns:
            self.sub_ids = metadata["eeg_sub_id"].values
        else:
            self.sub_ids = np.arange(len(metadata))

        # Targets
        if self.mode != "test":
            self.targets = metadata[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def _load_eeg(self, path, offset):
        """
        Loads EEG parquet, extracts 50s window, downsamples, and normalizes.
        """
        try:
            # Read specific columns to save memory/time
            eeg_df = pd.read_parquet(path, columns=Config.EEG_CHANNELS)

            # Calculate indices for 50s window
            # Raw data is 200Hz
            start_idx = int(offset * Config.EEG_RAW_SAMPLING_RATE)
            end_idx = start_idx + int(
                Config.EEG_DURATION_SECONDS * Config.EEG_RAW_SAMPLING_RATE
            )

            # Slice the dataframe
            eeg_data = eeg_df.iloc[start_idx:end_idx].values

            # Pad if the segment is shorter than expected (edge case)
            expected_len = int(
                Config.EEG_DURATION_SECONDS * Config.EEG_RAW_SAMPLING_RATE
            )
            if eeg_data.shape[0] < expected_len:
                pad_len = expected_len - eeg_data.shape[0]
                eeg_data = np.pad(eeg_data, ((0, pad_len), (0, 0)), mode="constant")

            # Fill NaNs
            eeg_data = np.nan_to_num(eeg_data, nan=0.0)

            # Downsample to 100Hz (::2)
            eeg_data = eeg_data[
                ::2, :
            ]  # Shape becomes (2500, 20) -> (5000, 20) if duration 50s
            # 50s * 200Hz = 10000 samples. ::2 -> 5000 samples.

            # Clip outliers to stabilize training
            eeg_data = np.clip(eeg_data, -1024, 1024)

            # Channel-wise Instance Normalization
            mean = np.mean(eeg_data, axis=0, keepdims=True)
            std = np.std(eeg_data, axis=0, keepdims=True)
            eeg_data = (eeg_data - mean) / (std + 1e-6)

            # Transpose to (Channels, Time) for 1D Conv
            eeg_data = eeg_data.transpose(1, 0)  # (20, 5000)

            return eeg_data.astype(np.float32)

        except Exception as e:
            # Fallback for corrupted files
            return np.zeros(
                (len(Config.EEG_CHANNELS), Config.EEG_SEQ_LEN), dtype=np.float32
            )

    def _load_spectrogram(self, path, offset):
        """
        Loads Spectrogram parquet, extracts 10m window, applies log transform.
        """
        try:
            # Load full spectrogram
            spec_df = pd.read_parquet(path)
            spec_arr = spec_df.values  # (Time, Freq)

            # Calculate 10m window indices (Spectrogram is ~0.5Hz, 2s per row)
            # 10 minutes = 600 seconds = 300 rows
            start_row = int(offset / 2)
            end_row = start_row + 300

            # Handle bounds
            max_rows = spec_arr.shape[0]
            start_row = max(0, start_row)
            end_row = min(max_rows, end_row)

            window = spec_arr[start_row:end_row, :]

            # Pad if too short
            if window.shape[0] < 300:
                pad_rows = 300 - window.shape[0]
                window = np.pad(window, ((0, pad_rows), (0, 0)), mode="constant")
            elif window.shape[0] > 300:
                window = window[:300, :]

            # Log transform (dB scale)
            window = np.log1p(window)

            # Handle NaNs
            window = np.nan_to_num(window, nan=0.0)

            return window.astype(np.float32)

        except Exception as e:
            return np.zeros((300, 400), dtype=np.float32)

    def _process_images(self, spec_window):
        """
        Generates Meso and Macro views from the 10m spectrogram window.
        """
        # spec_window shape: (300, Freqs)

        # --- Macro View (Global Context) ---
        # Resize full 10m window to 512x512
        macro_img = cv2.resize(
            spec_window, Config.MACRO_IMG_SIZE, interpolation=cv2.INTER_LINEAR
        )
        # Stack to 3 channels for pretrained models
        macro_img = np.stack([macro_img] * 3, axis=-1)

        # --- Meso View (Local Event) ---
        # Crop center 50s (approx 25 rows)
        # Center of 300 rows is 150
        center_idx = 150
        # 50 seconds / 2s per row = 25 rows
        half_rows = 12
        meso_crop = spec_window[center_idx - half_rows : center_idx + half_rows + 1, :]

        # Resize to 224x224
        meso_img = cv2.resize(
            meso_crop, Config.MESO_IMG_SIZE, interpolation=cv2.INTER_LINEAR
        )
        meso_img = np.stack([meso_img] * 3, axis=-1)

        return meso_img, macro_img

    def __getitem__(self, idx):
        # Caching Logic
        if self.load_cached_data:
            cache_name = f"{self.eeg_ids[idx]}_{self.sub_ids[idx]}.npy"
            cache_path = os.path.join(Config.CACHE_DIR, cache_name)

            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True).item()
                    micro = data["micro"]
                    meso = data["meso"]
                    macro = data["macro"]
                except:
                    # If load fails, recompute
                    micro = self._load_eeg(self.eeg_paths[idx], self.eeg_offsets[idx])
                    spec_window = self._load_spectrogram(
                        self.spec_paths[idx], self.spec_offsets[idx]
                    )
                    meso, macro = self._process_images(spec_window)
            else:
                # Compute and Save
                micro = self._load_eeg(self.eeg_paths[idx], self.eeg_offsets[idx])
                spec_window = self._load_spectrogram(
                    self.spec_paths[idx], self.spec_offsets[idx]
                )
                meso, macro = self._process_images(spec_window)

                # Save
                data_dict = {"micro": micro, "meso": meso, "macro": macro}
                np.save(cache_path, data_dict)
        else:
            # On-the-fly processing
            micro = self._load_eeg(self.eeg_paths[idx], self.eeg_offsets[idx])
            spec_window = self._load_spectrogram(
                self.spec_paths[idx], self.spec_offsets[idx]
            )
            meso, macro = self._process_images(spec_window)

        # --- Augmentations ---

        # 1. EEG Channel Dropout (Train only)
        if self.mode == "train":
            # Randomly zero out 1-2 channels with 50% probability
            if np.random.rand() < 0.5:
                num_drop = np.random.randint(1, 3)
                channels_idx = np.random.choice(micro.shape[0], num_drop, replace=False)
                micro[channels_idx, :] = 0.0

        # 2. Image Augmentations (Albumentations)
        if self.transform:
            res_meso = self.transform(image=meso)
            meso_tensor = res_meso["image"]

            res_macro = self.transform(image=macro)
            macro_tensor = res_macro["image"]
        else:
            # Fallback to simple tensor conversion
            meso_tensor = torch.tensor(meso.transpose(2, 0, 1), dtype=torch.float32)
            macro_tensor = torch.tensor(macro.transpose(2, 0, 1), dtype=torch.float32)

        micro_tensor = torch.tensor(micro, dtype=torch.float32)

        result = {
            "micro": micro_tensor,
            "meso": meso_tensor,
            "macro": macro_tensor,
        }

        if self.mode != "test":
            result["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return result


def get_loaders(debug=False, load_cached_data=False):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        debug (bool): If True, subsamples the dataset for quick testing.
        load_cached_data (bool): If True, enables caching mechanism in Dataset.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Subsampling
    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        if len(test_df) > Config.DEBUG_SAMPLE_SIZE:
            test_df = test_df.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            ).reset_index(drop=True)

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = HMSDataset(
        train_df,
        transform=train_transform,
        mode="train",
        load_cached_data=load_cached_data,
    )
    val_dataset = HMSDataset(
        val_df, transform=val_transform, mode="val", load_cached_data=load_cached_data
    )
    test_dataset = HMSDataset(
        test_df, transform=val_transform, mode="test", load_cached_data=load_cached_data
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
