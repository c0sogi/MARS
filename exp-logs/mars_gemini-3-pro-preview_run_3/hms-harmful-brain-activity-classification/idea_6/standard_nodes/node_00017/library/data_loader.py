import os
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Try importing joblib for parallel processing
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def preprocess_eeg(
    eeg_path,
    offset_seconds,
    duration=Config.EEG_DURATION,
    target_sr=Config.EEG_TARGET_SR,
):
    """
    Loads, slices, downsamples, and normalizes raw EEG data.
    """
    try:
        # Load parquet
        df = pd.read_parquet(eeg_path)

        # Handle missing channels by padding with zeros
        data = np.zeros((len(df), Config.N_EEG_CHANNELS), dtype=np.float32)
        for i, col in enumerate(Config.EEG_CHANNELS):
            if col in df.columns:
                data[:, i] = df[col].values

        # Calculate indices for slicing
        sr = Config.EEG_RAW_SR
        start_idx = int(offset_seconds * sr)
        end_idx = start_idx + (duration * sr)

        # Boundary checks
        if start_idx < 0:
            start_idx = 0

        # Slice data
        data = data[start_idx:end_idx, :]

        # Pad or clip if length doesn't match exactly
        target_len_raw = duration * sr
        if len(data) < target_len_raw:
            pad_len = target_len_raw - len(data)
            data = np.pad(data, ((0, pad_len), (0, 0)), "constant")
        elif len(data) > target_len_raw:
            data = data[:target_len_raw, :]

        # Downsample
        # 200Hz -> 50Hz
        num_samples = int(duration * target_sr)
        data = resample(data, num_samples, axis=0)

        # Handle NaNs (replace with 0)
        data = np.nan_to_num(data, nan=0.0)

        # Normalize (Channel-wise Z-score)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / (std + 1e-6)

        # Clip extreme outliers
        data = np.clip(data, -10, 10)

        return data.astype(np.float32)

    except Exception as e:
        # Return zero tensor on failure
        return np.zeros((Config.EEG_SEQ_LEN, Config.N_EEG_CHANNELS), dtype=np.float32)


def preprocess_spec(spec_path, offset_seconds, is_test=False):
    """
    Loads, slices, restacks, and resizes spectrogram data.
    """
    try:
        df = pd.read_parquet(spec_path)

        # Time Slicing
        if is_test:
            # Test files are exact 10m crops
            data_df = df
        else:
            # Train files are long; extract 10m window
            if "time" in df.columns:
                # Filter 600s window
                data_df = df[
                    (df["time"] >= offset_seconds) & (df["time"] < offset_seconds + 600)
                ]
                if data_df.empty:
                    # Fallback
                    start_idx = int(offset_seconds / 2)
                    data_df = df.iloc[start_idx : start_idx + 300]
            else:
                # Assume 2s per row if time column missing
                start_idx = int(offset_seconds / 2)
                data_df = df.iloc[start_idx : start_idx + 300]

        # Drop time column
        if "time" in data_df.columns:
            data_df = data_df.drop(columns=["time"])

        # Parse columns to separate 4 regions (LL, RL, LP, RP)
        regions = ["LL", "RL", "LP", "RP"]
        img_layers = []

        for region in regions:
            # Find columns for this region
            cols = [c for c in data_df.columns if c.startswith(f"{region}_")]

            if not cols:
                # Missing region fallback
                layer = np.zeros(Config.SPEC_SIZE, dtype=np.float32)
            else:
                # Sort columns by frequency value (e.g. "LL_0.59")
                col_map = []
                for c in cols:
                    try:
                        freq = float(c.split("_")[1])
                        col_map.append((c, freq))
                    except:
                        pass
                col_map.sort(key=lambda x: x[1])
                sorted_cols = [x[0] for x in col_map]

                # Extract data
                region_data = data_df[sorted_cols].values

                # Handle NaNs
                region_data = np.nan_to_num(region_data, nan=0.0)

                # Log transform
                region_data = np.log1p(region_data)

                # Resize to target size (256, 256)
                # cv2.resize expects (width, height). We map (Freq, Time) to (256, 256)
                layer = cv2.resize(
                    region_data, Config.SPEC_SIZE, interpolation=cv2.INTER_LINEAR
                )

            img_layers.append(layer)

        # Stack -> (256, 256, 4)
        img = np.stack(img_layers, axis=-1)

        # Transpose to (Channels, Height, Width) -> (4, 256, 256)
        img = img.transpose(2, 0, 1)

        # Normalize (Standardize)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-6)

        return img.astype(np.float32)

    except Exception as e:
        return np.zeros(
            (Config.N_SPEC_CHANNELS, Config.SPEC_SIZE[0], Config.SPEC_SIZE[1]),
            dtype=np.float32,
        )


def process_row(row, input_dir, is_test=False):
    """
    Helper to process a single row from metadata.
    """
    eeg_path = os.path.join(input_dir, row["eeg_path"])
    spec_path = os.path.join(input_dir, row["spec_path"])

    # EEG Processing
    eeg_offset = row.get("eeg_label_offset_seconds", 0)
    eeg_data = preprocess_eeg(eeg_path, eeg_offset)

    # Spectrogram Processing
    spec_offset = row.get("spectogram_label_offset_seconds", 0)
    spec_data = preprocess_spec(spec_path, spec_offset, is_test=is_test)

    # Target Processing
    if not is_test:
        target = row[Config.TARGET_COLS].values.astype(np.float32)
    else:
        target = np.zeros(len(Config.TARGET_COLS), dtype=np.float32)

    return eeg_data, spec_data, target


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG and Spectrogram data.
    """

    def __init__(self, eeg_data, spec_data, targets, augment=False):
        self.eeg_data = eeg_data
        self.spec_data = spec_data
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # Load data (mmap compatible)
        eeg = self.eeg_data[idx]
        spec = self.spec_data[idx]
        target = self.targets[idx]

        # Convert to tensor
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Apply augmentation
        if self.augment:
            spec_tensor = self.spec_augment(spec_tensor)

        return eeg_tensor, spec_tensor, target_tensor

    def spec_augment(self, spec, num_mask=2, freq_mask_param=25, time_mask_param=25):
        """
        Applies SpecAugment (Frequency and Time Masking).
        spec: (C, H, W)
        """
        C, H, W = spec.shape
        aug_spec = spec.clone()

        for _ in range(num_mask):
            # Frequency Masking
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, H - f))
            aug_spec[:, f0 : f0 + f, :] = 0.0

            # Time Masking
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, W - t))
            aug_spec[:, :, t0 : t0 + t] = 0.0

        return aug_spec


def get_dataloaders(debug=Config.DEBUG):
    """
    Generates Train, Val, and Test DataLoaders.
    Handles caching of processed data to disk.
    """
    seed_everything()

    def load_or_create_cache(csv_path, mode):
        df = pd.read_csv(csv_path)
        if debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)

        # Define cache paths
        eeg_cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_eeg.npy")
        spec_cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_spec.npy")
        target_cache_path = os.path.join(Config.CACHE_DIR, f"{mode}_targets.npy")

        # Check if cache exists
        if (
            os.path.exists(eeg_cache_path)
            and os.path.exists(spec_cache_path)
            and os.path.exists(target_cache_path)
        ):
            print(f"Loading cached {mode} data from {Config.CACHE_DIR}...")
            # Use mmap_mode to save RAM
            eeg_all = np.load(eeg_cache_path, mmap_mode="r")
            spec_all = np.load(spec_cache_path, mmap_mode="r")
            targets_all = np.load(target_cache_path, mmap_mode="r")
        else:
            print(f"Processing {mode} data from scratch (Rows: {len(df)})...")
            rows = [r for _, r in df.iterrows()]

            # Parallel processing if available
            if HAS_JOBLIB:
                results = Parallel(n_jobs=Config.NUM_WORKERS, backend="loky")(
                    delayed(process_row)(row, Config.INPUT_DIR, mode == "test")
                    for row in rows
                )
            else:
                results = [
                    process_row(row, Config.INPUT_DIR, mode == "test") for row in rows
                ]

            # Unpack results
            eeg_list, spec_list, target_list = zip(*results)

            # Stack into arrays
            eeg_all = np.stack(eeg_list)
            spec_all = np.stack(spec_list)
            targets_all = np.stack(target_list)

            # Save to cache
            np.save(eeg_cache_path, eeg_all)
            np.save(spec_cache_path, spec_all)
            np.save(target_cache_path, targets_all)

        return eeg_all, spec_all, targets_all

    # Load Data
    train_eeg, train_spec, train_targets = load_or_create_cache(
        Config.TRAIN_CSV, "train"
    )
    val_eeg, val_spec, val_targets = load_or_create_cache(Config.VAL_CSV, "val")
    test_eeg, test_spec, test_targets = load_or_create_cache(Config.TEST_CSV, "test")

    # Initialize Datasets
    train_dataset = EEGDataset(train_eeg, train_spec, train_targets, augment=True)
    val_dataset = EEGDataset(val_eeg, val_spec, val_targets, augment=False)
    test_dataset = EEGDataset(test_eeg, test_spec, test_targets, augment=False)

    # Initialize DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
