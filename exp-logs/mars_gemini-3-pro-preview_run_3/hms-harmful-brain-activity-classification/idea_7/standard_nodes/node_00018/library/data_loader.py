import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
import logging
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Configure Logger
logger = logging.getLogger("data_loader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class SpecAugment:
    """
    Applies Time and Frequency Masking to a Spectrogram.
    """

    def __init__(self, freq_mask_param, time_mask_param):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, spec):
        # spec shape: (Channels, Freq, Time)
        # torchaudio masking expects (..., Freq, Time)
        return self.time_mask(self.freq_mask(spec))


class EEGDataset(Dataset):
    def __init__(
        self,
        df,
        eeg_data,
        eeg_index_map,
        spec_data,
        spec_index_map,
        config,
        mode="train",
        augment=False,
    ):
        self.df = df.reset_index(drop=True)
        self.eeg_data = eeg_data
        self.eeg_index_map = eeg_index_map  # Dict mapping eeg_id -> (start, len)
        self.spec_data = spec_data
        self.spec_index_map = spec_index_map  # Dict mapping spec_id -> (start, len)
        self.config = config
        self.mode = mode
        self.augment = augment

        # MelSpectrogram Transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.EEG_SR,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        )

        # Augmentation
        self.spec_augment = SpecAugment(config.FREQ_MASK_PARAM, config.TIME_MASK_PARAM)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ==========================
        # Stream A: EEG Processing
        # ==========================
        eeg_id = row["eeg_id"]
        # Retrieve raw EEG from big array
        if eeg_id in self.eeg_index_map:
            start_idx, length = self.eeg_index_map[eeg_id]
            raw_eeg = self.eeg_data[start_idx : start_idx + length]
        else:
            # Fallback (should not happen if cache is correct)
            raw_eeg = np.zeros(
                (self.config.EEG_SAMPLES, len(self.config.EEG_CHANNELS)),
                dtype=np.float32,
            )

        # Slice 50s window
        # eeg_label_offset_seconds points to the start of the 10s labeled center.
        # The 50s sample is centered on this.
        # Wait, prompt says: "metadata... allows you to extract the original subsets... 50 second long subsample"
        # And: "eeg_label_offset_seconds - The time between the beginning of the consolidated EEG and this subsample."
        # So offset is the start of the 50s window.
        offset_sec = row.get("eeg_label_offset_seconds", 0)
        offset_samples = int(offset_sec * self.config.EEG_SR)

        # Handle potential bounds
        end_samples = offset_samples + self.config.EEG_SAMPLES

        if offset_samples < 0:
            offset_samples = 0

        # Extract crop
        if end_samples <= raw_eeg.shape[0]:
            eeg_crop = raw_eeg[offset_samples:end_samples]
        else:
            # Padding if window goes out of bounds
            eeg_crop = raw_eeg[offset_samples:]
            pad_len = self.config.EEG_SAMPLES - eeg_crop.shape[0]
            if pad_len > 0:
                eeg_crop = np.pad(eeg_crop, ((0, pad_len), (0, 0)), mode="constant")

        # Ensure shape (Time, Channels) -> (Channels, Time) for MelSpec
        eeg_tensor = torch.tensor(eeg_crop, dtype=torch.float32).permute(
            1, 0
        )  # (19, 10000)

        # Replace NaNs
        eeg_tensor = torch.nan_to_num(eeg_tensor, nan=0.0)

        # Compute MelSpectrogram
        # Output: (19, 128, Time)
        mel_spec = self.mel_transform(eeg_tensor)

        # Log Transform: log(S + eps)
        mel_spec = torch.log(mel_spec + 1e-6)

        # Resize to fixed width (512)
        # mel_spec shape: (C, Freq, Time). Interpolate expects (N, C, H, W) or (1, C, H, W)
        # We treat Freq as Height, Time as Width
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0),
            size=self.config.IMG_SIZE_EEG,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Instance Normalization (Standardize per sample)
        mean = mel_spec.mean(dim=(1, 2), keepdim=True)
        std = mel_spec.std(dim=(1, 2), keepdim=True)
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        # ==========================
        # Stream B: Spec Processing
        # ==========================
        spec_id = row["spectrogram_id"]
        if spec_id in self.spec_index_map:
            s_start, s_len = self.spec_index_map[spec_id]
            raw_spec = self.spec_data[s_start : s_start + s_len]
        else:
            raw_spec = np.zeros((300, 400), dtype=np.float32)

        # Slice 10m window
        # spectogram_label_offset_seconds
        spec_offset_sec = row.get("spectogram_label_offset_seconds", 0)
        # Kaggle specs are usually 0.5Hz resolution? Or 2s per row?
        # The prompt says "10 minute window".
        # The raw parquet usually covers much more.
        # We need to determine the row index from seconds.
        # Typically Kaggle specs are 2 seconds per row (0.5 Hz is freq res, but time res is 2s).
        # Let's assume 2s per row based on common dataset knowledge (300 rows = 600s = 10m).
        spec_offset_rows = int(spec_offset_sec / 2.0)
        spec_window_rows = 300  # 10 mins * 60 / 2 = 300 rows

        s_end = spec_offset_rows + spec_window_rows

        if s_end <= raw_spec.shape[0]:
            spec_crop = raw_spec[spec_offset_rows:s_end]
        else:
            spec_crop = raw_spec[spec_offset_rows:]
            pad_len = spec_window_rows - spec_crop.shape[0]
            if pad_len > 0:
                spec_crop = np.pad(spec_crop, ((0, pad_len), (0, 0)), mode="constant")

        # Raw spec shape: (Time, 400). 400 columns = 100 freq bins * 4 regions.
        # Reshape to (Time, 100, 4) -> (4, 100, Time)
        # Columns are usually LL, RL, LP, RP blocks.
        # We assume they are ordered.
        # Parse into 4 channels
        # (300, 400)
        spec_crop = np.nan_to_num(spec_crop, nan=0.0)  # Handle NaNs before log

        # Log transform
        spec_crop = np.log(spec_crop + 1e-6)

        # Reshape and Permute
        # Assumption: Columns 0-99 LL, 100-199 RL, 200-299 LP, 300-399 RP
        # This is standard for this dataset.
        spec_4ch = np.zeros((4, 100, 300), dtype=np.float32)
        for i in range(4):
            # Transpose to (Freq, Time)
            spec_4ch[i] = spec_crop[:, i * 100 : (i + 1) * 100].T

        # Resize to (256, 256)
        # We use cv2 for resizing numpy arrays
        resized_spec = np.zeros(
            (4, self.config.IMG_SIZE_SPEC[0], self.config.IMG_SIZE_SPEC[1]),
            dtype=np.float32,
        )
        for i in range(4):
            # cv2.resize expects (Width, Height) -> (Time, Freq)
            # We have (Freq, Time).
            # Let's resize (100, 300) -> (256, 256)
            img = spec_4ch[i]  # (100, 300)
            # cv2 resize takes (W, H). We want output (256, 256).
            resized_spec[i] = cv2.resize(
                img, self.config.IMG_SIZE_SPEC, interpolation=cv2.INTER_LINEAR
            )

        spec_tensor = torch.tensor(resized_spec, dtype=torch.float32)

        # Normalize Spec (Instance Norm)
        s_mean = spec_tensor.mean(dim=(1, 2), keepdim=True)
        s_std = spec_tensor.std(dim=(1, 2), keepdim=True)
        spec_tensor = (spec_tensor - s_mean) / (s_std + 1e-6)

        # ==========================
        # Augmentation & Return
        # ==========================
        if self.augment:
            mel_spec = self.spec_augment(mel_spec)
            spec_tensor = self.spec_augment(spec_tensor)

        if self.mode == "test":
            return (mel_spec, spec_tensor)
        else:
            # Targets
            target_cols = self.config.TARGET_COLS
            target = torch.tensor(
                row[target_cols].values.astype(np.float32), dtype=torch.float32
            )
            return (mel_spec, spec_tensor), target


def process_and_cache_eeg(unique_ids, base_dir, cache_path_data, cache_path_map):
    """
    Reads EEG parquet files, concatenates them into a single memory-mapped array,
    and saves the index map.
    """
    logger.info(f"Processing {len(unique_ids)} EEG files from {base_dir}...")

    data_list = []
    map_list = []  # (id, start, len)
    current_idx = 0

    # Use Config channels for consistency
    channels = Config.EEG_CHANNELS

    for eid in unique_ids:
        path = os.path.join(base_dir, f"{eid}.parquet")
        try:
            df = pd.read_parquet(path, columns=channels)
            vals = df.values.astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to read EEG {eid}: {e}. Using zeros.")
            vals = np.zeros((10000, 19), dtype=np.float32)  # Fallback size

        length = len(vals)
        data_list.append(vals)
        map_list.append([eid, current_idx, length])
        current_idx += length

    # Concatenate
    logger.info("Concatenating EEG data...")
    big_data = np.concatenate(data_list, axis=0)
    index_map = np.array(map_list, dtype=np.int64)

    # Save
    logger.info(
        f"Saving EEG cache to {cache_path_data} ({big_data.nbytes / 1e9:.2f} GB)..."
    )
    np.save(cache_path_data, big_data)
    np.save(cache_path_map, index_map)

    return big_data, index_map


def process_and_cache_specs(unique_ids, base_dir, cache_path_data, cache_path_map):
    """
    Reads Spectrogram parquet files, concatenates, and saves.
    """
    logger.info(f"Processing {len(unique_ids)} Spectrogram files from {base_dir}...")

    data_list = []
    map_list = []
    current_idx = 0

    for sid in unique_ids:
        path = os.path.join(base_dir, f"{sid}.parquet")
        try:
            df = pd.read_parquet(path)
            # Drop time column if exists, usually index is time, columns are freqs
            # We assume columns are the 400 data columns.
            # Filter just in case
            vals = df.values.astype(np.float32)
            # If time column is included (it shouldn't be in values if read correctly, but check)
            if vals.shape[1] > 400:
                vals = vals[:, 1:]  # Assume first is time
        except Exception as e:
            logger.warning(f"Failed to read Spec {sid}: {e}")
            vals = np.zeros((300, 400), dtype=np.float32)

        length = len(vals)
        data_list.append(vals)
        map_list.append([sid, current_idx, length])
        current_idx += length

    logger.info("Concatenating Spec data...")
    big_data = np.concatenate(data_list, axis=0)
    index_map = np.array(map_list, dtype=np.int64)

    logger.info(f"Saving Spec cache to {cache_path_data}...")
    np.save(cache_path_data, big_data)
    np.save(cache_path_map, index_map)

    return big_data, index_map


def load_cached_structure(cache_dir, prefix, ids, raw_dir, type_name):
    """
    Generic loader for the Big-Array + Map structure.
    type_name: 'eeg' or 'spec'
    """
    data_path = os.path.join(cache_dir, f"{prefix}_{type_name}_data.npy")
    map_path = os.path.join(cache_dir, f"{prefix}_{type_name}_map.npy")

    if os.path.exists(data_path) and os.path.exists(map_path):
        logger.info(f"Loading cached {type_name} data from {data_path}...")
        big_data = np.load(
            data_path, mmap_mode="r"
        )  # Use mmap to save RAM if needed, or load fully
        # Given 220GB RAM, we can load fully for speed
        big_data = np.array(big_data)
        index_map_arr = np.load(map_path)
    else:
        logger.info(f"Cache miss for {prefix} {type_name}. Generating...")
        if type_name == "eeg":
            big_data, index_map_arr = process_and_cache_eeg(
                ids, raw_dir, data_path, map_path
            )
        else:
            big_data, index_map_arr = process_and_cache_specs(
                ids, raw_dir, data_path, map_path
            )

    # Convert map array to dict for O(1) access
    # index_map_arr: [[id, start, len], ...]
    index_map = {int(row[0]): (int(row[1]), int(row[2])) for row in index_map_arr}

    return big_data, index_map


def get_dataloaders(train_df, val_df, test_df, load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    Handles caching for Train/Val (combined) and Test (separate).
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Prepare Train/Val Data
    # Combine IDs to create a unified cache for training data
    train_eeg_ids = train_df["eeg_id"].unique()
    val_eeg_ids = val_df["eeg_id"].unique()
    train_val_eeg_ids = np.unique(np.concatenate([train_eeg_ids, val_eeg_ids]))

    train_spec_ids = train_df["spectrogram_id"].unique()
    val_spec_ids = val_df["spectrogram_id"].unique()
    train_val_spec_ids = np.unique(np.concatenate([train_spec_ids, val_spec_ids]))

    # Load/Cache TrainVal EEG
    tv_eeg_data, tv_eeg_map = load_cached_structure(
        Config.WORKING_DIR, "trainval", train_val_eeg_ids, Config.TRAIN_EEGS_DIR, "eeg"
    )
    # Load/Cache TrainVal Spec
    tv_spec_data, tv_spec_map = load_cached_structure(
        Config.WORKING_DIR,
        "trainval",
        train_val_spec_ids,
        Config.TRAIN_SPECS_DIR,
        "spec",
    )

    # 2. Prepare Test Data
    test_eeg_ids = test_df["eeg_id"].unique()
    test_spec_ids = test_df["spectrogram_id"].unique()

    test_eeg_data, test_eeg_map = load_cached_structure(
        Config.WORKING_DIR, "test", test_eeg_ids, Config.TEST_EEGS_DIR, "eeg"
    )
    test_spec_data, test_spec_map = load_cached_structure(
        Config.WORKING_DIR, "test", test_spec_ids, Config.TEST_SPECS_DIR, "spec"
    )

    # 3. Create Datasets
    train_dataset = EEGDataset(
        train_df,
        tv_eeg_data,
        tv_eeg_map,
        tv_spec_data,
        tv_spec_map,
        Config,
        mode="train",
        augment=True,
    )

    val_dataset = EEGDataset(
        val_df,
        tv_eeg_data,
        tv_eeg_map,
        tv_spec_data,
        tv_spec_map,
        Config,
        mode="val",
        augment=False,
    )

    test_dataset = EEGDataset(
        test_df,
        test_eeg_data,
        test_eeg_map,
        test_spec_data,
        test_spec_map,
        Config,
        mode="test",
        augment=False,
    )

    # 4. Create Loaders
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
