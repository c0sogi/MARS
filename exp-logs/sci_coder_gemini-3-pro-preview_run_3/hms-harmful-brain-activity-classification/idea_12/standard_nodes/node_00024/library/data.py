import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


# ==========================================
# Caching & Data Loading Logic
# ==========================================
def process_spectrogram(spec_df, offset_seconds):
    """
    Extracts 10-minute window, separates regions, log-transforms, and resizes.
    Returns: numpy array of shape (256, 256, 4)
    """
    # 1. Crop 10-minute window (600 seconds)
    # Assuming dataframe index is time in seconds (standard for this dataset)
    # Fallback to row calculation if index is not appropriate
    try:
        # loc includes the end label, so we might get slightly more, but we resize anyway
        subset = spec_df.loc[offset_seconds : offset_seconds + 600]
    except:
        # Fallback: assume 0.5Hz (2s per row) if index lookup fails
        start_idx = int(offset_seconds / 2)
        subset = spec_df.iloc[start_idx : start_idx + 300]

    if subset.empty:
        # Edge case: return zeros
        return np.zeros((256, 256, 4), dtype=np.float32)

    # 2. Handle NaNs
    data = subset.fillna(0).values  # Shape: (Time, Freq_Cols)

    # 3. Separate Regions
    # Columns are like 'LL_0.59', 'RL_0.59', etc.
    cols = subset.columns
    regions = ["LL", "RL", "LP", "RP"]
    processed_regions = []

    for region in regions:
        # Filter columns for this region
        region_cols = [c for c in cols if c.startswith(region)]
        if not region_cols:
            # Should not happen, but safe fallback
            region_img = np.zeros((256, 256), dtype=np.float32)
        else:
            # Extract raw region data (Time, Freq)
            # We want (Freq, Time) for standard image view, but we resize anyway
            # Let's keep it as (Time, Freq) for now and resize
            region_data = subset[region_cols].fillna(0).values

            # Log transform
            region_data = np.log1p(np.abs(region_data))

            # Resize to (256, 256)
            # cv2.resize expects (Width, Height).
            # We want final shape (256, 256).
            region_img = cv2.resize(
                region_data, Config.STREAM_B_IMG_SIZE, interpolation=cv2.INTER_LINEAR
            )

        processed_regions.append(region_img)

    # Stack depth-wise: (256, 256, 4)
    return np.stack(processed_regions, axis=-1).astype(np.float32)


def process_eeg(eeg_df, offset_seconds):
    """
    Extracts 50-second window (10,000 samples) for the 19 channels.
    Returns: numpy array of shape (19, 10000)
    """
    fs = Config.SAMPLING_RATE
    start_sample = int(offset_seconds * fs)
    end_sample = start_sample + int(Config.EEG_DURATION * fs)

    # Extract channels
    try:
        # Select only the required 19 channels
        cols = Config.EEG_CHANNELS
        # iloc for row slicing
        subset = eeg_df.iloc[start_sample:end_sample]
        data = subset[cols].values.T  # Transpose to (Channels, Time)
    except Exception:
        # Padding or error handling
        data = np.zeros((Config.N_EEG_CHANNELS, int(Config.EEG_DURATION * fs)))

    # Handle NaNs
    data = np.nan_to_num(data, nan=0.0)

    # Pad if shorter than expected (e.g. end of file)
    if data.shape[1] < int(Config.EEG_DURATION * fs):
        pad_width = int(Config.EEG_DURATION * fs) - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

    return data.astype(np.float32)


def load_and_cache_data(mode="train", load_cached_data=True, debug=False):
    """
    Loads metadata, reads raw files, processes them into arrays, and caches them.
    mode: 'train', 'val', or 'test'
    """
    # Define paths
    cache_dir = Config.WORKING_DIR
    eeg_cache_path = os.path.join(cache_dir, f"{mode}_eeg_cache.npy")
    spec_cache_path = os.path.join(cache_dir, f"{mode}_spec_cache.npy")
    target_cache_path = os.path.join(cache_dir, f"{mode}_targets.npy")

    # Load Metadata
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_CSV)
    else:
        df = pd.read_csv(Config.TEST_CSV)

    if debug:
        df = df.iloc[:200]

    # Check Cache
    if (
        load_cached_data
        and os.path.exists(eeg_cache_path)
        and os.path.exists(spec_cache_path)
    ):
        print(f"Loading cached data for {mode}...")
        eeg_data = np.load(eeg_cache_path)
        spec_data = np.load(spec_cache_path)
        if mode != "test":
            targets = np.load(target_cache_path)
        else:
            targets = None
        return eeg_data, spec_data, targets, df

    print(f"Processing raw data for {mode} (Cache miss or force reload)...")

    # Pre-allocate arrays
    n_samples = len(df)
    eeg_data = np.zeros(
        (
            n_samples,
            Config.N_EEG_CHANNELS,
            int(Config.EEG_DURATION * Config.SAMPLING_RATE),
        ),
        dtype=np.float32,
    )
    spec_data = np.zeros(
        (
            n_samples,
            Config.STREAM_B_IMG_SIZE[0],
            Config.STREAM_B_IMG_SIZE[1],
            Config.IN_CHANNELS_B,
        ),
        dtype=np.float32,
    )

    if mode != "test":
        target_cols = [c.replace("_vote", "_prob") for c in Config.CLASS_NAMES]
        targets = df[target_cols].values.astype(np.float32)
    else:
        targets = None

    # Processing Loop
    # We use a cache for file reading to avoid re-reading the same parquet file multiple times
    # if multiple samples come from the same recording.
    # However, for memory safety, we won't cache too many full files.
    # Given the dataset structure, we'll just read as we go. OS file cache helps.

    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Processed {i}/{n_samples} samples...")

        # --- Process EEG ---
        eeg_path = os.path.join(Config.INPUT_DIR, row["eeg_path"])
        try:
            raw_eeg = pd.read_parquet(eeg_path)
            offset = row.get("eeg_label_offset_seconds", 0)
            eeg_data[i] = process_eeg(raw_eeg, offset)
        except Exception as e:
            print(f"Error processing EEG {eeg_path}: {e}")

        # --- Process Spectrogram ---
        spec_path = os.path.join(Config.INPUT_DIR, row["spec_path"])
        try:
            raw_spec = pd.read_parquet(spec_path)
            # For test set, metadata might not have offset, default to 0 if missing?
            # Test.csv does NOT have offsets in the provided description,
            # but usually test samples are just the 50s/10m clips directly.
            # However, provided test.csv schema in prompt only lists IDs.
            # We assume test files are already cropped or offsets are 0.
            # Looking at prompt: "test.csv... As there are no overlapping samples... many columns don't apply"
            # "test_eegs/ Exactly 50 seconds of EEG data."
            # "test_spectrograms/ Spectrograms assembled using exactly 10 minutes."
            offset = row.get("spectogram_label_offset_seconds", 0)
            spec_data[i] = process_spectrogram(raw_spec, offset)
        except Exception as e:
            print(f"Error processing Spec {spec_path}: {e}")

    # Save to Cache
    print(f"Saving cache for {mode}...")
    np.save(eeg_cache_path, eeg_data)
    np.save(spec_cache_path, spec_data)
    if targets is not None:
        np.save(target_cache_path, targets)

    return eeg_data, spec_data, targets, df


# ==========================================
# Dataset Class
# ==========================================
class EEGDataset(Dataset):
    def __init__(
        self, eeg_data, spec_data, targets=None, mode="train", transform=False
    ):
        self.eeg_data = eeg_data
        self.spec_data = spec_data
        self.targets = targets
        self.mode = mode
        self.transform = transform

        # Augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.eeg_data)

    def generate_physio_views(self, eeg_signal):
        """
        Generates 3 Mel-Spectrogram views (Slow, Fast, Broadband) from raw EEG.
        eeg_signal: (Channels, Time) numpy array
        Returns: Tensor (57, 128, 500)
        """
        # Convert to torch
        waveform = torch.from_numpy(eeg_signal).float()  # (19, 10000)

        views = []

        for view_config in Config.PHYSIO_VIEWS:
            # Calculate STFT params
            win_length = int(view_config["window_size_sec"] * Config.SAMPLING_RATE)
            hop_length = int(win_length * view_config["hop_length_ratio"])
            n_fft = 2 ** int(np.ceil(np.log2(win_length)))  # Next power of 2
            if n_fft < win_length:
                n_fft = win_length

            # Create MelSpectrogram transform
            # We construct it here (lightweight) or could cache in __init__
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=Config.SAMPLING_RATE,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=view_config["fmin"],
                f_max=view_config["fmax"],
                n_mels=Config.STREAM_A_IMG_SIZE[0],  # 128
                center=True,
                pad_mode="reflect",
                power=2.0,
            )

            # Apply transform -> (19, n_mels, time_steps)
            melspec = mel_transform(waveform)

            # Log transform (Power to DB)
            melspec = torchaudio.transforms.AmplitudeToDB()(melspec)

            # Resize to common time dimension (500)
            # Input to interpolate must be (Batch, Channels, Height, Width) or (Batch, Channels, Length)
            # Here we have (Channels, Freq, Time). Treat Channels as Batch or keep 3D?
            # Interpolate expects 3D (MiniBatch, Channels, Length) for 1D, or 4D for 2D.
            # We treat (19, 128, Time) as a batch of 19 images of size (128, Time)
            # Unsqueeze to (19, 1, 128, Time)
            melspec = melspec.unsqueeze(1)

            # Resize
            # Config.STREAM_A_IMG_SIZE is (128, 500) -> (Freq, Time)
            target_size = Config.STREAM_A_IMG_SIZE
            melspec = torch.nn.functional.interpolate(
                melspec, size=target_size, mode="bilinear", align_corners=False
            )

            # Squeeze back to (19, 128, 500)
            melspec = melspec.squeeze(1)

            # Normalize (simple instance normalization per view/channel to stabilize)
            mean = melspec.mean(dim=(1, 2), keepdim=True)
            std = melspec.std(dim=(1, 2), keepdim=True) + 1e-6
            melspec = (melspec - mean) / std

            views.append(melspec)

        # Stack all views along channel dimension
        # 3 views * 19 channels = 57 channels
        combined = torch.cat(views, dim=0)  # (57, 128, 500)

        return combined

    def __getitem__(self, idx):
        # --- Stream A: EEG -> Physio Views ---
        raw_eeg = self.eeg_data[idx]  # (19, 10000)
        stream_a = self.generate_physio_views(raw_eeg)  # (57, 128, 500)

        # --- Stream B: Pre-processed Spectrogram ---
        # Data is (256, 256, 4). Convert to (4, 256, 256)
        raw_spec = self.spec_data[idx]
        stream_b = torch.from_numpy(raw_spec).permute(2, 0, 1).float()

        # Normalize Stream B (Standard ImageNet-like or simple stats)
        # Using simple instance norm
        mean_b = stream_b.mean(dim=(1, 2), keepdim=True)
        std_b = stream_b.std(dim=(1, 2), keepdim=True) + 1e-6
        stream_b = (stream_b - mean_b) / std_b

        # --- Augmentation ---
        if self.transform and self.mode == "train":
            # Apply masking to Stream A
            stream_a = self.freq_mask(stream_a)
            stream_a = self.time_mask(stream_a)

            # Apply masking to Stream B
            stream_b = self.freq_mask(stream_b)
            stream_b = self.time_mask(stream_b)

        # --- Target ---
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (stream_a, stream_b), target
        else:
            return (stream_a, stream_b)


# ==========================================
# Data Loaders
# ==========================================
def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test.
    """
    # Load Data
    train_eeg, train_spec, train_targets, _ = load_and_cache_data(
        "train", load_cached_data=True, debug=debug
    )
    val_eeg, val_spec, val_targets, _ = load_and_cache_data(
        "val", load_cached_data=True, debug=debug
    )
    test_eeg, test_spec, _, _ = load_and_cache_data(
        "test", load_cached_data=True, debug=debug
    )

    # Debug Mode: Subset
    if debug:
        subset_size = 200
        train_eeg = train_eeg[:subset_size]
        train_spec = train_spec[:subset_size]
        train_targets = train_targets[:subset_size]
        val_eeg = val_eeg[:subset_size]
        val_spec = val_spec[:subset_size]
        val_targets = val_targets[:subset_size]

    # Datasets
    train_dataset = EEGDataset(
        train_eeg, train_spec, train_targets, mode="train", transform=True
    )
    val_dataset = EEGDataset(
        val_eeg, val_spec, val_targets, mode="val", transform=False
    )
    test_dataset = EEGDataset(test_eeg, test_spec, None, mode="test", transform=False)

    # Loaders
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
