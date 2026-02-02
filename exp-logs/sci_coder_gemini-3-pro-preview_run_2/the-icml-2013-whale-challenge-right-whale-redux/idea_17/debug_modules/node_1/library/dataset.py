import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Call Detection.
    """

    def __init__(self, data, targets=None, mode="train"):
        """
        Args:
            data (np.ndarray): Array of spectrograms (N, n_mels, time_steps).
            targets (np.ndarray, optional): Array of labels (N,).
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.data = data
        self.targets = targets
        self.mode = mode

        # Augmentations
        # Aggressive frequency masking as per 'Golden Recipe'
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load spectrogram: (n_mels, time_steps)
        spec = torch.tensor(self.data[idx], dtype=torch.float32)

        # Apply Augmentations only in training mode
        if self.mode == "train":
            spec = self.freq_masking(spec)

        # Add channel dimension: (1, n_mels, time_steps) for CNN input
        spec = spec.unsqueeze(0)

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            return spec, label
        else:
            # For test set, return spectrogram and dummy label (or index)
            # Returning just spec usually, but returning dummy for consistency if needed
            return spec, torch.tensor(0.0)


def load_and_process_audio(file_path):
    """
    Loads audio, computes MelSpectrogram, converts to DB, and applies Instance Norm.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    # Load audio
    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception as e:
        # Fallback or error handling; usually shouldn't happen with clean metadata
        print(f"Error loading {full_path}: {e}")
        return None

    # Resample if necessary (though dataset is known to be 2000Hz)
    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=Config.SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Enforce fixed length (Pad/Truncate)
    target_length = int(Config.SAMPLE_RATE * Config.DURATION)
    num_samples = waveform.size(-1)

    if num_samples < target_length:
        pad_amount = target_length - num_samples
        # Pad last dimension on the right
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif num_samples > target_length:
        # Truncate
        waveform = waveform[:, :target_length]

    # Generate Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )
    spec = mel_transform(waveform)

    # Convert to DB (Log-Mel)
    # top_db clamps the dynamic range to fix the noise floor
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)
    spec = db_transform(spec)

    # Instance Normalization (Zero-Mean, Unit-Variance per clip)
    mean = spec.mean()
    std = spec.std()
    if std > 0:
        spec = (spec - mean) / std
    else:
        spec = spec - mean

    # Squeeze channel dim if present (1, F, T) -> (F, T) for storage efficiency
    return spec.squeeze(0).numpy()


def get_data(df, cache_name, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    """
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    targets_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_targets.npy")
    clips_path = os.path.join(
        Config.WORKING_DIR, f"{cache_name}_clips.npy"
    )  # For test set tracking

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(data_path):
        print(f"Loading {cache_name} data from cache...")
        data = np.load(data_path)

        targets = None
        if os.path.exists(targets_path):
            targets = np.load(targets_path)

        clips = None
        if os.path.exists(clips_path):
            clips = np.load(clips_path, allow_pickle=True)

        return data, targets, clips

    # 2. Process from Scratch
    print(f"Processing {cache_name} data from scratch...")

    data_list = []
    targets_list = []
    clips_list = []

    # Iterate through metadata
    for _, row in df.iterrows():
        spec = load_and_process_audio(row["file_path"])
        if spec is not None:
            data_list.append(spec)
            if "label" in row:
                targets_list.append(row["label"])
            if "clip" in row:
                clips_list.append(row["clip"])

    data = np.stack(data_list)

    targets = None
    if targets_list:
        targets = np.array(targets_list, dtype=np.float32)

    clips = None
    if clips_list:
        clips = np.array(clips_list)

    # 3. Save to Cache
    print(f"Saving {cache_name} data to cache at {Config.WORKING_DIR}...")
    np.save(data_path, data)
    if targets is not None:
        np.save(targets_path, targets)
    if clips is not None:
        np.save(clips_path, clips)

    return data, targets, clips


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        train_loader, val_loader, test_loader, test_clips
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("Debug mode: Using subset of data.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)
        cache_suffix = "_debug"
    else:
        cache_suffix = ""

    # Process/Load Data
    train_data, train_targets, _ = get_data(
        train_df, f"train{cache_suffix}", load_cached_data
    )
    val_data, val_targets, _ = get_data(val_df, f"val{cache_suffix}", load_cached_data)
    test_data, _, test_clips = get_data(
        test_df, f"test{cache_suffix}", load_cached_data
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_data, train_targets, mode="train")
    val_dataset = WhaleDataset(val_data, val_targets, mode="val")
    test_dataset = WhaleDataset(test_data, targets=None, mode="test")

    # --- Weighted Random Sampler for Training ---
    # Calculate weights to balance the classes
    class_counts = np.bincount(train_targets.astype(int))
    # Handle potential zero counts in debug mode
    if len(class_counts) < 2:
        class_weights = [1.0, 1.0]
    else:
        class_weights = 1.0 / class_counts

    sample_weights = [class_weights[int(t)] for t in train_targets]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Use sampler instead of shuffle=True
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
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

    print(
        f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )

    return train_loader, val_loader, test_loader, test_clips
