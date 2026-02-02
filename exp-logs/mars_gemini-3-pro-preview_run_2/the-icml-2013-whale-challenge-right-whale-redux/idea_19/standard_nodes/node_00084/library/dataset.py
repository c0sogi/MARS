import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library import config

# Ensure deterministic behavior for transforms
torch.manual_seed(config.SEED)


def compute_spectrogram(file_path):
    """
    Computes the Mel Spectrogram for a given audio file using the 'Golden Recipe'.

    Steps:
    1. Load audio.
    2. Compute Mel Spectrogram (High Res).
    3. Convert to dB (Power to dB).
    4. Clamp dynamic range (Top DB).
    5. Instance Standardization.
    """
    try:
        # Load audio using soundfile (robust for .aif)
        audio, sr = sf.read(file_path)

        # Ensure audio is float32
        audio = audio.astype(np.float32)

        # Handle channels: if stereo, average to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Convert to tensor
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, time)

        # Define Mel Spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
            normalized=False,
        )

        # Compute Spec
        spec = mel_transform(waveform)

        # Convert to dB (Power to dB)
        # 10 * log10(x)
        spec_db = 10.0 * torch.log10(torch.clamp(spec, min=1e-10))

        # Top DB Clamping
        max_val = spec_db.max()
        min_val = max_val - config.TOP_DB
        spec_db = torch.clamp(spec_db, min=min_val)

        # Instance Standardization
        mean = spec_db.mean()
        std = spec_db.std()

        if std > 1e-6:
            spec_norm = (spec_db - mean) / std
        else:
            spec_norm = spec_db - mean

        return spec_norm.numpy()  # Returns (1, n_mels, time)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero array of expected shape as fallback
        # Expected time dim: ~2s * 2000Hz / 64 hop ~ 63 frames
        return np.zeros((1, config.N_MELS, 63), dtype=np.float32)


def get_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from cache or computes it from scratch.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data_array, targets_array, clip_names_array)
    """
    # Ensure cache directory exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    data_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_data.npy")
    target_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_targets.npy")
    clip_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_clips.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(data_cache_path) and os.path.exists(target_cache_path):
            print(f"Loading {cache_prefix} data from cache...")
            data = np.load(data_cache_path)
            targets = np.load(target_cache_path)

            # Clips are optional (mostly for test)
            clips = None
            if os.path.exists(clip_cache_path):
                clips = np.load(clip_cache_path, allow_pickle=True)

            return data, targets, clips

    # 2. Compute from scratch
    print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    # Pre-allocate list
    data_list = []
    targets_list = []
    clips_list = []

    for _, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(config.INPUT_ROOT, row["file_path"])

        # Compute Spectrogram
        spec = compute_spectrogram(full_path)
        data_list.append(spec)

        # Handle Targets
        if "label" in row:
            targets_list.append(row["label"])
        else:
            targets_list.append(-1)  # Placeholder for test

        # Handle Clip names
        if "clip" in row:
            clips_list.append(row["clip"])
        else:
            clips_list.append("")

    # Stack into arrays
    # Note: Different files might have slightly different lengths due to rounding.
    # We need to ensure consistent shape or handle it.
    # Based on analysis, all are 2.0s or close. We will pad/crop to a fixed width if necessary
    # or rely on the fact that batching handles it if we use a custom collate,
    # but for simplicity and speed with CNNs, fixed size is better.
    # The analysis showed strict 2000Hz and ~2s.
    # Let's find max length and pad, or crop to median.
    # For this implementation, we assume consistency or pad to max.

    # Find max time dimension
    max_len = max([x.shape[2] for x in data_list])

    # Pad to max_len
    padded_data = []
    for x in data_list:
        pad_amt = max_len - x.shape[2]
        if pad_amt > 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad_amt)), mode="constant")
        elif pad_amt < 0:
            x = x[:, :, :max_len]
        padded_data.append(x)

    data_array = np.stack(padded_data).astype(np.float32)
    targets_array = np.array(targets_list, dtype=np.int64)
    clips_array = np.array(clips_list)

    # Save to cache
    np.save(data_cache_path, data_array)
    np.save(target_cache_path, targets_array)
    if len(clips_list) > 0:
        np.save(clip_cache_path, clips_array)

    return data_array, targets_array, clips_array


class WhaleDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (np.ndarray): Shape (N, 1, F, T)
            targets (np.ndarray): Shape (N,)
            transform (callable, optional): Transform to apply to the spectrogram.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data (C, F, T)
        spec = torch.from_numpy(self.data[idx])
        target = self.targets[idx]

        if self.transform:
            spec = self.transform(spec)

        return spec, torch.tensor(target, dtype=torch.float32)


def get_transforms(phase):
    """
    Returns transforms for the given phase.
    Aggressive SpecAugment for training.
    """
    if phase == "train":
        return torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=config.FREQ_MASK_PARAM
            ),
            torchaudio.transforms.TimeMasking(time_mask_param=config.TIME_MASK_PARAM),
        )
    else:
        return None


def get_dataloaders(load_cached_data=True, debug_limit=None):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug_limit (int, optional): Limit dataset size for debugging.

    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader, 'test_clips': clips}
    """
    # 1. Load Data
    train_X, train_y, _ = get_data(config.TRAIN_CSV, "train", load_cached_data)
    val_X, val_y, _ = get_data(config.VAL_CSV, "val", load_cached_data)
    test_X, test_y, test_clips = get_data(config.TEST_CSV, "test", load_cached_data)

    # Debugging
    if debug_limit:
        train_X, train_y = train_X[:debug_limit], train_y[:debug_limit]
        val_X, val_y = val_X[:debug_limit], val_y[:debug_limit]
        test_X, test_y = test_X[:debug_limit], test_y[:debug_limit]
        test_clips = test_clips[:debug_limit]

    # 2. Datasets
    train_dataset = WhaleDataset(train_X, train_y, transform=get_transforms("train"))
    val_dataset = WhaleDataset(val_X, val_y, transform=get_transforms("val"))
    test_dataset = WhaleDataset(test_X, test_y, transform=get_transforms("test"))

    # 3. Sampler for Class Imbalance
    # Calculate weights
    class_counts = np.bincount(train_y)
    # Handle potential zero count if debug_limit is small
    if len(class_counts) < 2:
        class_weights = np.ones(2)
    else:
        class_weights = 1.0 / class_counts

    sample_weights = class_weights[train_y]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_clips": test_clips,
    }
