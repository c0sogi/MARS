import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import PathConfig, AudioConfig, TrainConfig, LABEL_TO_IDX


class SpeechCommandsDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands.
    Holds all waveforms in memory as a NumPy array for fast access.
    """

    def __init__(self, waveforms: np.ndarray, targets: np.ndarray):
        """
        Args:
            waveforms: (N, 1, T) float32 array of audio data.
            targets: (N,) int64 array of label indices.
        """
        self.waveforms = waveforms
        self.targets = targets

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # Convert to tensor; data is already float32
        waveform = torch.from_numpy(self.waveforms[idx])
        label = self.targets[idx]
        return waveform, label


def process_audio(file_path, target_length=16000):
    """
    Reads an audio file and ensures it is exactly target_length samples.
    Returns:
        samples: (1, T) numpy array or List[(1, T)] for splitting silence.
        is_split: Boolean indicating if the file was split into multiple chunks.
    """
    try:
        # soundfile reads as (frames, channels) or (frames,)
        data, sr = sf.read(file_path, dtype="float32")

        # Ensure Mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Standardize length
        current_len = len(data)

        if current_len == target_length:
            return data.reshape(1, target_length), False

        elif current_len < target_length:
            # Pad with zeros
            pad_width = target_length - current_len
            # Right pad
            padded = np.pad(data, (0, pad_width), mode="constant")
            return padded.reshape(1, target_length), False

        else:
            # If significantly longer (e.g. background noise), we might want to split it
            # Heuristic: If it's a 'silence' file (checked by caller usually, but here we check length)
            # For this function, we'll return the center crop by default unless handled externally.
            # However, to handle the 'silence' expansion logic cleanly:

            # Center Crop
            start = (current_len - target_length) // 2
            cropped = data[start : start + target_length]
            return cropped.reshape(1, target_length), False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros((1, target_length), dtype="float32"), False


def load_subset(
    subset_name, csv_path, path_config, audio_config, load_cached_data=True
):
    """
    Loads audio data for a specific subset (train/val/test).
    Handles caching and 'silence' class expansion.
    """
    cache_dir = os.path.join(path_config.working_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    data_cache_path = os.path.join(cache_dir, f"{subset_name}_data.npy")
    targets_cache_path = os.path.join(cache_dir, f"{subset_name}_targets.npy")

    # 1. Try Load Cache
    if (
        load_cached_data
        and os.path.exists(data_cache_path)
        and os.path.exists(targets_cache_path)
    ):
        print(f"Loading {subset_name} data from cache...")
        try:
            waveforms = np.load(data_cache_path)
            targets = np.load(targets_cache_path)
            return waveforms, targets
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch
    print(f"Processing {subset_name} data from scratch...")
    df = pd.read_csv(csv_path)

    waveforms_list = []
    targets_list = []

    target_length = audio_config.n_samples

    for _, row in df.iterrows():
        filepath = os.path.join(path_config.input_root, row["filepath"])
        label_str = row["label"]
        label_idx = LABEL_TO_IDX.get(label_str, LABEL_TO_IDX["unknown"])

        # Special handling for 'silence' in Training set to balance data
        # Background noise files are long; we split them into 1s chunks
        if label_str == "silence" and subset_name == "train":
            try:
                data, _ = sf.read(filepath, dtype="float32")
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                num_samples = len(data)
                num_chunks = num_samples // target_length

                for i in range(num_chunks):
                    chunk = data[i * target_length : (i + 1) * target_length]
                    waveforms_list.append(chunk.reshape(1, target_length))
                    targets_list.append(label_idx)
            except Exception as e:
                print(f"Error reading silence file {filepath}: {e}")
        else:
            # Standard processing
            wav, _ = process_audio(filepath, target_length)
            waveforms_list.append(wav)
            targets_list.append(label_idx)

    # Stack into arrays
    if waveforms_list:
        waveforms = np.stack(waveforms_list).astype(np.float32)
        targets = np.array(targets_list, dtype=np.int64)
    else:
        # Fallback for empty dataset (should not happen)
        waveforms = np.zeros((0, 1, target_length), dtype=np.float32)
        targets = np.zeros((0,), dtype=np.int64)

    # 3. Save Cache
    print(f"Saving {subset_name} cache to {cache_dir}...")
    np.save(data_cache_path, waveforms)
    np.save(targets_cache_path, targets)

    return waveforms, targets


def get_dataloaders(
    path_config: PathConfig,
    audio_config: AudioConfig,
    train_config: TrainConfig,
    load_cached_data: bool = True,
):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements WeightedRandomSampler for training to handle class imbalance.
    """

    # --- Load Data ---
    train_x, train_y = load_subset(
        "train", path_config.train_csv, path_config, audio_config, load_cached_data
    )
    val_x, val_y = load_subset(
        "val", path_config.val_csv, path_config, audio_config, load_cached_data
    )
    test_x, test_y = load_subset(
        "test", path_config.test_csv, path_config, audio_config, load_cached_data
    )

    # --- Create Datasets ---
    train_dataset = SpeechCommandsDataset(train_x, train_y)
    val_dataset = SpeechCommandsDataset(val_x, val_y)
    test_dataset = SpeechCommandsDataset(test_x, test_y)

    # --- Weighted Sampler for Training ---
    # Calculate class weights: weight_i = 1 / count_i
    class_counts = np.bincount(train_y, minlength=len(LABEL_TO_IDX))

    # Avoid division by zero
    class_weights = 1.0 / (class_counts + 1e-6)

    # Assign a weight to each sample corresponding to its label
    sample_weights = class_weights[train_y]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_y), replacement=True
    )

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )

    print(
        f"Data Loaded: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})"
    )

    return train_loader, val_loader, test_loader
