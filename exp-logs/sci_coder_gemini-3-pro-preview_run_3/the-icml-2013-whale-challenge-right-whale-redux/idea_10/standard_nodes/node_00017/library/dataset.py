import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(Config.SEED)


class WhaleDataset(Dataset):
    def __init__(self, features, labels=None, mode="train"):
        """
        Args:
            features (np.ndarray): Array of spectrograms (N, C, F, T).
            labels (np.ndarray, optional): Array of labels (N,).
            mode (str): 'train', 'val', or 'test'. Controls augmentation.
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.mode = mode

        # Augmentations for training
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPECAUG_TIME_MASK
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPECAUG_FREQ_MASK
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Shape: (C, F, T)
        x = self.features[idx]

        # Apply SpecAugment only in training mode
        if self.mode == "train" and Config.USE_SPECAUG:
            # SpecAugment expects (..., F, T)
            # We clone to avoid modifying the cached tensor in memory
            x = x.clone()
            x = self.freq_masking(x)
            x = self.time_masking(x)

        if self.labels is not None:
            y = self.labels[idx]
            # Return label as float for BCE loss
            return x, y
        else:
            # Dummy label for test set
            return x, torch.zeros(1)


def process_audio_file(file_path, target_samples):
    """
    Reads audio, pads/crops to target length, returns tensor.
    """
    try:
        audio, sr = sf.read(file_path)
    except Exception as e:
        # Fallback for potentially corrupted files
        return torch.zeros(target_samples)

    # Ensure correct length
    if len(audio) < target_samples:
        # Pad with zeros
        padding = target_samples - len(audio)
        audio = np.pad(audio, (0, padding), "constant")
    elif len(audio) > target_samples:
        # Crop
        audio = audio[:target_samples]

    return torch.from_numpy(audio).float()


def min_max_norm(spectrogram):
    """
    Applies per-instance Min-Max normalization.
    Input: (C, F, T)
    Output: (C, F, T)
    Cite solution_lesson_node_00015
    """
    max_val = spectrogram.max()
    min_val = spectrogram.min()
    return (spectrogram - min_val) / (max_val - min_val + 1e-6)


def load_and_process_data(csv_path, split_name, load_cached_data=True):
    """
    Loads metadata, checks cache, processes audio if needed, returns features/labels.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {split_name} data from cache: {cache_path}")
        try:
            data = np.load(cache_path)
            features = data["features"]
            labels = data["labels"] if "labels" in data else None
            return features, labels
        except Exception as e:
            print(f"Failed to load cache ({e}). Re-processing...")

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Setup transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    target_samples = int(Config.DURATION * Config.SR)

    features_list = []
    labels_list = []

    # Iterate through all files
    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Load and fix length
        audio_tensor = process_audio_file(full_path, target_samples)

        # Compute Spectrogram
        # audio_tensor shape: (T,)
        spec = mel_transform(audio_tensor)  # -> (n_mels, time)

        # Log scale
        spec = amp_to_db(spec)

        # Add channel dim: (1, F, T)
        spec = spec.unsqueeze(0)

        # Normalization
        # Always apply Min-Max Norm as per Lesson 15, regardless of FREQ_NORM flag which is now False
        spec = min_max_norm(spec)

        features_list.append(spec.numpy())

        if "label" in row:
            labels_list.append(row["label"])

    features = np.stack(features_list)
    labels = np.array(labels_list) if labels_list else None

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    if labels is not None:
        np.savez(cache_path, features=features, labels=labels)
    else:
        np.savez(cache_path, features=features)

    print(f"Saved {split_name} data to cache.")
    return features, labels


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get Training and Validation DataLoaders.
    """
    # Load Data
    train_X, train_y = load_and_process_data(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_X, val_y = load_and_process_data(Config.VAL_CSV, "val", load_cached_data)

    # Create Datasets
    train_dataset = WhaleDataset(train_X, train_y, mode="train")
    val_dataset = WhaleDataset(val_X, val_y, mode="val")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup/BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Entry point to get Test DataLoader and clip names.
    """
    test_X, _ = load_and_process_data(Config.TEST_CSV, "test", load_cached_data)
    test_dataset = WhaleDataset(test_X, None, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Also return the clip names for submission mapping
    df_test = pd.read_csv(Config.TEST_CSV)
    clip_names = df_test["clip_name"].values

    return test_loader, clip_names
