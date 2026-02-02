import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Call Detection.
    Handles Instance Standardization and Augmentation on pre-computed Mel Spectrograms.
    """

    def __init__(self, data, targets, clip_names=None, transform=None, mode="train"):
        """
        Args:
            data (np.ndarray): Array of Mel Spectrograms (N, n_mels, time).
            targets (np.ndarray): Array of labels (N,). None for test.
            clip_names (np.ndarray): Array of filenames (N,). Only for test.
            transform (nn.Module): Augmentation transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.data = data
        self.targets = targets
        self.clip_names = clip_names
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve pre-computed Mel Spectrogram
        # Shape: (n_mels, time)
        spec = self.data[idx]

        # Convert to Tensor and add channel dimension: (1, n_mels, time)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)

        # --- Instance Standardization ---
        # Zero-Mean, Unit-Variance applied per clip
        mean = spec_tensor.mean()
        std = spec_tensor.std()
        spec_tensor = (spec_tensor - mean) / (std + 1e-6)

        # --- Augmentation ---
        # Applied only during training
        if self.transform:
            spec_tensor = self.transform(spec_tensor)

        if self.mode == "test":
            return spec_tensor, self.clip_names[idx]
        else:
            # Return float target for BCEWithLogitsLoss
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return spec_tensor, target


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    """
    if mode == "train" and Config.spec_augment:
        return torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.freq_mask_param
            ),
            torchaudio.transforms.TimeMasking(time_mask_param=Config.time_mask_param),
        )
    return None


def load_audio_and_process(file_path):
    """
    Reads audio file, enforces duration, and computes Mel Spectrogram.
    Returns: numpy array of shape (n_mels, time)
    """
    full_path = os.path.join(Config.input_root, file_path)

    # 1. Load Audio
    try:
        wav, sr = sf.read(full_path)
    except Exception:
        # Fallback for read errors (safety net)
        sr = Config.sample_rate
        wav = np.zeros(int(Config.duration * sr))

    # Handle Multi-channel (Convert to Mono)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    # 2. Pad or Crop to Fixed Duration
    target_samples = int(Config.duration * Config.sample_rate)
    current_samples = len(wav)

    if current_samples < target_samples:
        pad_width = target_samples - current_samples
        wav = np.pad(wav, (0, pad_width), mode="constant")
    elif current_samples > target_samples:
        wav = wav[:target_samples]

    # 3. Compute Mel Spectrogram
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # (1, time)

    # Note: We instantiate transforms here. For massive scale, moving this out is better,
    # but for this dataset size, it's negligible compared to I/O.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.sample_rate,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length,
        n_mels=Config.n_mels,
        f_min=Config.fmin,
        f_max=Config.fmax,
        normalized=False,  # Explicitly False to preserve environmental noise characteristics
        center=True,
    )

    spec = mel_transform(wav_tensor)

    # 4. Amplitude to DB
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.top_db)
    spec_db = db_transform(spec)

    # Remove channel dim for storage: (1, n_mels, time) -> (n_mels, time)
    return spec_db.squeeze(0).numpy()


def process_dataset(csv_path, cache_base_path, load_cached_data=True, is_test=False):
    """
    Orchestrates data loading, processing, and caching.
    """
    # Define cache filenames
    data_cache = cache_base_path.replace(".npy", "_data.npy")
    targets_cache = cache_base_path.replace(".npy", "_targets.npy")
    clips_cache = cache_base_path.replace(".npy", "_clips.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(data_cache):
        data = np.load(data_cache)
        if is_test:
            if os.path.exists(clips_cache):
                clips = np.load(clips_cache, allow_pickle=True)
                return data, None, clips
        else:
            if os.path.exists(targets_cache):
                targets = np.load(targets_cache)
                return data, targets, None

    # 2. Process from Scratch
    df = pd.read_csv(csv_path)

    if Config.debug:
        df = df.head(Config.debug_sample_size)

    data_list = []
    targets_list = []
    clips_list = []

    for _, row in df.iterrows():
        spec = load_audio_and_process(row["file_path"])
        data_list.append(spec)

        if is_test:
            clips_list.append(row["clip"])
        else:
            targets_list.append(row["label"])

    # Stack into arrays
    data_arr = np.stack(data_list)  # (N, n_mels, time)

    # 3. Save to Cache
    os.makedirs(os.path.dirname(data_cache), exist_ok=True)
    np.save(data_cache, data_arr)

    if is_test:
        clips_arr = np.array(clips_list)
        np.save(clips_cache, clips_arr)
        return data_arr, None, clips_arr
    else:
        targets_arr = np.array(targets_list)
        np.save(targets_cache, targets_arr)
        return data_arr, targets_arr, None


def get_dataloaders(load_cached_data=True):
    """
    Returns train and validation DataLoaders.
    """
    # --- Train ---
    train_data, train_targets, _ = process_dataset(
        Config.train_csv,
        Config.train_cache_file,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    train_dataset = WhaleDataset(
        train_data, train_targets, transform=get_transforms(mode="train"), mode="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Validation ---
    val_data, val_targets, _ = process_dataset(
        Config.val_csv,
        Config.val_cache_file,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    val_dataset = WhaleDataset(val_data, val_targets, transform=None, mode="val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns test DataLoader.
    """
    test_data, _, test_clips = process_dataset(
        Config.test_csv,
        Config.test_cache_file,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    test_dataset = WhaleDataset(
        test_data, None, clip_names=test_clips, transform=None, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader
