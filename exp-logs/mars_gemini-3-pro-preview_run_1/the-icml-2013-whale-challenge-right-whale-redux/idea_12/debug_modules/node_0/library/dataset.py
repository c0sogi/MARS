import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T

from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(Config.SEED)


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram + AmplitudeToDB transform pipeline.
    """
    return torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            normalized=True,
        ),
        T.AmplitudeToDB(top_db=80.0),
    )


def preprocess_audio(filepath, transform_pipeline):
    """
    Loads audio, pads/truncates to fixed length, and computes spectrogram.
    """
    full_path = os.path.join(Config.INPUT_ROOT, filepath)

    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception as e:
        # Fallback for corrupted files: return silent waveform
        waveform = torch.zeros(1, Config.NUM_SAMPLES)
        sr = Config.SAMPLE_RATE

    # Resample if necessary (though analysis showed consistent 2000Hz)
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Fix length to Config.NUM_SAMPLES
    current_len = waveform.shape[1]
    if current_len < Config.NUM_SAMPLES:
        pad_amount = Config.NUM_SAMPLES - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_len > Config.NUM_SAMPLES:
        waveform = waveform[:, : Config.NUM_SAMPLES]

    # Compute Spectrogram
    # Output shape: (1, n_mels, time_steps)
    spec = transform_pipeline(waveform)

    # Squeeze channel dim for storage (will add back in Dataset) -> (n_mels, time_steps)
    return spec.squeeze(0).numpy()


def process_and_cache_subset(csv_path, subset_name, load_cached_data):
    """
    Handles caching logic for a specific subset (train, val, test).
    """
    data_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_labels.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(data_path)
        and os.path.exists(ids_path)
        and (subset_name == "test" or os.path.exists(labels_path))
    )

    if load_cached_data and cache_exists:
        print(f"Loading {subset_name} data from cache...")
        data = np.load(data_path)
        ids = np.load(ids_path)
        labels = np.load(labels_path) if subset_name != "test" else None
        return data, labels, ids

    print(f"Processing {subset_name} data from scratch...")
    df = pd.read_csv(csv_path)

    # Debugging: subset if configured
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    transform = get_spectrogram_transform()

    data_list = []
    labels_list = []
    ids_list = []

    for _, row in df.iterrows():
        spec = preprocess_audio(row["filepath"], transform)
        data_list.append(spec)
        ids_list.append(row["clip"])
        if "label" in row:
            labels_list.append(row["label"])

    # Convert to numpy arrays
    data_arr = np.array(data_list, dtype=np.float32)
    ids_arr = np.array(ids_list)

    # Save to cache
    np.save(data_path, data_arr)
    np.save(ids_path, ids_arr)

    if labels_list:
        labels_arr = np.array(labels_list, dtype=np.float32)
        np.save(labels_path, labels_arr)
        return data_arr, labels_arr, ids_arr
    else:
        return data_arr, None, ids_arr


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, ids=None, augment=False):
        """
        Args:
            data (np.array): Shape (N, n_mels, time_steps)
            labels (np.array): Shape (N,)
            ids (np.array): Shape (N,)
            augment (bool): Whether to apply SpecAugment
        """
        self.data = data
        self.labels = labels
        self.ids = ids
        self.augment = augment

        # Augmentations
        # Time Masking: Limit to ~200ms.
        self.time_masking = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load spectrogram: (n_mels, time_steps)
        spec = torch.from_numpy(self.data[idx])

        # Add channel dimension: (1, n_mels, time_steps)
        spec = spec.unsqueeze(0)

        # Apply Augmentation
        if self.augment:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # Instance-wise Normalization
        # Standardize: (x - mean) / (std + eps)
        mean = spec.mean()
        std = spec.std()
        if std > 0:
            spec = (spec - mean) / (std + 1e-6)
        else:
            spec = spec - mean

        # Prepare return values
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec, label
        else:
            # For test set, return clip ID
            clip_id = self.ids[idx]
            return spec, clip_id


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for train, val, and test sets.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Process/Load Data
    train_X, train_y, _ = process_and_cache_subset(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_X, val_y, _ = process_and_cache_subset(Config.VAL_CSV, "val", load_cached_data)
    test_X, _, test_ids = process_and_cache_subset(
        Config.TEST_CSV, "test", load_cached_data
    )

    # 2. Create Datasets
    # Train: Augmentation enabled
    train_dataset = WhaleDataset(train_X, train_y, augment=True)
    # Val: No augmentation
    val_dataset = WhaleDataset(val_X, val_y, augment=False)
    # Test: No augmentation, returns IDs
    test_dataset = WhaleDataset(test_X, ids=test_ids, augment=False)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability with Mixup/BatchNorm
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
