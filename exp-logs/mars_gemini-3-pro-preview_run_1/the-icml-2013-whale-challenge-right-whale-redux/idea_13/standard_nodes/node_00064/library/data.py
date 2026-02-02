import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from library.config import Config
from library.utils import set_seed


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Handles accessing pre-processed spectrograms and applying SpecAugment during training.
    """

    def __init__(self, data, labels=None, training=False):
        """
        Args:
            data (np.ndarray): Array of shape (N, n_mels, time_steps).
            labels (np.ndarray, optional): Array of shape (N,). Defaults to None.
            training (bool): If True, applies SpecAugment.
        """
        self.data = data
        self.labels = labels
        self.training = training

        # Augmentations
        # Time mask: Max 200ms (approx 20 frames with hop=10ms)
        self.time_masking = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data: (n_mels, time_steps)
        spec = self.data[idx]

        # Convert to tensor and add channel dimension: (1, n_mels, time_steps)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)

        if self.training:
            # Apply SpecAugment
            # Note: Input to masking transforms must be (..., freq, time)
            # Our spec is (1, 128, 201)
            spec_tensor = self.freq_masking(spec_tensor)
            spec_tensor = self.time_masking(spec_tensor)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec_tensor, label
        else:
            # For test set, return dummy label or just the clip name (handled by index mapping outside)
            return spec_tensor, torch.tensor(0.0)


def preprocess_audio(filepath):
    """
    Loads audio, resamples, pads/crops, computes Log-Mel Spectrogram, and normalizes.

    Returns:
        np.ndarray: Processed spectrogram of shape (n_mels, time_steps).
    """
    try:
        waveform, sample_rate = torchaudio.load(filepath)
    except Exception as e:
        # Fallback for corrupted files (should not happen based on metadata check)
        # Create silent waveform
        waveform = torch.zeros(1, Config.N_SAMPLES)
        sample_rate = Config.SR

    # Resample if necessary
    if sample_rate != Config.SR:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=Config.SR)
        waveform = resampler(waveform)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Fix length (Pad or Crop)
    if waveform.shape[1] < Config.N_SAMPLES:
        padding = Config.N_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > Config.N_SAMPLES:
        waveform = waveform[:, : Config.N_SAMPLES]

    # Compute Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
        power=Config.POWER,
    )

    mel_spec = mel_transform(waveform)

    # Log-Mel
    log_mel_spec = torch.log(mel_spec + 1e-6)

    # Normalize
    log_mel_spec = (log_mel_spec - Config.NORM_MEAN) / Config.NORM_STD

    # Squeeze channel dim for storage (added back in Dataset)
    # Shape becomes (n_mels, time_steps)
    return log_mel_spec.squeeze(0).numpy()


def load_subset(subset_name, csv_path, load_cached_data=True, debug_size=None):
    """
    Loads data for a specific subset (train, val, test).
    Handles caching logic.
    """
    data_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_labels.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{subset_name}_ids.npy")

    # Check if cache exists
    cache_exists = os.path.exists(data_path) and os.path.exists(ids_path)
    if subset_name != "test":
        cache_exists = cache_exists and os.path.exists(labels_path)

    if load_cached_data and cache_exists:
        print(f"Loading {subset_name} data from cache...")
        data = np.load(data_path)
        ids = np.load(ids_path, allow_pickle=True)
        if subset_name != "test":
            labels = np.load(labels_path)
        else:
            labels = None

        # Handle debug size on cached data
        if debug_size is not None:
            data = data[:debug_size]
            ids = ids[:debug_size]
            if labels is not None:
                labels = labels[:debug_size]

        return data, labels, ids

    # If not cached or force reload
    print(f"Processing {subset_name} data from scratch...")
    df = pd.read_csv(csv_path)

    if debug_size is not None:
        df = df.iloc[:debug_size]

    data_list = []
    labels_list = []
    ids_list = []

    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])
        spec = preprocess_audio(full_path)
        data_list.append(spec)
        ids_list.append(row["clip"])

        if "label" in row:
            labels_list.append(row["label"])

    data_arr = np.array(data_list, dtype=np.float32)
    ids_arr = np.array(ids_list)

    # Save to cache
    np.save(data_path, data_arr)
    np.save(ids_path, ids_arr)

    if labels_list:
        labels_arr = np.array(labels_list, dtype=np.int64)
        np.save(labels_path, labels_arr)
    else:
        labels_arr = None

    return data_arr, labels_arr, ids_arr


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Main function to prepare DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.
        debug_size (int, optional): If set, limits dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """
    set_seed(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Load Train Data
    train_data, train_labels, _ = load_subset(
        "train", Config.TRAIN_CSV, load_cached_data, debug_size
    )

    # 2. Load Val Data
    val_data, val_labels, _ = load_subset(
        "val", Config.VAL_CSV, load_cached_data, debug_size
    )

    # 3. Load Test Data
    test_data, _, test_ids = load_subset(
        "test", Config.TEST_CSV, load_cached_data, debug_size
    )

    # 4. Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, training=True)
    val_dataset = WhaleDataset(val_data, val_labels, training=False)
    test_dataset = WhaleDataset(test_data, labels=None, training=False)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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
        f"Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader, test_ids
