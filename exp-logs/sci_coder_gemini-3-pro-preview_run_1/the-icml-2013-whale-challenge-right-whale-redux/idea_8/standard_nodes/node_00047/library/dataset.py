import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior for transforms where possible
set_seed(Config.SEED)


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram + AmplitudeToDB transform pipeline.
    """
    return torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=80),
    )


def process_audio_file(filepath, transform, target_samples):
    """
    Loads an audio file, pads/crops it to target_samples, and converts to spectrogram.
    """
    try:
        waveform, sr = torchaudio.load(filepath)

        # Ensure correct sample rate (though analysis showed all are 2000Hz)
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Ensure Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Crop to fixed duration
        current_samples = waveform.shape[1]
        if current_samples < target_samples:
            pad_amount = target_samples - current_samples
            # Pad at the end
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif current_samples > target_samples:
            # Crop from the beginning (or center, but start is safer for consistency)
            waveform = waveform[:, :target_samples]

        # Convert to Spectrogram
        spec = transform(waveform)
        return spec.squeeze(0).numpy()  # Return as numpy array (C, F, T) -> (F, T)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        # Return a silent spectrogram of correct shape as fallback
        # Shape: n_mels, time_steps
        # time_steps = target_samples // hop_length + 1
        n_steps = (target_samples // Config.HOP_LENGTH) + 1
        return np.zeros((Config.N_MELS, n_steps), dtype=np.float32)


def load_or_process_data(df, split_name, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    """
    # Define cache paths
    suffix = "_debug" if Config.DEBUG else ""
    cache_data_path = os.path.join(Config.WORK_DIR, f"{split_name}{suffix}_data.npy")
    cache_labels_path = os.path.join(
        Config.WORK_DIR, f"{split_name}{suffix}_labels.npy"
    )
    cache_ids_path = os.path.join(Config.WORK_DIR, f"{split_name}{suffix}_ids.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_data_path):
        print(f"Loading cached {split_name} data from {cache_data_path}...")
        data = np.load(cache_data_path)

        # Load labels if they exist (Test set might not have labels in some flows, but here we handle it)
        if os.path.exists(cache_labels_path):
            labels = np.load(cache_labels_path)
        else:
            labels = None

        # Load IDs
        if os.path.exists(cache_ids_path):
            ids = np.load(cache_ids_path, allow_pickle=True)
        else:
            ids = df["clip"].values

        return data, labels, ids

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch...")

    # Apply Debug limit
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLES).copy()
        print(f"DEBUG Mode: Process limited to {len(df)} samples.")

    transform = get_spectrogram_transform()
    target_samples = int(Config.SAMPLE_RATE * Config.DURATION)

    data_list = []
    labels_list = []
    ids_list = df["clip"].values

    # Pre-calculate full paths
    # Metadata filepath is relative to input root
    full_paths = [os.path.join(Config.INPUT_ROOT, p) for p in df["filepath"].values]

    # Extract labels if present
    has_labels = "label" in df.columns
    if has_labels:
        raw_labels = df["label"].values

    for i, path in enumerate(full_paths):
        spec = process_audio_file(path, transform, target_samples)
        data_list.append(spec)
        if has_labels:
            labels_list.append(raw_labels[i])

    # Convert to numpy arrays
    data = np.stack(data_list).astype(np.float32)  # Shape: (N, F, T)

    if has_labels:
        labels = np.array(labels_list, dtype=np.float32)
    else:
        labels = None

    # 3. Save to cache
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    np.save(cache_data_path, data)
    np.save(cache_ids_path, ids_list)
    if labels is not None:
        np.save(cache_labels_path, labels)

    print(f"Saved processed {split_name} data to {Config.WORK_DIR}")

    return data, labels, ids_list


class WhaleDataset(Dataset):
    def __init__(self, data, labels, ids, mean=None, std=None, transform=None):
        """
        Args:
            data (np.ndarray): Spectrogram data (N, F, T).
            labels (np.ndarray): Labels (N,). None for test set.
            ids (np.ndarray): Clip IDs.
            mean (float): Global mean for normalization.
            std (float): Global std for normalization.
            transform (callable): Augmentation transform.
        """
        self.data = data
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data: (F, T)
        spec = self.data[idx]

        # Convert to tensor: (1, F, T)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)

        # Apply Normalization
        if self.mean is not None and self.std is not None:
            spec_tensor = (spec_tensor - self.mean) / (self.std + 1e-6)

        # Apply Augmentation (only if transform is provided)
        if self.transform:
            spec_tensor = self.transform(spec_tensor)

        # Prepare Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return spec_tensor, label, self.ids[idx]
        else:
            return spec_tensor, self.ids[idx]


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Process/Load Data
    train_data, train_labels, train_ids = load_or_process_data(
        df_train, "train", load_cached_data
    )
    val_data, val_labels, val_ids = load_or_process_data(
        df_val, "val", load_cached_data
    )
    test_data, test_labels, test_ids = load_or_process_data(
        df_test, "test", load_cached_data
    )

    # 3. Compute Statistics for Normalization (from Training Data)
    # We compute this every time as it's fast on loaded arrays
    global_mean = np.mean(train_data)
    global_std = np.std(train_data)
    print(
        f"Global Normalization Stats - Mean: {global_mean:.4f}, Std: {global_std:.4f}"
    )

    # 4. Define Augmentations (SpecAugment) for Training
    # Constraint: Time Mask max 200ms. Config.TIME_MASK_PARAM is set to 12 (approx 192ms)
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
        torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
    )

    # 5. Create Datasets
    train_dataset = WhaleDataset(
        train_data,
        train_labels,
        train_ids,
        mean=global_mean,
        std=global_std,
        transform=train_transform,
    )

    val_dataset = WhaleDataset(
        val_data, val_labels, val_ids, mean=global_mean, std=global_std, transform=None
    )

    test_dataset = WhaleDataset(
        test_data, None, test_ids, mean=global_mean, std=global_std, transform=None
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup stability
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
