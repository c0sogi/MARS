import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_spectrogram_transform():
    """
    Constructs the MelSpectrogram and AmplitudeToDB transform pipeline.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        power=2.0,
    )
    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def preprocess_audio(filepath):
    """
    Reads an audio file, ensures 2.0s duration, and converts to a normalized Log-Mel Spectrogram.

    Returns:
        np.ndarray: Shape (1, n_mels, time_frames)
    """
    full_path = os.path.join(Config.INPUT_ROOT, filepath)

    try:
        waveform, sr = torchaudio.load(full_path)
    except Exception:
        # Fallback for read errors: create silent waveform
        target_len = int(Config.SR * Config.DURATION)
        waveform = torch.zeros(1, target_len)
        sr = Config.SR

    # Resample if necessary
    if sr != Config.SR:
        resampler = T.Resample(sr, Config.SR)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure fixed length of 2.0 seconds
    target_len = int(Config.SR * Config.DURATION)
    current_len = waveform.shape[1]

    if current_len < target_len:
        padding = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # Compute Spectrogram
    transform = get_spectrogram_transform()
    spec = transform(waveform)  # Shape: (1, n_mels, time)

    # Instance-wise Normalization (Standardization)
    mean = spec.mean()
    std = spec.std()
    if std > 0:
        spec = (spec - mean) / (std + 1e-6)
    else:
        spec = spec - mean

    return spec.numpy()


def process_and_cache_data(df, cache_name, load_cached_data=True):
    """
    Loads processed data from disk or computes it from scratch and saves it.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    data_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(data_path) and os.path.exists(ids_path):
        # For labeled data, check labels file too
        if "label" in df.columns:
            if os.path.exists(labels_path):
                print(f"Loading {cache_name} data from cache...")
                data = np.load(data_path)
                labels = np.load(labels_path)
                ids = np.load(ids_path)
                return data, labels, ids
        else:
            # For test data (no labels)
            print(f"Loading {cache_name} data from cache...")
            data = np.load(data_path)
            ids = np.load(ids_path)
            return data, None, ids

    # Process from scratch
    print(f"Processing {cache_name} data from scratch...")

    data_list = []
    labels_list = []
    ids_list = []

    # Handle Debug mode
    if Config.DEBUG:
        df = df.iloc[: Config.DEBUG_SAMPLES]

    for _, row in df.iterrows():
        spec = preprocess_audio(row["filepath"])
        data_list.append(spec)
        ids_list.append(row["clip"])

        if "label" in row:
            labels_list.append(row["label"])

    data = np.stack(data_list)  # Shape: (N, 1, F, T)
    ids = np.array(ids_list)

    np.save(data_path, data)
    np.save(ids_path, ids)

    if labels_list:
        labels = np.array(labels_list, dtype=np.float32)
        np.save(labels_path, labels)
    else:
        labels = None

    return data, labels, ids


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, clip_ids=None, transform=None):
        """
        PyTorch Dataset for Whale Calls.
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.clip_ids = clip_ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spec = self.data[idx]  # Shape: (1, F, T)

        # Apply augmentation (SpecAugment)
        if self.transform:
            spec = self.transform(spec)

        if self.labels is not None:
            label = self.labels[idx]
            return spec, label
        else:
            clip_id = self.clip_ids[idx]
            return spec, clip_id


def get_transforms(mode="train"):
    """
    Returns the SpecAugment transform for training.
    """
    if mode == "train":
        return torch.nn.Sequential(
            T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
            T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
        )
    return None


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Performs Mixup augmentation on a batch.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates data loading, caching, and DataLoader creation.
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Process or Load Data
    train_data, train_labels, train_ids = process_and_cache_data(
        df_train, "train", load_cached_data
    )
    val_data, val_labels, val_ids = process_and_cache_data(
        df_val, "val", load_cached_data
    )
    test_data, _, test_ids = process_and_cache_data(df_test, "test", load_cached_data)

    # Initialize Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, train_ids, transform=get_transforms("train")
    )

    val_dataset = WhaleDataset(val_data, val_labels, val_ids, transform=None)

    test_dataset = WhaleDataset(test_data, None, test_ids, transform=None)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
