import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn
from library.config import Config


def get_class_names():
    """
    Reads the sample_submission.csv to get the correct list of classes.
    """
    sample_sub_path = os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
    df = pd.read_csv(sample_sub_path)
    # The first column is 'fname', the rest are class labels
    return df.columns[1:].tolist()


def process_audio(filepath, mel_transform, db_transform):
    """
    Loads, resamples, pads/truncates, and converts audio to normalized Log-Mel Spectrogram.
    Returns a numpy array of shape (n_mels, time_steps).
    """
    full_path = os.path.join(Config.INPUT_ROOT, filepath)
    target_samples = Config.DURATION * Config.SAMPLE_RATE

    try:
        # Load audio
        wav, sr = torchaudio.load(full_path)
    except Exception as e:
        # Return a silent spectrogram in case of read error (though metadata should be clean)
        # Expected shape: (128, ~1876)
        return np.zeros((Config.N_MELS, 1876), dtype=np.float32)

    # Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
        wav = resampler(wav)

    # Convert to Mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Fix Duration (Pad or Truncate)
    current_samples = wav.shape[-1]
    if current_samples < target_samples:
        pad_amount = target_samples - current_samples
        wav = torch.nn.functional.pad(wav, (0, pad_amount))
    elif current_samples > target_samples:
        wav = wav[:, :target_samples]

    # Compute Spectrogram
    spec = mel_transform(wav)
    spec = db_transform(spec)

    # Instance Normalization
    # (x - mean) / (std + eps)
    mean = spec.mean()
    std = spec.std()
    spec = (spec - mean) / (std + 1e-6)

    # Remove channel dimension for storage (1, F, T) -> (F, T)
    return spec.squeeze(0).numpy()


def prepare_dataset(df, class_names, cache_prefix, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch and saves to cache.
    Returns X (spectrograms), y (labels), fnames.
    """
    cache_x_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_X.npy")
    cache_y_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_y.npy")
    cache_f_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_fnames.npy")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(cache_x_path)
        and os.path.exists(cache_y_path)
    ):
        print(f"Loading cached {cache_prefix} dataset...")
        try:
            X = np.load(cache_x_path)
            y = np.load(cache_y_path)
            fnames = np.load(cache_f_path)
            return X, y, fnames
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Processing {cache_prefix} dataset...")

    # Initialize Transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    # Pre-compute class mapping
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    num_classes = len(class_names)

    X_list = []
    y_list = []
    fnames_list = []

    for _, row in df.iterrows():
        # Process Audio
        spec = process_audio(row["filepath"], mel_transform, db_transform)
        X_list.append(spec)

        # Process Labels
        label_vec = np.zeros(num_classes, dtype=np.float32)
        if "labels" in row and pd.notna(row["labels"]):
            tags = row["labels"].split(",")
            for tag in tags:
                if tag in class_to_idx:
                    label_vec[class_to_idx[tag]] = 1.0
        y_list.append(label_vec)

        fnames_list.append(row["fname"])

    # Stack into arrays
    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    fnames = np.array(fnames_list)

    # 3. Save to Cache
    print(f"Saving {cache_prefix} dataset to cache...")
    np.save(cache_x_path, X)
    np.save(cache_y_path, y)
    np.save(cache_f_path, fnames)

    return X, y, fnames


class AudioDataset(Dataset):
    def __init__(self, X, y, fnames, transform=None):
        """
        Args:
            X (np.ndarray): Spectrograms (N, n_mels, time_steps)
            y (np.ndarray): Labels (N, num_classes)
            fnames (np.ndarray): Filenames
            transform (callable, optional): Augmentation transform
        """
        self.X = X
        self.y = y
        self.fnames = fnames
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve spectrogram
        spec = self.X[idx]  # Shape: (F, T)

        # Convert to tensor and add channel dimension: (1, F, T)
        spec = torch.from_numpy(spec).unsqueeze(0)

        # Apply augmentations (e.g., SpecAugment)
        if self.transform:
            spec = self.transform(spec)

        # Retrieve target
        target = torch.from_numpy(self.y[idx])

        fname = self.fnames[idx]

        return spec, target, fname


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # 2. Handle DEBUG mode
    if Config.DEBUG:
        print(f"DEBUG Mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Get Class Names
    class_names = get_class_names()

    # 4. Prepare Data Arrays (Cache or Process)
    X_train, y_train, f_train = prepare_dataset(
        df_train, class_names, "train", load_cached_data
    )
    X_val, y_val, f_val = prepare_dataset(df_val, class_names, "val", load_cached_data)
    X_test, y_test, f_test = prepare_dataset(
        df_test, class_names, "test", load_cached_data
    )

    # 5. Define Augmentations (SpecAugment for Training)
    train_transform = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK
        ),
        torchaudio.transforms.TimeMasking(time_mask_param=Config.SPEC_AUG_TIME_MASK),
    )

    # 6. Create Datasets
    train_dataset = AudioDataset(X_train, y_train, f_train, transform=train_transform)
    val_dataset = AudioDataset(X_val, y_val, f_val, transform=None)
    test_dataset = AudioDataset(X_test, y_test, f_test, transform=None)

    # 7. Create DataLoaders
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
