import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T

from library.config import Config
from library.utils import get_weighted_sampler


class SpeechCommandDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        """
        Args:
            features (Tensor): Tensor of shape (N, 1, n_mels, time)
            labels (Tensor): Tensor of shape (N,)
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. SpecAugment).
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # features are already (1, n_mels, time)
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram + AmplitudeToDB pipeline.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )

    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def process_audio_file(file_path, transform_pipeline):
    """
    Loads audio, pads/crops, computes spectrogram, normalizes.
    Returns Tensor of shape (1, n_mels, time).
    """
    # Load audio
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception:
        # Fallback for corrupted files: return silent waveform
        waveform = torch.zeros(1, Config.NUM_SAMPLES)
        sr = Config.SAMPLE_RATE

    # Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Crop to fixed duration
    num_samples = waveform.shape[1]
    target_samples = Config.NUM_SAMPLES

    if num_samples < target_samples:
        # Pad with zeros
        padding = target_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif num_samples > target_samples:
        # Crop
        waveform = waveform[:, :target_samples]

    # Compute Spectrogram
    # transform_pipeline expects (channel, time) input
    spec = transform_pipeline(waveform)

    # Instance Normalization: (x - mean) / std
    mean = spec.mean()
    std = spec.std()
    if std > 1e-6:
        spec = (spec - mean) / std
    else:
        spec = spec - mean

    return spec


def get_data_cache(df, split_name, load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    feats_path = os.path.join(cache_dir, f"{split_name}_features.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(feats_path) and os.path.exists(labels_path):
        print(f"Loading {split_name} data from cache...")
        features = np.load(feats_path)
        labels = np.load(labels_path)
        return torch.from_numpy(features), torch.from_numpy(labels)

    print(f"Processing {split_name} data from scratch ({len(df)} samples)...")

    # Prepare transform pipeline
    transform_pipeline = get_spectrogram_transform()

    features_list = []
    labels_list = []

    # Iterate through dataframe
    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Process audio
        spec = process_audio_file(full_path, transform_pipeline)
        features_list.append(spec.numpy())

        # Process label
        label_str = row["label"]
        # Default to 'unknown' if label not found (safety check)
        label_id = Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])
        labels_list.append(label_id)

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    # Stack into arrays
    features = np.stack(features_list)  # (N, 1, n_mels, time)
    labels = np.array(labels_list)  # (N,)

    # Save to cache
    np.save(feats_path, features)
    np.save(labels_path, labels)
    print(f"Saved {split_name} cache to {cache_dir}")

    return torch.from_numpy(features), torch.from_numpy(labels)


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to get DataLoaders.

    Args:
        load_cached_data (bool): Whether to try loading from .npy cache.
        debug (bool): If True, loads a small subset of data for testing.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("Debug mode: reducing dataset size.")
        df_train = df_train.iloc[:200]
        df_val = df_val.iloc[:100]
        df_test = df_test.iloc[:100]

    # 2. Get Data (Cached or Processed)
    # Use distinct cache names for debug to avoid overwriting full cache
    train_split_name = "train_debug" if debug else "train"
    val_split_name = "val_debug" if debug else "val"
    test_split_name = "test_debug" if debug else "test"

    X_train, y_train = get_data_cache(df_train, train_split_name, load_cached_data)
    X_val, y_val = get_data_cache(df_val, val_split_name, load_cached_data)
    X_test, y_test = get_data_cache(df_test, test_split_name, load_cached_data)

    # 3. Define Augmentations for Training
    train_transform = torch.nn.Sequential(
        T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
        T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
    )

    # 4. Create Datasets
    train_dataset = SpeechCommandDataset(X_train, y_train, transform=train_transform)
    val_dataset = SpeechCommandDataset(X_val, y_val, transform=None)
    test_dataset = SpeechCommandDataset(X_test, y_test, transform=None)

    # 5. Create Sampler for Training
    # We use the original dataframe to calculate weights.
    # The order of rows in df_train matches the order in X_train.
    train_sampler = get_weighted_sampler(df_train, label_col="label")

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
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
