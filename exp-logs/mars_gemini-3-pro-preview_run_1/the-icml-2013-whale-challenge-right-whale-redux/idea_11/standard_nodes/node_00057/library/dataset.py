import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_and_process_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads audio data, processes it into Log-Mel Spectrograms, and caches the result.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    data_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_data.npy")
    labels_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_labels.npy")
    ids_path = os.path.join(Config.CACHE_DIR, f"{cache_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(data_path) and os.path.exists(ids_path):
        try:
            data = np.load(data_path)
            ids = np.load(ids_path, allow_pickle=True)
            labels = None
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
            return data, labels, ids
        except Exception as e:
            print(f"Cache load failed for {cache_name}, reprocessing. Error: {e}")

    # Process from scratch
    print(f"Processing {cache_name} data from scratch...")
    df = pd.read_csv(metadata_path)
    n_samples = len(df)

    # Initialize Transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    # Determine output shape using a dummy sample
    dummy_waveform = torch.zeros(1, int(Config.SAMPLE_RATE * Config.DURATION))
    dummy_spec = mel_transform(dummy_waveform)
    _, n_freq, n_time = dummy_spec.shape

    # Pre-allocate arrays
    data_arr = np.zeros((n_samples, n_freq, n_time), dtype=np.float32)
    labels_arr = (
        np.zeros(n_samples, dtype=np.float32) if "label" in df.columns else None
    )
    ids_arr = df["clip"].values

    for i, row in df.iterrows():
        filepath = os.path.join(Config.INPUT_ROOT, row["filepath"])
        try:
            # Load audio
            waveform, sr = torchaudio.load(filepath)

            # Resample if necessary
            if sr != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Convert to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Pad or Trim to fixed duration
            target_len = int(Config.SAMPLE_RATE * Config.DURATION)
            current_len = waveform.shape[1]
            if current_len < target_len:
                pad_amt = target_len - current_len
                waveform = F.pad(waveform, (0, pad_amt))
            elif current_len > target_len:
                waveform = waveform[:, :target_len]

            # Compute Spectrogram
            spec = mel_transform(waveform)
            spec = amp_to_db(spec)

            data_arr[i] = spec.squeeze(0).numpy()

            if labels_arr is not None:
                labels_arr[i] = row["label"]

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # Keep zero-initialized placeholder on error

    # Save to cache
    np.save(data_path, data_arr)
    np.save(ids_path, ids_arr)
    if labels_arr is not None:
        np.save(labels_path, labels_arr)

    return data_arr, labels_arr, ids_arr


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, is_train=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        self.is_train = is_train

        # SpecAugment: Frequency and Time Masking
        # Time mask param 20 corresponds to approx 200ms (20 frames * 10ms/frame)
        self.spec_aug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=Config.FREQ_MASK_PARAM
            ),
            torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Add channel dimension: (F, T) -> (1, F, T)
        spec = self.data[idx].unsqueeze(0)

        if self.is_train:
            spec = self.spec_aug(spec)

        if self.labels is not None:
            return spec, self.labels[idx]
        else:
            # Return dummy label for test set
            return spec, torch.tensor(0.0)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # 1. Train Set
    train_data, train_labels, _ = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "train.csv"), "train", load_cached_data
    )
    train_dataset = WhaleDataset(train_data, train_labels, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop last to ensure consistent batch sizes for Mixup
    )

    # 2. Validation Set
    val_data, val_labels, _ = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "val.csv"), "val", load_cached_data
    )
    val_dataset = WhaleDataset(val_data, val_labels, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Test Set
    test_data, _, test_ids = load_and_process_data(
        os.path.join(Config.METADATA_DIR, "test.csv"), "test", load_cached_data
    )
    test_dataset = WhaleDataset(test_data, labels=None, is_train=False)
    # Attach IDs to the dataset object for easy retrieval during inference
    test_dataset.ids = test_ids
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
