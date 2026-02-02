import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config

# Ensure deterministic behavior
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class WhaleDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        """
        Args:
            data (np.ndarray): Input data of shape (N, 1, F, T).
            labels (np.ndarray, optional): Labels of shape (N,).
            transform (callable, optional): Augmentations to apply.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data and convert to tensor
        x = self.data[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Ensure shape is (1, F, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Apply augmentations (SpecAugment)
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            y = torch.tensor(y, dtype=torch.float32)
            return x, y
        else:
            return x


class AudioPreprocessor:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=Config.NORMALIZED_MEL,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        self.target_length = int(Config.SR * Config.DURATION)

    def process_file(self, file_path):
        try:
            # Load audio using soundfile
            audio, sr = sf.read(file_path)
            audio = torch.from_numpy(audio).float()

            # Handle channels: convert to mono
            if audio.dim() > 1:
                if audio.shape[1] > 1:
                    audio = torch.mean(audio, dim=1)
                else:
                    audio = audio.squeeze()

            # Resample if necessary
            if sr != Config.SR:
                resampler = torchaudio.transforms.Resample(sr, Config.SR)
                audio = resampler(audio)

            # Pad or Truncate to fixed duration
            current_len = audio.shape[0]
            if current_len < self.target_length:
                pad_len = self.target_length - current_len
                audio = torch.nn.functional.pad(audio, (0, pad_len))
            elif current_len > self.target_length:
                audio = audio[: self.target_length]

            # Generate Log-Mel Spectrogram
            # Input to mel_transform should be (1, time) or (time) -> it handles it, but unsqueeze is safer
            spec = self.mel_transform(audio.unsqueeze(0))
            spec = self.db_transform(spec)

            # Instance-wise standardization (Cite solution_lesson_node_00025)
            # Normalize each spectrogram to zero mean and unit variance
            mean = spec.mean()
            std = spec.std()
            if std > 1e-6:
                spec = (spec - mean) / std
            else:
                spec = spec - mean

            # Result shape is (1, n_mels, time_frames)
            return spec

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return silent spectrogram
            n_frames = 1 + (self.target_length // Config.HOP_LENGTH)
            return torch.zeros((1, Config.N_MELS, n_frames))


def load_and_cache_data(csv_path, cache_prefix, load_cached_data=True, debug=False):
    """
    Loads data from CSV, processes it, and caches to .npy files.
    """
    # Determine cache filenames
    suffix = "_debug" if debug else ""
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_data{suffix}.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels{suffix}.npy")
    clips_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_clips{suffix}.npy")

    # Try to load from cache
    if load_cached_data and os.path.exists(data_path):
        print(f"Loading cached data from {data_path}...")
        data = np.load(data_path)
        labels = np.load(labels_path) if os.path.exists(labels_path) else None
        return data, labels

    # Process from scratch
    print(f"Processing data from {csv_path} (Debug={debug})...")
    df = pd.read_csv(csv_path)

    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE)

    preprocessor = AudioPreprocessor()
    data_list = []
    labels_list = []
    clips_list = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        spec = preprocessor.process_file(file_path)
        data_list.append(spec.numpy())

        if "label" in row:
            labels_list.append(row["label"])
        if "clip" in row:
            clips_list.append(row["clip"])

    # Stack and save
    data = np.stack(data_list)  # Shape (N, 1, F, T)
    np.save(data_path, data)

    labels = None
    if labels_list:
        labels = np.array(labels_list, dtype=np.int64)
        np.save(labels_path, labels)

    if clips_list:
        clips = np.array(clips_list)
        np.save(clips_path, clips)

    print(f"Processed {len(data)} samples and saved to {Config.WORKING_DIR}")
    return data, labels


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Creates Train and Validation DataLoaders.
    """
    # Load Data
    train_x, train_y = load_and_cache_data(
        Config.TRAIN_CSV, "train", load_cached_data, debug
    )
    val_x, val_y = load_and_cache_data(Config.VAL_CSV, "val", load_cached_data, debug)

    # Define Transforms (SpecAugment)
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=10),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    )

    # Create Datasets
    train_dataset = WhaleDataset(train_x, train_y, transform=train_transform)
    val_dataset = WhaleDataset(val_x, val_y, transform=None)

    # Weighted Random Sampler for Class Imbalance
    class_counts = np.bincount(train_y)
    if len(class_counts) < 2:
        # Fallback for debug if only one class exists
        sample_weights = np.ones(len(train_y))
    else:
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_y]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_y), replacement=True
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
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

    return train_loader, val_loader


def get_test_loader(load_cached_data=True, debug=Config.DEBUG):
    """
    Creates Test DataLoader and returns clip names.
    """
    test_x, _ = load_and_cache_data(Config.TEST_CSV, "test", load_cached_data, debug)

    # Load clip names
    suffix = "_debug" if debug else ""
    clips_path = os.path.join(Config.WORKING_DIR, f"test_clips{suffix}.npy")

    if os.path.exists(clips_path):
        test_clips = np.load(clips_path)
    else:
        # Fallback if clips weren't cached (shouldn't happen if load_and_cache_data ran)
        df = pd.read_csv(Config.TEST_CSV)
        if debug:
            df = df.head(Config.DEBUG_SUBSET_SIZE)
        test_clips = df["clip"].values

    test_dataset = WhaleDataset(test_x, labels=None, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_clips
