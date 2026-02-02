import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import set_seed


class SpeechCommandsDataset(Dataset):
    """
    In-memory dataset for Speech Commands.
    Holds pre-processed spectrograms and applies SpecAugment during training.
    """

    def __init__(self, features, labels, is_train=False, transform=None):
        self.features = features
        self.labels = labels
        self.is_train = is_train
        self.transform = transform

        # SpecAugment transforms
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.FREQ_MASK_PARAM
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.TIME_MASK_PARAM
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features are already (C, F, T) -> (1, 128, 101)
        # We assume features are stored as numpy arrays or tensors
        feature = self.features[idx]
        label = self.labels[idx]

        # Convert to tensor if not already
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()

        # Apply SpecAugment if training
        if self.is_train:
            feature = self.freq_mask(feature)
            feature = self.time_mask(feature)

        # Ensure label is long tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return feature, label


def get_spectrogram_transform():
    """
    Creates the MelSpectrogram transform based on Config.
    """
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
    )


def preprocess_waveform(waveform, sr, mel_transform):
    """
    Converts raw waveform to normalized Log-Mel Spectrogram.
    """
    # 1. Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # 2. Adjust Length (Pad or Truncate)
    target_len = int(Config.SAMPLE_RATE * Config.DURATION)
    current_len = waveform.shape[1]

    if current_len > target_len:
        # Truncate (take the beginning for determinism)
        waveform = waveform[:, :target_len]
    elif current_len < target_len:
        # Pad with zeros
        padding = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # 3. Compute Mel Spectrogram
    # Input: (1, samples), Output: (1, n_mels, time)
    spec = mel_transform(waveform)

    # 4. Log Transform (Log-Mel)
    # Add epsilon for numerical stability
    spec = torch.log(spec + 1e-9)

    # 5. Instance Normalization
    # Standardize per sample: (x - mean) / std
    mean = spec.mean()
    std = spec.std()
    if std > 0:
        spec = (spec - mean) / std
    else:
        spec = spec - mean

    return spec


def process_and_cache_data(
    metadata_path, cache_prefix, load_cached_data=True, is_test=False
):
    """
    Loads metadata, processes audio into spectrograms, and caches results to disk.
    """
    features_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_features.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(features_path)
        and os.path.exists(labels_path)
    ):
        print(f"Loading cached data from {features_path}...")
        try:
            features = np.load(features_path)
            labels = np.load(labels_path)
            return features, labels
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Debug subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Processing subset of {len(df)} samples.")

    mel_transform = get_spectrogram_transform()

    features_list = []
    labels_list = []

    # Pre-map labels to IDs
    if not is_test:
        df["label_id"] = df["label"].map(Config.LABEL2ID)

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)

            # Process
            spec = preprocess_waveform(waveform, sr, mel_transform)

            # Store as numpy (float32) to save space/memory
            features_list.append(spec.numpy())

            if is_test:
                labels_list.append(-1)  # Placeholder
            else:
                labels_list.append(row["label_id"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Skip corrupted files
            continue

    # Stack into arrays
    features = np.stack(features_list).astype(np.float32)
    labels = np.array(labels_list).astype(np.int64)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(features_path, features)
    np.save(labels_path, labels)
    print(f"Saved processed data to {features_path}")

    return features, labels


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Val, and Test.
    """
    set_seed(Config.SEED)

    # ==========================================
    # 1. Prepare Training Data
    # ==========================================
    train_features, train_labels = process_and_cache_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Calculate Class Weights for Sampling
    # Count occurrences of each class index
    class_counts = np.bincount(train_labels, minlength=Config.NUM_CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Weights are inverse of frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[train_labels]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataset = SpeechCommandsDataset(train_features, train_labels, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Handles shuffling and balancing
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    # ==========================================
    # 2. Prepare Validation Data
    # ==========================================
    val_features, val_labels = process_and_cache_data(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=load_cached_data,
        is_test=False,
    )

    val_dataset = SpeechCommandsDataset(val_features, val_labels, is_train=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ==========================================
    # 3. Prepare Test Data
    # ==========================================
    test_features, test_labels = process_and_cache_data(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=load_cached_data,
        is_test=True,
    )

    test_dataset = SpeechCommandsDataset(test_features, test_labels, is_train=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"DataLoaders Ready:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val  : {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test : {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
