import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility across the module
set_seed(Config.SEED)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Wraps pre-processed spectrograms and applies augmentations.
    """

    def __init__(self, data, labels, transform=None, is_test=False):
        """
        Args:
            data (np.ndarray): Array of spectrograms (N, 1, F, T).
            labels (np.ndarray): Array of labels (N,) or clip names for test.
            transform (callable, optional): Transform to be applied on a sample.
            is_test (bool): Whether this is a test dataset (returns clip name instead of label).
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data instance
        # Data is stored as (1, F, T) float32 numpy array
        spec = torch.from_numpy(self.data[idx])

        # Apply transforms (e.g., SpecAugment)
        if self.transform:
            spec = self.transform(spec)

        if self.is_test:
            # For test, labels are clip names
            return spec, self.labels[idx]
        else:
            # For train/val, labels are targets
            # Return label as float for BCEWithLogitsLoss
            return spec, torch.tensor(self.labels[idx], dtype=torch.float32)


def get_transforms(train=True):
    """
    Returns the transformation pipeline.
    """
    if train:
        # SpecAugment for training: Frequency and Time Masking
        return torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )
    return None


def preprocess_waveform(waveform, sr):
    """
    Converts waveform to Mel Spectrogram with specific parameters and standardization.
    """
    # 1. Fixed Length Padding/Truncation to 2.0s
    target_len = int(Config.DURATION * Config.SR)
    current_len = waveform.shape[1]

    if current_len < target_len:
        pad_amount = target_len - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_len > target_len:
        waveform = waveform[:, :target_len]

    # 2. Generate Mel Spectrogram
    # Using high-resolution parameters from Config
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        normalized=Config.MEL_NORMALIZED,
    )

    spec = mel_transform(waveform)

    # 3. Log Amplitude
    spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)

    # 4. Instance Standardization (Zero Mean, Unit Variance)
    mean = spec.mean()
    std = spec.std()

    # Avoid division by zero
    if std > 1e-6:
        spec = (spec - mean) / std
    else:
        spec = spec - mean

    return spec


def process_files(df, root_dir):
    """
    Iterates through a dataframe, loads audio, and processes it into spectrograms.
    """
    data_list = []
    label_list = []

    print(f"Processing {len(df)} files...")

    for idx, row in df.iterrows():
        file_path = os.path.join(root_dir, row["file_path"])

        try:
            # Load audio using soundfile
            wav, sr = sf.read(file_path)
            wav_tensor = torch.tensor(wav, dtype=torch.float32)

            # Ensure channel dimension (1, samples)
            if wav_tensor.ndim == 1:
                wav_tensor = wav_tensor.unsqueeze(0)

            # Preprocess
            spec = preprocess_waveform(wav_tensor, sr)  # Returns (1, 128, 63)

            data_list.append(spec.numpy())

            if "label" in row:
                label_list.append(row["label"])
            elif "clip" in row:
                label_list.append(row["clip"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Fallback: create silent spectrogram to maintain alignment
            target_len = int(Config.DURATION * Config.SR)
            dummy_wav = torch.zeros(1, target_len)
            spec = preprocess_waveform(dummy_wav, Config.SR)
            data_list.append(spec.numpy())

            if "label" in row:
                label_list.append(0)  # Assume noise
            elif "clip" in row:
                label_list.append(row["clip"])

    return np.stack(data_list), np.array(label_list)


def load_data(load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    """
    # Define cache paths
    train_data_path = os.path.join(Config.WORKING_DIR, "train_data.npy")
    train_labels_path = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    test_data_path = os.path.join(Config.WORKING_DIR, "test_data.npy")
    test_clips_path = os.path.join(Config.WORKING_DIR, "test_clips.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists and we want to load it
    cache_exists = (
        os.path.exists(train_data_path)
        and os.path.exists(train_labels_path)
        and os.path.exists(test_data_path)
        and os.path.exists(test_clips_path)
    )

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_data = np.load(train_data_path)
        train_labels = np.load(train_labels_path)
        test_data = np.load(test_data_path)
        test_clips = np.load(test_clips_path)
        return train_data, train_labels, test_data, test_clips

    print("Cache not found or forced reload. Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine Train and Val for Stratified K-Fold
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: limiting to {Config.DEBUG_SUBSET_SIZE} samples.")
        full_train_df = full_train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # Process Data
    print("Processing Training Data...")
    train_data, train_labels = process_files(full_train_df, Config.INPUT_ROOT)

    print("Processing Test Data...")
    test_data, test_clips = process_files(test_df, Config.INPUT_ROOT)

    # Save to Cache
    print("Saving data to cache...")
    np.save(train_data_path, train_data)
    np.save(train_labels_path, train_labels)
    np.save(test_data_path, test_data)
    np.save(test_clips_path, test_clips)

    return train_data, train_labels, test_data, test_clips


def get_train_val_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation loaders for a specific fold using Stratified K-Fold.
    """
    # Load all labeled data
    train_data, train_labels, _, _ = load_data(load_cached_data)

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Generate splits
    splits = list(skf.split(train_data, train_labels))

    if fold_idx >= Config.N_FOLDS:
        raise ValueError(f"Fold index {fold_idx} out of range (0-{Config.N_FOLDS-1})")

    train_idx, val_idx = splits[fold_idx]

    # Subset data
    X_train = train_data[train_idx]
    y_train = train_labels[train_idx]
    X_val = train_data[val_idx]
    y_val = train_labels[val_idx]

    # Create Datasets
    train_dataset = WhaleDataset(X_train, y_train, transform=get_transforms(train=True))
    val_dataset = WhaleDataset(X_val, y_val, transform=get_transforms(train=False))

    # Weighted Random Sampler to handle class imbalance
    class_counts = np.bincount(y_train.astype(int))
    # Handle edge case for debug mode with missing classes
    if len(class_counts) < 2:
        class_weights = np.ones(2)
    else:
        class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
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


def get_test_loader(load_cached_data=True):
    """
    Returns the test loader.
    """
    _, _, test_data, test_clips = load_data(load_cached_data)

    test_dataset = WhaleDataset(
        test_data, test_clips, transform=get_transforms(train=False), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
