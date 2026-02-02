import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior for transforms where possible
set_seed(Config.SEED)


class SpecAugment:
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
    """

    def __init__(self, freq_mask_param, time_mask_param):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, spec):
        # spec: (C, H, W)
        return self.time_mask(self.freq_mask(spec))


class SpeechCommandDataset(Dataset):
    """
    PyTorch Dataset for Speech Commands.
    Holds pre-processed spectrogram features and labels in memory.
    """

    def __init__(self, features, labels, fnames=None, transform=None):
        """
        Args:
            features (Tensor): Tensor of shape (N, 1, H, W) containing spectrograms.
            labels (Tensor): Tensor of shape (N,) containing integer labels.
            fnames (list, optional): List of filenames corresponding to the samples.
            transform (callable, optional): Transform to apply to the spectrogram (e.g., SpecAugment).
        """
        self.features = features
        self.labels = labels
        self.fnames = fnames
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]  # (1, 224, 224)
        y = self.labels[idx]

        # Apply augmentations if provided (usually for training)
        if self.transform:
            x = self.transform(x)

        # Return filename if available (useful for test set tracking)
        if self.fnames is not None:
            return x, y, self.fnames[idx]

        return x, y


def load_audio_fixed_length(
    file_path, target_sr=Config.SAMPLE_RATE, max_samples=Config.MAX_SAMPLES
):
    """
    Loads audio, resamples if needed, and pads/trims to a fixed length.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        # Fallback for corrupted files: return silence
        print(f"Warning: Failed to load {file_path}. Returning silence. Error: {e}")
        return torch.zeros(1, max_samples)

    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Pad or Trim
    num_samples = waveform.shape[1]
    if num_samples < max_samples:
        padding = max_samples - num_samples
        # Pad at the end
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif num_samples > max_samples:
        # Trim (take the first max_samples)
        waveform = waveform[:, :max_samples]

    return waveform


def compute_spectrogram(waveform):
    """
    Computes Log-Mel Spectrogram, resizes to target image size, and applies Instance Normalization.
    """
    # 1. Generate Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )
    spec = mel_transform(waveform)

    # 2. Convert to Log Scale (AmplitudeToDB)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec)

    # 3. Instance Normalization
    # (x - mean) / (std + eps)
    mean = spec.mean()
    std = spec.std()
    spec = (spec - mean) / (std + 1e-6)

    return spec


def process_data_split(df, audio_root, label_map):
    """
    Iterates through the dataframe, loads audio, computes features, and returns tensors.
    """
    features_list = []
    labels_list = []
    fnames_list = []

    total_files = len(df)
    print(f"Processing {total_files} files...")

    for idx, row in df.iterrows():
        # Construct full path
        # Metadata file_path is relative to input root (e.g., train/audio/...)
        # But Config.INPUT_ROOT is "./input".
        # The metadata file_path already includes "train/audio" or "test/audio".
        # So we join INPUT_ROOT with the relative path.
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Load and process
        waveform = load_audio_fixed_length(full_path)
        spec = compute_spectrogram(waveform)  # Shape: (1, 224, 224)

        # Handle Label
        label_str = row["label"]
        label_idx = label_map.get(label_str, label_map["unknown"])

        features_list.append(spec)
        labels_list.append(label_idx)
        fnames_list.append(row["fname"])

        # Optional: Print progress every 5000 files
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{total_files}")

    # Stack into tensors
    features = torch.stack(features_list)  # (N, 1, 224, 224)
    labels = torch.tensor(labels_list, dtype=torch.long)  # (N,)

    return features, labels, fnames_list


def get_dataloaders(
    debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to prepare DataLoaders. Handles caching, splitting, and sampling.
    """
    set_seed(Config.SEED)

    # 1. Prepare Mappings
    label_map = {label: i for i, label in enumerate(Config.LABELS)}
    print(f"Label Map: {label_map}")

    # 2. Define Cache Paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    prefix = "debug_" if debug_subset_size is not None else ""

    cache_files = {
        "train": {
            "feat": os.path.join(cache_dir, f"{prefix}train_features.npy"),
            "lbl": os.path.join(cache_dir, f"{prefix}train_labels.npy"),
            "fname": os.path.join(cache_dir, f"{prefix}train_fnames.npy"),
        },
        "val": {
            "feat": os.path.join(cache_dir, f"{prefix}val_features.npy"),
            "lbl": os.path.join(cache_dir, f"{prefix}val_labels.npy"),
            "fname": os.path.join(cache_dir, f"{prefix}val_fnames.npy"),
        },
        "test": {
            "feat": os.path.join(cache_dir, f"{prefix}test_features.npy"),
            "lbl": os.path.join(cache_dir, f"{prefix}test_labels.npy"),
            "fname": os.path.join(cache_dir, f"{prefix}test_fnames.npy"),
        },
    }

    # 3. Load or Process Data
    datasets = {}

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Apply Debug Subset if requested
    if debug_subset_size is not None:
        print(f"DEBUG MODE: Reducing datasets to {debug_subset_size} samples.")
        df_train = df_train.iloc[:debug_subset_size]
        df_val = df_val.iloc[:debug_subset_size]
        df_test = df_test.iloc[:debug_subset_size]

    splits = [("train", df_train), ("val", df_val), ("test", df_test)]

    for split_name, df in splits:
        paths = cache_files[split_name]

        # Check if cache exists
        cache_exists = (
            os.path.exists(paths["feat"])
            and os.path.exists(paths["lbl"])
            and os.path.exists(paths["fname"])
        )

        if load_cached_data and cache_exists:
            print(f"Loading {split_name} data from cache...")
            features = torch.from_numpy(np.load(paths["feat"]))
            labels = torch.from_numpy(np.load(paths["lbl"]))
            fnames = np.load(paths["fname"]).tolist()
        else:
            print(f"Processing {split_name} data from scratch...")
            features, labels, fnames = process_data_split(
                df, Config.INPUT_ROOT, label_map
            )

            # Save to cache
            print(f"Saving {split_name} data to cache...")
            np.save(paths["feat"], features.numpy())
            np.save(paths["lbl"], labels.numpy())
            np.save(paths["fname"], np.array(fnames))

        # Create Dataset
        # Apply SpecAugment only for Train
        transform = None
        if split_name == "train":
            transform = SpecAugment(
                freq_mask_param=Config.FREQ_MASK_PARAM,
                time_mask_param=Config.TIME_MASK_PARAM,
            )

        datasets[split_name] = SpeechCommandDataset(
            features=features, labels=labels, fnames=fnames, transform=transform
        )

    # 4. Create DataLoaders

    # Train Loader with WeightedRandomSampler
    train_dataset = datasets["train"]
    train_labels = train_dataset.labels.numpy()

    # Compute class weights
    class_counts = np.bincount(train_labels, minlength=Config.NUM_CLASSES)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Val Loader (Shuffle=False)
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Test Loader (Shuffle=False)
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    print("DataLoaders created successfully.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader
