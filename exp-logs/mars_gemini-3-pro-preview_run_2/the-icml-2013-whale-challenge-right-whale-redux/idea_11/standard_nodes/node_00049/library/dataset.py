import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class WhaleDataset(Dataset):
    """
    Dataset class for Right Whale Detection.
    Handles waveform loading (from memory), Mel Spectrogram generation,
    Instance Standardization, and SpecAugment.
    """

    def __init__(self, data, labels, is_training=False):
        """
        Args:
            data (np.ndarray): Array of waveforms.
            labels (np.ndarray): Array of labels (or None).
            is_training (bool): Whether to apply augmentation.
        """
        self.data = data
        self.labels = labels
        self.is_training = is_training

        # ==========================================
        # Audio Transforms
        # ==========================================
        # 1. Mel Spectrogram
        # Note: normalized=False to preserve environmental noise characteristics (Pink noise)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=False,
        )

        # 2. Amplitude to DB (Log Scale)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

        # 3. Augmentations (SpecAugment)
        # Only applied during training
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.MASK_TIME_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.MASK_FREQ_PARAM
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Get Waveform
        # shape: (Time,)
        waveform = self.data[idx]

        # Convert to Tensor
        waveform = torch.from_numpy(waveform).float()

        # 2. Generate Mel Spectrogram
        # Input: (Time,) -> Output: (n_mels, Time)
        spec = self.mel_transform(waveform)

        # 3. Log Scale
        spec = self.db_transform(spec)

        # 4. Instance Standardization
        # Zero-Mean, Unit-Variance per clip
        mean = spec.mean()
        std = spec.std()
        # Add epsilon to prevent division by zero
        spec = (spec - mean) / (std + 1e-6)

        # 5. Augmentation (Training Only)
        if self.is_training and Config.USE_SPECAUG:
            # SpecAugment
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 6. Add Channel Dimension
        # Model expects (1, Freq, Time)
        spec = spec.unsqueeze(0)

        # 7. Prepare Label
        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        return spec, label


def load_dataset_data(csv_path, cache_prefix, load_cached_data=True):
    """
    Loads audio data from disk or cache.

    Args:
        csv_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data, labels, clips)
            data: np.ndarray of waveforms
            labels: np.ndarray of labels (or None)
            clips: np.ndarray of clip filenames (for test set)
    """
    # Ensure cache directory exists
    cache_dir = os.path.join(Config.WORKING_DIR, Config.PROJECT_NAME)
    os.makedirs(cache_dir, exist_ok=True)

    data_cache_path = os.path.join(cache_dir, f"{cache_prefix}_data.npy")
    labels_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    clips_cache_path = os.path.join(cache_dir, f"{cache_prefix}_clips.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(data_cache_path):
        print(f"Loading {cache_prefix} data from cache: {data_cache_path}")
        data = np.load(data_cache_path)

        labels = None
        if os.path.exists(labels_cache_path):
            labels = np.load(labels_cache_path)

        clips = None
        if os.path.exists(clips_cache_path):
            clips = np.load(clips_cache_path)

        return data, labels, clips

    # 2. Process from Source
    print(f"Processing {cache_prefix} data from source CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Handle Debug Mode
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SUBSET_SIZE)
        print(f"DEBUG MODE: Reduced {cache_prefix} dataset to {len(df)} samples.")

    data_list = []
    labels_list = []
    clips_list = []

    # Target length for padding/cropping (2.0s * 2000Hz = 4000 samples)
    target_length = int(Config.SAMPLE_RATE * 2.0)

    for _, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        try:
            # Load audio
            y, sr = sf.read(file_path)

            # Ensure fixed length
            if len(y) < target_length:
                padding = target_length - len(y)
                y = np.pad(y, (0, padding), "constant")
            elif len(y) > target_length:
                y = y[:target_length]

            data_list.append(y)

            # Handle Label
            if "label" in row:
                labels_list.append(row["label"])
            else:
                labels_list.append(-1)  # Placeholder

            # Handle Clip Name (for submission)
            if "clip" in row:
                clips_list.append(row["clip"])

        except Exception as e:
            print(f"Warning: Failed to load {file_path}. Error: {e}")
            continue

    # Convert to Numpy Arrays
    data = np.array(data_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32) if labels_list else None
    clips = np.array(clips_list) if clips_list else None

    # 3. Save to Cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(data_cache_path, data)
    if labels is not None:
        np.save(labels_cache_path, labels)
    if clips is not None:
        np.save(clips_cache_path, clips)

    return data, labels, clips


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_clips)
    """
    # 1. Load Data
    train_data, train_labels, _ = load_dataset_data(
        Config.TRAIN_CSV, "train", load_cached_data
    )
    val_data, val_labels, _ = load_dataset_data(Config.VAL_CSV, "val", load_cached_data)
    test_data, _, test_clips = load_dataset_data(
        Config.TEST_CSV, "test", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = WhaleDataset(train_data, train_labels, is_training=True)
    val_dataset = WhaleDataset(val_data, val_labels, is_training=False)
    test_dataset = WhaleDataset(test_data, None, is_training=False)

    # 3. Create Weighted Sampler for Training
    # Calculate weights to balance the classes
    # train_labels are 0 or 1
    class_counts = np.bincount(train_labels.astype(int))

    # Avoid division by zero if a class is missing (e.g. in small debug subset)
    if len(class_counts) < 2:
        class_weights = [1.0, 1.0]
    else:
        class_weights = 1.0 / class_counts

    sample_weights = [class_weights[int(l)] for l in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BatchNorm stability
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

    return train_loader, val_loader, test_loader, test_clips
