import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T
from library.config import Config

# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)


class WhaleDataset(Dataset):
    """
    Custom Dataset for Right Whale Detection.
    Expects pre-processed Log-Mel Spectrograms.
    """

    def __init__(self, data, labels, transform=None, mean=0.0, std=1.0):
        """
        Args:
            data (np.ndarray): Input data of shape (N, 1, F, T).
            labels (np.ndarray): Labels of shape (N,) or None for test set.
            transform (callable, optional): Augmentation transforms.
            mean (float): Dataset mean for normalization.
            std (float): Dataset std for normalization.
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is (1, F, T)
        x = self.data[idx]

        # Apply Augmentations (e.g., SpecAugment)
        if self.transform:
            x = self.transform(x)

        # Normalize
        # Avoid division by zero with a small epsilon
        x = (x - self.mean) / (self.std + 1e-6)

        if self.labels is not None:
            return x, self.labels[idx]
        else:
            return x


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline.
    """
    if mode == "train":
        return torch.nn.Sequential(
            T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
            T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
        )
    else:
        return torch.nn.Identity()


def get_spectrogram_transform():
    """
    Returns the transformation pipeline to convert waveform to Log-Mel Spectrogram.
    """
    return torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
        ),
        T.AmplitudeToDB(),
    )


def load_and_process_audio(filepath, target_samples, transform_fn):
    """
    Loads audio, pads/truncates, and converts to spectrogram.
    """
    try:
        waveform, sr = torchaudio.load(filepath)

        # Resample if necessary (though dataset is 2kHz)
        if sr != Config.SAMPLE_RATE:
            resampler = T.Resample(sr, Config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Truncate to fixed duration
        current_samples = waveform.shape[1]
        if current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_samples > target_samples:
            waveform = waveform[:, :target_samples]

        # Convert to Spectrogram
        spec = transform_fn(waveform)
        return spec.numpy()

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        # Return a zero tensor of expected shape to maintain batch integrity
        # Shape: (1, n_mels, time_steps)
        # Time steps approx target_samples // hop_length + 1
        n_frames = target_samples // Config.HOP_LENGTH + 1
        return np.zeros((1, Config.N_MELS, n_frames), dtype=np.float32)


def process_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Loads audio from metadata, converts to spectrograms, and caches to .npy files.
    """
    data_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_data.npy")
    labels_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(data_path):
        # For test set, labels might not exist
        if "test" in cache_prefix or os.path.exists(labels_path):
            print(f"Loading cached data for {cache_prefix} from {data_path}...")
            data = np.load(data_path)
            labels = np.load(labels_path) if os.path.exists(labels_path) else None
            return data, labels

    print(f"Processing and caching data for {cache_prefix}...")

    transform_fn = get_spectrogram_transform()
    target_samples = int(Config.DURATION * Config.SAMPLE_RATE)

    data_list = []
    labels_list = []

    for _, row in df.iterrows():
        # Metadata filepath is relative to input root
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])

        spec = load_and_process_audio(full_path, target_samples, transform_fn)
        data_list.append(spec)

        if "label" in row:
            labels_list.append(row["label"])

    # Stack into a single numpy array
    data_arr = np.stack(data_list)  # Shape: (N, 1, F, T)
    np.save(data_path, data_arr)

    labels_arr = None
    if labels_list:
        labels_arr = np.array(labels_list, dtype=np.float32)
        np.save(labels_path, labels_arr)

    return data_arr, labels_arr


def get_dataloaders(load_cached_data=True, debug=Config.DEBUG):
    """
    Main function to prepare DataLoaders.
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Debugging: Subset data
    if debug:
        print(f"DEBUG mode: Using {Config.DEBUG_SUBSET_SIZE} samples per split.")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # 2. Process and Cache Data
    train_data, train_labels = process_and_cache_data(
        df_train, "train", load_cached_data
    )
    val_data, val_labels = process_and_cache_data(df_val, "val", load_cached_data)
    test_data, _ = process_and_cache_data(df_test, "test", load_cached_data)

    # 3. Compute Normalization Statistics (from Train set only)
    stats_path = os.path.join(Config.WORKING_DIR, "train_stats.npy")
    if load_cached_data and os.path.exists(stats_path):
        stats = np.load(stats_path)
        mean, std = stats[0], stats[1]
    else:
        mean = np.mean(train_data)
        std = np.std(train_data)
        np.save(stats_path, np.array([mean, std]))

    print(f"Dataset Stats - Mean: {mean:.6f}, Std: {std:.6f}")

    # 4. Create Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, transform=get_transforms("train"), mean=mean, std=std
    )

    val_dataset = WhaleDataset(
        val_data, val_labels, transform=get_transforms("val"), mean=mean, std=std
    )

    test_dataset = WhaleDataset(
        test_data, None, transform=get_transforms("test"), mean=mean, std=std
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
