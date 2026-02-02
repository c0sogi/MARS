import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Serves pre-computed Log-Mel Spectrograms and applies on-the-fly augmentation.
    """

    def __init__(self, data, labels=None, transform=None):
        """
        Args:
            data (np.ndarray): Input data of shape (N, 1, F, T).
            labels (np.ndarray, optional): Labels of shape (N,).
            transform (nn.Module, optional): Augmentation transforms.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        # Data is expected to be (1, F, T)
        x = torch.from_numpy(self.data[idx]).float()

        # Apply transforms (e.g., SpecAugment) if provided
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        else:
            # Return dummy label for test set
            return x, torch.zeros(1)


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline based on the mode.
    """
    if mode == "train":
        # SpecAugment with constraints defined in Config
        return torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
            T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
        )
    else:
        return None


def process_audio_file(filepath):
    """
    Loads an audio file and converts it to a Log-Mel Spectrogram.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        np.ndarray: Log-Mel Spectrogram of shape (1, n_mels, time_steps).
    """
    try:
        waveform, sample_rate = torchaudio.load(filepath)
    except Exception as e:
        # Fallback for corrupted files: return silent waveform
        # This shouldn't happen given the data checks, but good for robustness
        print(f"Error loading {filepath}: {e}")
        waveform = torch.zeros(1, Config.NUM_SAMPLES)
        sample_rate = Config.SAMPLE_RATE

    # Resample if necessary (though dataset is 2kHz)
    if sample_rate != Config.SAMPLE_RATE:
        resampler = T.Resample(sample_rate, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Fix Length (Pad or Truncate)
    if waveform.shape[1] < Config.NUM_SAMPLES:
        padding = Config.NUM_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif waveform.shape[1] > Config.NUM_SAMPLES:
        waveform = waveform[:, : Config.NUM_SAMPLES]

    # Generate Mel Spectrogram
    # n_fft=1024 ensures high frequency resolution
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
    )

    mel_spec = mel_spec_transform(waveform)

    # Convert to Log Scale (dB)
    # standard epsilon is handled by AmplitudeToDB or manually
    # AmplitudeToDB expects power spectrogram
    db_transform = T.AmplitudeToDB(top_db=80.0)
    log_mel_spec = db_transform(mel_spec)

    return log_mel_spec.numpy()


def load_or_create_cache(df, name, debug=False, load_cached_data=True):
    """
    Loads data from .npy cache if available, otherwise processes audio files
    and creates the cache.
    """
    suffix = "_debug" if debug else ""
    data_filename = f"{name}{suffix}_data.npy"
    labels_filename = f"{name}{suffix}_labels.npy"
    ids_filename = f"{name}{suffix}_ids.npy"

    data_path = os.path.join(Config.WORKING_DIR, data_filename)
    labels_path = os.path.join(Config.WORKING_DIR, labels_filename)
    ids_path = os.path.join(Config.WORKING_DIR, ids_filename)

    has_labels = "label" in df.columns

    # Check if cache exists
    cache_exists = os.path.exists(data_path) and os.path.exists(ids_path)
    if has_labels:
        cache_exists = cache_exists and os.path.exists(labels_path)

    # 1. Try to load from cache
    if load_cached_data and cache_exists:
        print(f"Loading {name} set from cache...")
        try:
            data = np.load(data_path)
            # Verify consistency
            if len(data) == len(df):
                if has_labels:
                    labels = np.load(labels_path)
                    return data, labels
                else:
                    return data, None
            else:
                print(f"Cache size mismatch for {name}. Recomputing...")
        except Exception as e:
            print(f"Error loading cache for {name}: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {name} set audio files...")
    data_list = []
    label_list = []
    id_list = []

    for idx, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])
        spec = process_audio_file(full_path)
        data_list.append(spec)
        id_list.append(row["clip"])

        if has_labels:
            label_list.append(row["label"])

    # Stack into arrays
    data_array = np.stack(data_list)
    ids_array = np.array(id_list)

    # Save to cache
    np.save(data_path, data_array)
    np.save(ids_path, ids_array)

    if has_labels:
        labels_array = np.array(label_list, dtype=np.float32)
        np.save(labels_path, labels_array)
        return data_array, labels_array

    return data_array, None


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print("DEBUG MODE: Using subset of data.")
        df_train = df_train.head(100)
        df_val = df_val.head(50)
        df_test = df_test.head(50)

    # Process/Load Data
    train_data, train_labels = load_or_create_cache(
        df_train, "train", debug, load_cached_data
    )
    val_data, val_labels = load_or_create_cache(df_val, "val", debug, load_cached_data)
    test_data, _ = load_or_create_cache(df_test, "test", debug, load_cached_data)

    print(
        f"Data Shapes: Train {train_data.shape}, Val {val_data.shape}, Test {test_data.shape}"
    )

    # Create Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, transform=get_transforms(mode="train")
    )

    val_dataset = WhaleDataset(
        val_data, val_labels, transform=get_transforms(mode="val")
    )

    test_dataset = WhaleDataset(
        test_data, labels=None, transform=get_transforms(mode="test")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Useful for Mixup/BatchNorm stability
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
