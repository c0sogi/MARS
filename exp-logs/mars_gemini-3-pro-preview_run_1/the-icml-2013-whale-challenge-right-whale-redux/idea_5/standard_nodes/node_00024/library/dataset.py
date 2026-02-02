import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T
from library.config import Config
from library.utils import set_seed

# Ensure deterministic behavior
set_seed(Config.SEED)


class WhaleDataset(Dataset):
    def __init__(self, data, labels, is_training=False):
        """
        Args:
            data (Tensor): Shape [N, 1, F, T]
            labels (Tensor): Shape [N]
            is_training (bool): Whether to apply augmentation
        """
        self.data = data
        self.labels = labels
        self.is_training = is_training

        # Augmentations
        # SpecAugment: Frequency Masking and Time Masking
        # Time mask parameter is strictly limited to 20 frames (approx 200ms)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM)
        self.time_mask = T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve pre-computed spectrogram
        spec = self.data[idx]  # [1, F, T]
        label = self.labels[idx]

        if self.is_training:
            # Apply SpecAugment during training
            # Input to these transforms should be [..., freq, time]
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        return spec, label


def load_audio_file(filepath):
    """
    Loads an audio file, resamples if needed, mixes to mono, and pads/crops to fixed duration.
    """
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        # Return silent waveform if load fails to prevent crash
        return torch.zeros(1, Config.NUM_SAMPLES)

    # Resample if necessary
    if sr != Config.SAMPLE_RATE:
        resampler = T.Resample(sr, Config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Ensure fixed length (Config.NUM_SAMPLES)
    current_len = waveform.shape[1]
    if current_len < Config.NUM_SAMPLES:
        # Pad with zeros at the end
        pad_amount = Config.NUM_SAMPLES - current_len
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
    elif current_len > Config.NUM_SAMPLES:
        # Center crop to preserve the middle of the clip (likely where the event is)
        start = (current_len - Config.NUM_SAMPLES) // 2
        waveform = waveform[:, start : start + Config.NUM_SAMPLES]

    return waveform


def process_and_cache_data(df, cache_name, load_cached_data=True):
    """
    Loads audio, converts to Log-Mel Spectrogram, and caches the result as .npy files.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    data_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_data.npy")
    label_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_labels.npy")

    # Check cache
    if load_cached_data and os.path.exists(data_path) and os.path.exists(label_path):
        try:
            data = np.load(data_path)
            labels = np.load(label_path)
            return torch.from_numpy(data), torch.from_numpy(labels)
        except Exception:
            pass  # Failed to load, recompute

    # Define Transforms
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.SAMPLE_RATE,
        n_fft=Config.N_FFT,
        win_length=Config.WIN_LENGTH,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.F_MIN,
        f_max=Config.F_MAX,
        center=True,
        pad_mode="reflect",
    )
    db_transform = T.AmplitudeToDB(top_db=80.0)

    data_list = []
    label_list = []

    # Iterate and Process
    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # Load and process waveform
        waveform = load_audio_file(full_path)

        # Convert to Spectrogram
        spec = mel_transform(waveform)
        spec = db_transform(spec)

        data_list.append(spec.numpy())

        # Handle Label
        if "label" in row:
            label_list.append(row["label"])
        else:
            # Dummy label for test set
            label_list.append(0.0)

    # Stack and Save
    data_array = np.stack(data_list).astype(np.float32)  # [N, 1, F, T]
    label_array = np.array(label_list).astype(np.float32)  # [N]

    np.save(data_path, data_array)
    np.save(label_path, label_array)

    return torch.from_numpy(data_array), torch.from_numpy(label_array)


def get_dataloaders(debug=Config.DEBUG, load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Handle Debug Mode
    suffix = ""
    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:100]
        test_df = test_df.iloc[:100]
        suffix = "_debug"

    # Process Data (Load from cache or compute)
    train_data, train_labels = process_and_cache_data(
        train_df, f"train{suffix}", load_cached_data
    )
    val_data, val_labels = process_and_cache_data(
        val_df, f"val{suffix}", load_cached_data
    )
    test_data, test_labels = process_and_cache_data(
        test_df, f"test{suffix}", load_cached_data
    )

    # Create Datasets
    train_ds = WhaleDataset(train_data, train_labels, is_training=True)
    val_ds = WhaleDataset(val_data, val_labels, is_training=False)
    test_ds = WhaleDataset(test_data, test_labels, is_training=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last to avoid issues with batch norm or mixup on small batches
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
