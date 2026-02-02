import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class AudioProcessor:
    """
    Handles loading raw audio and converting it to Log-Mel Spectrograms.
    """

    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            center=True,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80)

    def process_file(self, filepath):
        """
        Loads audio, fixes length, and computes log-mel spectrogram.
        Returns a numpy array of shape (C, F, T).
        """
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            # Fallback for corrupted files: return silent waveform
            waveform = torch.zeros(1, Config.FIXED_NUM_SAMPLES)
            sr = Config.SR

        # Resample if necessary (though dataset is uniform 2kHz)
        if sr != Config.SR:
            resampler = T.Resample(sr, Config.SR)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Fix length (Pad or Truncate)
        current_len = waveform.shape[1]
        target_len = Config.FIXED_NUM_SAMPLES

        if current_len < target_len:
            padding = target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > target_len:
            waveform = waveform[:, :target_len]

        # Compute Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)

        return log_mel_spec.numpy()


class SpecAugment:
    """
    Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
    Enforces the constraint of max time mask width = 200ms.
    """

    def __init__(self):
        self.freq_mask = T.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK_PARAM
        )
        # Constraint: Max time mask width 200ms (approx 20 frames)
        self.time_mask = T.TimeMasking(time_mask_param=Config.SPEC_AUG_TIME_MASK_PARAM)

    def __call__(self, spec):
        # spec is expected to be a Tensor of shape (C, F, T)
        aug_spec = spec
        for _ in range(Config.SPEC_AUG_NUM_FREQ_MASKS):
            aug_spec = self.freq_mask(aug_spec)
        for _ in range(Config.SPEC_AUG_NUM_TIME_MASKS):
            aug_spec = self.time_mask(aug_spec)
        return aug_spec


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    """

    def __init__(self, data, targets=None, ids=None, transform=None):
        """
        Args:
            data (np.ndarray): Spectrogram data (N, C, F, T)
            targets (np.ndarray, optional): Labels (N,)
            ids (np.ndarray, optional): Clip IDs (N,)
            transform (callable, optional): Augmentation transform
        """
        self.data = data
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert numpy to tensor
        spec = torch.tensor(self.data[idx], dtype=torch.float32)

        # Ensure channel dimension exists (C, F, T)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        # Apply augmentation if provided
        if self.transform:
            spec = self.transform(spec)

        # Normalize (simple min-max scaling or standardization could be added here if needed)
        # For this task, raw dB values are often sufficient, but we ensure float32.

        item = {"data": spec}

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["label"] = label

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def load_or_process_data(mode, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.

    Args:
        mode (str): 'train', 'val', or 'test'
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data, targets/ids)
    """
    Config.setup_dirs()
    processor = AudioProcessor()

    # Define paths based on mode
    if mode == "train":
        csv_path = Config.TRAIN_CSV
        data_cache = Config.TRAIN_DATA_CACHE
        label_cache = Config.TRAIN_LABELS_CACHE
        has_labels = True
    elif mode == "val":
        csv_path = Config.VAL_CSV
        data_cache = Config.VAL_DATA_CACHE
        label_cache = Config.VAL_LABELS_CACHE
        has_labels = True
    elif mode == "test":
        csv_path = Config.TEST_CSV
        data_cache = Config.TEST_DATA_CACHE
        label_cache = Config.TEST_IDS_CACHE  # Reusing variable for IDs
        has_labels = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(data_cache) and os.path.exists(label_cache):
        print(f"Loading {mode} data from cache...")
        data = np.load(data_cache)
        targets_or_ids = np.load(label_cache)
        return data, targets_or_ids

    # 2. Process from scratch
    print(f"Processing {mode} data from scratch...")
    df = pd.read_csv(csv_path)

    # Pre-allocate list
    data_list = []
    targets_or_ids_list = []

    for _, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_ROOT, row["filepath"])

        # Process audio
        spec = processor.process_file(full_path)
        data_list.append(spec)

        # Store label or ID
        if has_labels:
            targets_or_ids_list.append(row["label"])
        else:
            targets_or_ids_list.append(row["clip"])

    # Convert to numpy arrays
    data_arr = np.array(data_list, dtype=np.float32)

    if has_labels:
        targets_or_ids_arr = np.array(targets_or_ids_list, dtype=np.int64)
    else:
        targets_or_ids_arr = np.array(targets_or_ids_list, dtype=object)

    # 3. Save to cache
    print(f"Saving {mode} data to cache...")
    np.save(data_cache, data_arr)
    np.save(label_cache, targets_or_ids_arr)

    return data_arr, targets_or_ids_arr


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and returns DataLoaders for train, val, and test.
    """
    set_seed(Config.SEED)

    # Load Data
    train_data, train_labels = load_or_process_data("train", load_cached_data)
    val_data, val_labels = load_or_process_data("val", load_cached_data)
    test_data, test_ids = load_or_process_data("test", load_cached_data)

    # Define Transforms
    train_transform = SpecAugment()
    # No augmentation for val/test
    val_transform = None
    test_transform = None

    # Create Datasets
    train_dataset = WhaleDataset(
        train_data, targets=train_labels, transform=train_transform
    )
    val_dataset = WhaleDataset(val_data, targets=val_labels, transform=val_transform)
    test_dataset = WhaleDataset(test_data, ids=test_ids, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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
