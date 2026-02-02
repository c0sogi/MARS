import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_ROOT,
    SAMPLE_RATE,
    NUM_SAMPLES,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    F_MIN,
    F_MAX,
    LABEL_TO_IDX,
    BATCH_SIZE,
    NUM_WORKERS,
    WORKING_DIR,
    DEVICE,
)
from library.utils import get_cached_numpy

# ==========================================
# 1. Helper Functions for Data Processing
# ==========================================


def _compute_waveforms(df, input_root):
    """
    Reads audio files, resamples, pads/truncates, and returns a numpy array of waveforms.
    """
    num_files = len(df)
    waveforms = np.zeros((num_files, NUM_SAMPLES), dtype=np.float32)

    for idx, row in df.iterrows():
        # Construct full path. Metadata paths are relative to input root.
        # Note: input_root is ./input, paths in df are like train/audio/...
        # So we join input_root with the relative path.
        full_path = os.path.join(input_root, row["file_path"])

        try:
            sig, sr = torchaudio.load(full_path)

            # Resample if necessary
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                sig = resampler(sig)

            # Mix to mono if necessary
            if sig.shape[0] > 1:
                sig = torch.mean(sig, dim=0, keepdim=True)

            # Squeeze channel dim
            sig = sig.squeeze(0)

            # Pad or Truncate
            sig_len = sig.shape[0]
            if sig_len < NUM_SAMPLES:
                # Pad with zeros
                padding = NUM_SAMPLES - sig_len
                sig = torch.nn.functional.pad(sig, (0, padding))
            elif sig_len > NUM_SAMPLES:
                # Truncate
                sig = sig[:NUM_SAMPLES]

            waveforms[idx] = sig.numpy()

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            # Leave as zeros (silence) in case of error

    return waveforms


def _compute_labels(df):
    """
    Converts string labels to integer indices.
    """
    labels = (
        df["label"]
        .map(LABEL_TO_IDX)
        .fillna(LABEL_TO_IDX["unknown"])
        .astype(np.int64)
        .values
    )
    return labels


def _compute_fnames(df):
    """
    Extracts filenames as a numpy array of strings.
    """
    return df["fname"].values.astype(str)


def load_data_split(metadata_path, split_name, load_cached_data=True):
    """
    Loads metadata, checks cache for processed arrays, or computes them.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Define cache filenames
    waveforms_file = f"{split_name}_waveforms.npy"
    labels_file = f"{split_name}_labels.npy"
    fnames_file = f"{split_name}_fnames.npy"

    # Load or compute waveforms
    waveforms = get_cached_numpy(
        waveforms_file,
        _compute_waveforms,
        load_cached_data,
        df=df,
        input_root=INPUT_ROOT,
    )

    # Load or compute labels
    labels = get_cached_numpy(labels_file, _compute_labels, load_cached_data, df=df)

    # Load or compute fnames
    fnames = get_cached_numpy(fnames_file, _compute_fnames, load_cached_data, df=df)

    return waveforms, labels, fnames


# ==========================================
# 2. Dataset Class
# ==========================================


class SpeechCommandDataset(Dataset):
    def __init__(self, waveforms, labels, fnames, is_train=False):
        """
        Args:
            waveforms (np.ndarray): Array of shape (N, NUM_SAMPLES).
            labels (np.ndarray): Array of shape (N,).
            fnames (np.ndarray): Array of shape (N,).
            is_train (bool): Whether to apply augmentation.
        """
        self.waveforms = torch.from_numpy(waveforms)
        self.labels = torch.from_numpy(labels).long()
        self.fnames = fnames
        self.is_train = is_train

        # Spectrogram Transform
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )

        # Amplitude to DB (Log-Mel)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Augmentations (SpecAugment)
        # Calibrated to <20% of dimensions
        # Time steps ~ 101, Freq bins = 128
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=25)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        fname = self.fnames[idx]

        # 1. Generate Log-Mel Spectrogram
        # waveform shape: (16000,) -> unsqueeze to (1, 16000) for transform if needed?
        # MelSpectrogram expects (..., time). It handles 1D fine.
        spec = self.melspec(waveform)  # Shape: (n_mels, time)
        spec = self.amplitude_to_db(spec)

        # 2. Instance Normalization
        # (spec - mean) / std
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 3. Augmentation (Train only)
        if self.is_train:
            # SpecAugment expects (channel, freq, time) or (freq, time)
            # Torchaudio transforms work on (..., freq, time)
            # We treat n_mels as freq.
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        # Ensure shape is (1, n_mels, time) for CNN input
        spec = spec.unsqueeze(0)

        return spec, label, fname


# ==========================================
# 3. DataLoader Factory
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Implements WeightedRandomSampler for the training set.
    """
    print("Preparing DataLoaders...")

    # 1. Load Data
    train_waves, train_labels, train_fnames = load_data_split(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_waves, val_labels, val_fnames = load_data_split(
        VAL_METADATA_PATH, "val", load_cached_data
    )
    test_waves, test_labels, test_fnames = load_data_split(
        TEST_METADATA_PATH, "test", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = SpeechCommandDataset(
        train_waves, train_labels, train_fnames, is_train=True
    )
    val_dataset = SpeechCommandDataset(
        val_waves, val_labels, val_fnames, is_train=False
    )
    test_dataset = SpeechCommandDataset(
        test_waves, test_labels, test_fnames, is_train=False
    )

    # 3. Handle Class Imbalance for Training
    # Calculate weights: 1 / count
    print("Computing class weights for WeightedRandomSampler...")
    class_counts = np.bincount(train_labels)
    # Avoid division by zero if a class is missing (unlikely given EDA)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Assign a weight to each sample
    sample_weights = class_weights[train_labels]
    sample_weights = torch.from_numpy(sample_weights).float()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(
        f"DataLoaders ready. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches."
    )
    return train_loader, val_loader, test_loader
