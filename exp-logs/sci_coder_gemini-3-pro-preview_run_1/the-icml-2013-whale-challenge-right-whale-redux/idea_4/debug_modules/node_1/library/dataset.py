import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from library.config import (
    INPUT_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
    SAMPLE_RATE,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    TIME_MASK_PARAM,
    FREQ_MASK_PARAM,
    MIXUP_ALPHA,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG,
    DEBUG_SUBSET_SIZE,
)

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_transforms(mode="train"):
    """
    Returns the SpecAugment transforms for training.
    """
    if mode == "train":
        return torch.nn.Sequential(
            T.TimeMasking(time_mask_param=TIME_MASK_PARAM),
            T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM),
        )
    else:
        return torch.nn.Identity()


def mixup_data(x, y, alpha=MIXUP_ALPHA, device="cpu"):
    """
    Applies Mixup to the batch.
    Returns:
        mixed_x: The mixed input tensor.
        y_a: The labels of the first set of samples.
        y_b: The labels of the second set of samples.
        lam: The mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def compute_spectrogram(filepath):
    """
    Loads an audio file and converts it to a Log-Mel Spectrogram.
    """
    try:
        waveform, sr = torchaudio.load(filepath)

        # Resample if necessary (though dataset is consistent)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad if shorter than expected (rare but safe)
        # We expect roughly 2 seconds.
        # If significantly shorter, we might pad.
        # For this dataset, clips are generally uniform or we handle variable length via resizing/pooling later.
        # However, EfficientNet expects fixed size or we rely on Global Pooling.
        # Here we just compute the spec; the backbone handles dimensions via adaptive pooling or we crop/pad.
        # Given the task description, clips are short. We'll compute spec as is.

        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        melspec = mel_transform(waveform)

        # Convert to Log-Mel
        log_melspec = T.AmplitudeToDB(top_db=80)(melspec)

        return log_melspec.squeeze(0).numpy()  # Shape: (n_mels, time)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        # Return a zero array of expected shape as fallback
        # Approx time frames: 2s * 2000 / 20 = 100 frames
        return np.zeros((N_MELS, 100), dtype=np.float32)


def load_and_cache_data(split, df, load_cached_data=True):
    """
    Loads data from cache or computes it from scratch and saves it.
    """
    data_path = os.path.join(WORKING_DIR, f"{split}_data.npy")
    labels_path = os.path.join(WORKING_DIR, f"{split}_labels.npy")
    ids_path = os.path.join(WORKING_DIR, f"{split}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(data_path) and os.path.exists(ids_path):
        # Check labels existence only for labeled sets
        if "label" not in df.columns or os.path.exists(labels_path):
            print(f"Loading {split} data from cache...")
            data = np.load(data_path)
            ids = np.load(ids_path)
            labels = np.load(labels_path) if "label" in df.columns else None
            return data, labels, ids

    # 2. Compute from scratch
    print(f"Processing {split} data from scratch...")
    data_list = []
    label_list = []
    id_list = []

    for idx, row in df.iterrows():
        full_path = os.path.join(INPUT_ROOT, row["filepath"])
        spec = compute_spectrogram(full_path)

        # Ensure consistent time dimension if necessary, or keep variable.
        # For batching, we usually need fixed size.
        # Let's pad/crop to a fixed width of 100 frames (2 seconds) for simplicity and consistency.
        # 2000 samples/sec * 2 sec / 20 hop = 200 samples? No.
        # 2000 Hz. 2 seconds = 4000 samples.
        # Hop = 20 samples. 4000 / 20 = 200 frames.
        # Wait, config says Win=50, Hop=20.
        # Let's check the output shape of one file roughly.
        # If we just append, we might have ragged arrays if lengths differ.
        # We will pad/crop to a fixed length of 128 frames (power of 2 friendly) or similar.
        # Let's target 100 frames (approx 2s).
        target_len = 100
        c, t = spec.shape
        if t < target_len:
            pad_amt = target_len - t
            spec = np.pad(spec, ((0, 0), (0, pad_amt)), mode="constant")
        elif t > target_len:
            spec = spec[:, :target_len]

        data_list.append(spec)
        id_list.append(row["clip"])

        if "label" in row:
            label_list.append(row["label"])

    data = np.stack(data_list).astype(np.float32)  # (N, n_mels, time)
    ids = np.array(id_list)

    np.save(data_path, data)
    np.save(ids_path, ids)

    if label_list:
        labels = np.array(label_list).astype(np.float32)
        np.save(labels_path, labels)
    else:
        labels = None

    return data, labels, ids


class WhaleDataset(Dataset):
    def __init__(self, data, labels, transforms=None, stats=None):
        """
        Args:
            data (np.ndarray): Shape (N, n_mels, time)
            labels (np.ndarray): Shape (N,) or None
            transforms (nn.Module): Augmentation transforms
            stats (tuple): (mean, std) for normalization
        """
        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.stats = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load data: (n_mels, time)
        x = self.data[idx]

        # Convert to tensor
        x = torch.from_numpy(x)

        # Normalize
        if self.stats:
            mean, std = self.stats
            x = (x - mean) / (std + 1e-6)

        # Apply Transforms (SpecAugment)
        # Transforms expect (channel, freq, time) or (freq, time)?
        # Torchaudio TimeMasking expects (..., freq, time).
        # Our x is (freq, time). We assume channel dim is handled or not needed.
        # Actually, TimeMasking works on tensor.
        if self.transforms:
            # Add channel dim for transform if needed, or just apply
            # Torchaudio transforms usually work on (C, F, T) or (F, T).
            # Let's ensure (1, F, T) for consistency with CNNs.
            x = x.unsqueeze(0)  # (1, F, T)
            x = self.transforms(x)
        else:
            x = x.unsqueeze(0)  # (1, F, T)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        else:
            # Return dummy label for test set
            return x, torch.tensor(0.0, dtype=torch.float32)


def get_dataloaders(debug=DEBUG, load_cached_data=True):
    """
    Main function to prepare datasets and dataloaders.
    """
    # Load Metadata
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)

    if debug:
        df_train = df_train.head(DEBUG_SUBSET_SIZE)
        df_val = df_val.head(DEBUG_SUBSET_SIZE)
        df_test = df_test.head(DEBUG_SUBSET_SIZE)

    # Load/Cache Data
    # Note: We use "train" cache for training data, "val" for validation, etc.
    train_data, train_labels, _ = load_and_cache_data(
        "train", df_train, load_cached_data
    )
    val_data, val_labels, _ = load_and_cache_data("val", df_val, load_cached_data)
    test_data, _, test_ids = load_and_cache_data("test", df_test, load_cached_data)

    # Compute Statistics on Train Data
    mean = np.mean(train_data)
    std = np.std(train_data)
    stats = (mean, std)
    print(f"Dataset Stats - Mean: {mean:.4f}, Std: {std:.4f}")

    # Create Datasets
    train_dataset = WhaleDataset(
        train_data, train_labels, transforms=get_transforms("train"), stats=stats
    )

    val_dataset = WhaleDataset(
        val_data, val_labels, transforms=get_transforms("val"), stats=stats
    )

    test_dataset = WhaleDataset(
        test_data, None, transforms=get_transforms("test"), stats=stats
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    return train_loader, val_loader, test_loader, test_ids
