import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from library.config import AudioConfig, TrainConfig
from library.utils import set_seed

# Set seed for reproducibility across operations
set_seed(TrainConfig.seed)


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Call Detection.
    Wraps pre-processed spectrogram tensors and applies on-the-fly augmentations.
    """

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of spectrograms (N, n_mels, time).
            labels (np.ndarray, optional): Array of binary labels (N,).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (Spectrogram)
        img = self.images[idx]

        # Convert numpy to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

        # Ensure Channel Dimension: (F, T) -> (1, F, T)
        # ConvNeXt expects (B, C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)

        # Apply Augmentations (e.g., SpecAugment)
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        else:
            # For inference, just return the image
            return img


def load_audio_clip(rel_path):
    """
    Reads an audio file, ensures mono channel, and pads/crops to fixed duration.
    """
    full_path = os.path.join(TrainConfig.INPUT_ROOT, rel_path)

    try:
        # Read audio file
        audio, sr = sf.read(full_path)

        # Ensure Mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Fix Length (Pad or Crop)
        target_len = AudioConfig.num_samples
        current_len = len(audio)

        if current_len < target_len:
            pad_width = target_len - current_len
            audio = np.pad(audio, (0, pad_width), mode="constant")
        elif current_len > target_len:
            audio = audio[:target_len]

        return torch.from_numpy(audio).float()

    except Exception as e:
        print(f"Warning: Error reading {rel_path}: {e}. Returning silent clip.")
        return torch.zeros(AudioConfig.num_samples).float()


def generate_spectrograms(df, split_name):
    """
    Iterates through the dataframe, loads audio, and generates normalized Log-Mel Spectrograms.
    Returns: (images_numpy, labels_numpy_or_None)
    """
    print(f"Processing {split_name} data from raw audio...")

    # Define Spectrogram Transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=AudioConfig.sr,
        n_fft=AudioConfig.n_fft,
        win_length=AudioConfig.win_length,
        hop_length=AudioConfig.hop_length,
        f_min=AudioConfig.fmin,
        f_max=AudioConfig.fmax,
        n_mels=AudioConfig.n_mels,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    # Define Log Transform
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

    images_list = []
    labels_list = []
    has_labels = "label" in df.columns

    count = 0
    total = len(df)

    for _, row in df.iterrows():
        # 1. Load Waveform
        waveform = load_audio_clip(row["file_path"])

        # 2. Compute Mel Spectrogram
        # waveform: (time) -> spec: (n_mels, time)
        spec = mel_transform(waveform)

        # 3. Log Scale
        spec = db_transform(spec)

        # 4. Instance-level Min-Max Normalization
        min_val = spec.min()
        max_val = spec.max()
        # Avoid division by zero
        spec = (spec - min_val) / (max_val - min_val + 1e-6)

        # Store as float32 numpy array to save memory
        images_list.append(spec.numpy().astype(np.float32))

        if has_labels:
            labels_list.append(row["label"])

        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count}/{total} clips...")

    # Stack into a single array: (N, n_mels, time)
    images = np.stack(images_list)

    if has_labels:
        labels = np.array(labels_list, dtype=np.float32)
        return images, labels
    else:
        return images, None


def get_data(split_name, csv_path, load_cached_data=True):
    """
    Manages data loading with caching mechanism.
    """
    # Determine cache path
    suffix = "_debug" if TrainConfig.debug else ""
    cache_filename = f"{split_name}{suffix}.npz"
    cache_path = os.path.join(TrainConfig.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}...")
        try:
            data = np.load(cache_path)
            images = data["images"]
            labels = data["labels"] if "labels" in data else None
            print(f"  Loaded {len(images)} samples.")
            return images, labels
        except Exception as e:
            print(f"  Failed to load cache ({e}). Recomputing...")

    # 2. Compute from scratch
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle Debug Mode
    if TrainConfig.debug:
        print(
            f"  DEBUG MODE: limiting {split_name} to {TrainConfig.debug_samples} samples."
        )
        df = df.head(TrainConfig.debug_samples)

    images, labels = generate_spectrograms(df, split_name)

    # 3. Save to cache
    os.makedirs(TrainConfig.CACHE_DIR, exist_ok=True)
    print(f"Saving {split_name} data to {cache_path}...")
    save_dict = {"images": images}
    if labels is not None:
        save_dict["labels"] = labels
    np.savez(cache_path, **save_dict)

    return images, labels


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed .npz files.

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n=== Initializing Data Pipeline ===")

    # 1. Load Data (with caching)
    train_X, train_y = get_data("train", TrainConfig.TRAIN_CSV, load_cached_data)
    val_X, val_y = get_data("val", TrainConfig.VAL_CSV, load_cached_data)
    test_X, _ = get_data("test", TrainConfig.TEST_CSV, load_cached_data)

    # 2. Define Augmentations (Training Only)
    # SpecAugment: Time and Frequency Masking
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(
            time_mask_param=TrainConfig.spec_aug_time_mask
        ),
        torchaudio.transforms.FrequencyMasking(
            freq_mask_param=TrainConfig.spec_aug_freq_mask
        ),
    )

    # 3. Instantiate Datasets
    train_dataset = WhaleDataset(train_X, train_y, transform=train_transform)
    val_dataset = WhaleDataset(val_X, val_y, transform=None)
    test_dataset = WhaleDataset(test_X, None, transform=None)

    # 4. Create DataLoaders
    # Pin memory speeds up host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
        drop_last=True,  # Helpful for Mixup stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=False,
        num_workers=TrainConfig.num_workers,
        pin_memory=True,
    )

    print("=== Data Pipeline Ready ===\n")
    return train_loader, val_loader, test_loader
