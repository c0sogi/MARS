import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def get_transforms(mode="train"):
    """
    Returns albumentations transforms for the dual-stream input.
    Input format is assumed to be (Height, Width, Channels).
    """
    if mode == "train":
        return A.Compose(
            [
                # XYMasking acts as SpecAugment (Time and Frequency masking)
                A.XYMasking(
                    num_masks_x=(1, 3),
                    mask_x_length=(1, 20),
                    num_masks_y=(1, 3),
                    mask_y_length=(1, 20),
                    fill_value=0,
                    p=0.5,
                ),
            ]
        )
    else:
        return A.Compose([])


def mixup_data(x1, x2, y, alpha=1.0, device="cuda"):
    """
    Applies MixUp augmentation to the batch.
    Returns:
        mixed_x1: Mixed Stream A inputs
        mixed_x2: Mixed Stream B inputs
        y_a: Targets for first component
        y_b: Targets for second component
        lam: Mixing lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, y_a, y_b, lam


class EEGDataset(Dataset):
    def __init__(self, eeg_data, spec_data, targets=None, transform=None):
        """
        Args:
            eeg_data: Numpy array (or mmap) of shape (N, 128, 256, 19)
            spec_data: Numpy array (or mmap) of shape (N, 256, 256, 4)
            targets: Numpy array of shape (N, 6) or None
            transform: Albumentations Compose object
        """
        self.eeg_data = eeg_data
        self.spec_data = spec_data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        # Load data from mmap and cast to float32 for processing
        X_eeg = self.eeg_data[idx].astype(np.float32)  # (128, 256, 19)
        X_spec = self.spec_data[idx].astype(np.float32)  # (256, 256, 4)

        # Apply transforms
        if self.transform:
            # Apply same masking logic to both streams or independently.
            # Here we apply independently as they represent different domains.
            X_eeg = self.transform(image=X_eeg)["image"]
            X_spec = self.transform(image=X_spec)["image"]

        # Convert to PyTorch Tensors and rearrange to (Channels, Height, Width)
        # Input: (H, W, C) -> Output: (C, H, W)
        X_eeg = torch.tensor(X_eeg).permute(2, 0, 1)  # -> (19, 128, 256)
        X_spec = torch.tensor(X_spec).permute(2, 0, 1)  # -> (4, 256, 256)

        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return (X_eeg, X_spec), y
        else:
            return (X_eeg, X_spec)


def process_eeg_signal(file_path, eeg_names=Config.EEG_NAMES):
    """
    Loads raw EEG, computes MelSpectrogram, and stacks channels.
    Returns: Numpy array (128, 256, 19)
    """
    try:
        df = pd.read_parquet(file_path, columns=eeg_names)
    except Exception:
        # Fallback for corrupt files
        return np.zeros((Config.N_MELS, 256, len(eeg_names)), dtype=np.float32)

    # Handle NaNs (replace with 0 to avoid artifacts)
    data = df.values
    data = np.nan_to_num(data, nan=0.0)

    # Pad or Crop to fixed duration (10000 samples)
    target_len = Config.DURATION * Config.SR
    current_len = data.shape[0]

    if current_len < target_len:
        pad_width = target_len - current_len
        data = np.pad(data, ((0, pad_width), (0, 0)), mode="constant")
    elif current_len > target_len:
        start = (current_len - target_len) // 2
        data = data[start : start + target_len, :]

    # Convert to Tensor: (Time, Channels) -> (Channels, Time)
    tensor_data = torch.tensor(data.T, dtype=torch.float32)

    # Compute MelSpectrogram
    # Output shape: (Channels, n_mels, time_steps)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        center=True,
        power=2.0,
    )

    # Suppress warnings for short signals if any
    try:
        melspec = mel_transform(tensor_data)
    except Exception:
        return np.zeros((Config.N_MELS, 256, len(eeg_names)), dtype=np.float32)

    # Log Transform
    melspec = torch.log(melspec + 1e-6)

    # Channel-wise Normalization
    mean = melspec.mean(dim=(1, 2), keepdim=True)
    std = melspec.std(dim=(1, 2), keepdim=True) + 1e-6
    melspec = (melspec - mean) / std

    # Resize Time Dimension to 256
    # Current time dim is approx 10000/39 = 257. We interpolate to 256.
    if melspec.shape[2] != 256:
        melspec = torch.nn.functional.interpolate(
            melspec.unsqueeze(0),
            size=(Config.N_MELS, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    # Rearrange to (H, W, C) -> (128, 256, 19)
    melspec = melspec.permute(1, 2, 0).numpy()

    return melspec


def process_spectrogram_image(file_path):
    """
    Loads Kaggle Spectrogram, extracts 4 regions, resizes and stacks.
    Returns: Numpy array (256, 256, 4)
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        return np.zeros((Config.SPEC_SIZE[0], Config.SPEC_SIZE[1], 4), dtype=np.float32)

    regions = ["LL", "RL", "LP", "RP"]
    processed_channels = []

    # Fill NaNs with 0 (or global mean, but 0 is safe for log)
    df = df.fillna(0)

    for region in regions:
        # Identify columns for this region
        cols = [c for c in df.columns if region in c]

        if not cols:
            img = np.zeros(Config.SPEC_SIZE, dtype=np.float32)
        else:
            img = df[cols].values  # (Time, Freq)

            # Log transform
            img = np.log(img + 1e-6)

            # Normalize
            mean = img.mean()
            std = img.std() + 1e-6
            img = (img - mean) / std

            # Resize to (256, 256)
            # cv2.resize expects (Width, Height)
            img = cv2.resize(
                img,
                (Config.SPEC_SIZE[1], Config.SPEC_SIZE[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        processed_channels.append(img)

    # Stack depth-wise
    result = np.stack(processed_channels, axis=-1)
    return result


def generate_cache(metadata_df, mode, load_cached_data=True):
    """
    Generates or loads cached numpy arrays for the dataset.
    Uses memory mapping to handle large datasets efficiently.
    """
    cache_dir = Config.WORKING_DIR
    eeg_path = os.path.join(cache_dir, f"{mode}_eeg.npy")
    spec_path = os.path.join(cache_dir, f"{mode}_spec.npy")
    target_path = os.path.join(cache_dir, f"{mode}_targets.npy")

    # Check if cache exists
    if load_cached_data and os.path.exists(eeg_path) and os.path.exists(spec_path):
        print(f"Loading cached data for {mode}...")
        eeg_data = np.load(eeg_path, mmap_mode="r")
        spec_data = np.load(spec_path, mmap_mode="r")
        if mode != "test":
            targets = np.load(target_path)
        else:
            targets = None
        return eeg_data, spec_data, targets

    print(f"Generating cache for {mode} (Rows: {len(metadata_df)})...")
    os.makedirs(cache_dir, exist_ok=True)

    n_samples = len(metadata_df)

    # Define shapes
    eeg_shape = (n_samples, 128, 256, 19)
    spec_shape = (n_samples, 256, 256, 4)

    # Create memory-mapped files for writing
    # We use float16 to save disk space and memory (approx 100GB total for train)
    eeg_mmap = np.lib.format.open_memmap(
        eeg_path, mode="w+", dtype=np.float16, shape=eeg_shape
    )
    spec_mmap = np.lib.format.open_memmap(
        spec_path, mode="w+", dtype=np.float16, shape=spec_shape
    )

    if mode != "test":
        targets = np.zeros((n_samples, 6), dtype=np.float32)
    else:
        targets = None

    # Reset index to ensure enumeration matches array indices
    metadata_df = metadata_df.reset_index(drop=True)

    # Iterate and process
    for i, row in metadata_df.iterrows():
        # Construct full paths
        eeg_file = os.path.join(Config.INPUT_DIR, row["eeg_path"])
        spec_file = os.path.join(Config.INPUT_DIR, row["spec_path"])

        # Process Stream A
        eeg_img = process_eeg_signal(eeg_file)
        eeg_mmap[i] = eeg_img.astype(np.float16)

        # Process Stream B
        spec_img = process_spectrogram_image(spec_file)
        spec_mmap[i] = spec_img.astype(np.float16)

        # Targets
        if mode != "test":
            targets[i] = row[Config.TARGET_COLS].values.astype(np.float32)

        # Flush periodically to ensure data is written to disk
        if i % 1000 == 0:
            eeg_mmap.flush()
            spec_mmap.flush()

    # Final flush
    eeg_mmap.flush()
    spec_mmap.flush()

    if mode != "test":
        np.save(target_path, targets)

    print(f"Cache generation complete for {mode}.")

    # Re-open in read-only mode
    eeg_data = np.load(eeg_path, mmap_mode="r")
    spec_data = np.load(spec_path, mmap_mode="r")

    return eeg_data, spec_data, targets


def get_dataloader(
    mode, batch_size=Config.BATCH_SIZE, load_cached_data=True, shuffle=True
):
    """
    Creates and returns a DataLoader for the specified mode.
    Handles metadata loading and cache generation.
    """
    # Load Metadata
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_CSV)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_CSV)
    elif mode == "test":
        df = pd.read_csv(Config.TEST_CSV)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Generate or Load Data
    eeg_data, spec_data, targets = generate_cache(
        df, mode, load_cached_data=load_cached_data
    )

    # Create Dataset
    transform = get_transforms(mode)
    dataset = EEGDataset(eeg_data, spec_data, targets, transform=transform)

    # Create Loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=(mode == "train"),
    )

    return loader
