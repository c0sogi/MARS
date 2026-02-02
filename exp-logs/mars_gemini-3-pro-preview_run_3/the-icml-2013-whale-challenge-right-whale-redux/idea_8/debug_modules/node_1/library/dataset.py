import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Serves pre-computed spectrograms from memory and applies on-the-fly augmentation.
    """

    def __init__(self, images, labels=None, is_train=False):
        """
        Args:
            images (np.ndarray): Array of spectrograms (N, H, W).
            labels (np.ndarray, optional): Array of labels (N,).
            is_train (bool): Whether this is the training set (enables augmentation).
        """
        self.images = images
        self.labels = labels
        self.is_train = is_train

        # SpecAugment parameters
        self.mask_time_prob = Config.MASK_TIME_PROB
        self.mask_freq_prob = Config.MASK_FREQ_PROB
        self.mask_time_len = Config.MASK_TIME_LENGTH
        self.mask_freq_len = Config.MASK_FREQ_LENGTH

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image: shape (H, W) -> (Freq, Time)
        img = self.images[idx]

        # Convert to tensor and add channel dimension: (1, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # Apply SpecAugment if in training mode
        if self.is_train and Config.USE_SPECAUG:
            img_tensor = self.apply_spec_augment(img_tensor)

        # Return (image, label) or (image, dummy_label)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            # For test set, return a dummy label
            return img_tensor, torch.tensor([0.0])

    def apply_spec_augment(self, spec):
        """
        Applies Time and Frequency Masking to the spectrogram.
        Args:
            spec (torch.Tensor): Spectrogram tensor of shape (1, Freq, Time).
        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        # Time Masking
        if np.random.rand() < self.mask_time_prob:
            mask_len = np.random.randint(1, self.mask_time_len + 1)
            masker = torchaudio.transforms.TimeMasking(time_mask_param=mask_len)
            spec = masker(spec)

        # Frequency Masking
        if np.random.rand() < self.mask_freq_prob:
            mask_len = np.random.randint(1, self.mask_freq_len + 1)
            masker = torchaudio.transforms.FrequencyMasking(freq_mask_param=mask_len)
            spec = masker(spec)

        return spec


def compute_mel_spectrogram(file_path):
    """
    Reads an audio file, computes the Log-Mel Spectrogram, applies Frequency-Wise
    Normalization, and resizes it to the target dimensions.

    Args:
        file_path (str): Path to the .aif audio file.

    Returns:
        np.ndarray: Processed spectrogram of shape (H, W).
    """
    try:
        # Load audio using soundfile (robust for .aif)
        audio, sr = sf.read(file_path)

        # Handle multi-channel audio (downmix to mono)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Compute Mel Spectrogram
        # Note: We rely on the raw audio duration.
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            power=2.0,
        )
        melspec = mel_transform(audio_tensor)

        # Convert to Log Scale (dB)
        db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        melspec_db = db_transform(melspec)  # Shape: (n_mels, time)

        # Frequency-Wise Normalization (Spectral Subtraction)
        # Normalize each frequency bin independently across time
        if Config.FREQ_WISE_NORM:
            mean = melspec_db.mean(dim=1, keepdim=True)
            std = melspec_db.std(dim=1, keepdim=True)
            melspec_db = (melspec_db - mean) / (std + 1e-6)

        # Resize to fixed image size (H, W) = (224, 224)
        # Convert to numpy for OpenCV resizing
        spec_np = melspec_db.numpy()

        # cv2.resize expects (Width, Height).
        # Our spec_np is (Freq/Height, Time/Width).
        # We want output (224, 224).
        resized_spec = cv2.resize(
            spec_np,
            (Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return resized_spec

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero array as fallback
        return np.zeros(Config.IMAGE_SIZE, dtype=np.float32)


def process_and_cache_data(df, cache_path, load_cached_data=True, debug=False):
    """
    Processes the dataset defined in the dataframe.
    Implements caching logic: loads from .npz if available, otherwise computes and saves.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_path (str): Path to save/load the .npz cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, processes a small subset.

    Returns:
        tuple: (images, labels) numpy arrays.
    """
    # Adjust cache path for debug mode to avoid overwriting full cache
    if debug:
        df = df.head(Config.DEBUG_SUBSET_SIZE).copy()
        cache_path = cache_path.replace(".npz", "_debug.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            images = data["images"]
            labels = data["labels"] if "labels" in data else None
            return images, labels
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {len(df)} samples (Caching to {cache_path})...")
    images = []
    labels = []

    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        spec = compute_mel_spectrogram(full_path)
        images.append(spec)

        if "label" in row:
            labels.append(row["label"])

    images = np.array(images, dtype=np.float32)

    if len(labels) > 0:
        labels = np.array(labels, dtype=np.float32)
        # Save compressed to save space
        np.savez_compressed(cache_path, images=images, labels=labels)
        return images, labels
    else:
        np.savez_compressed(cache_path, images=images)
        return images, None


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    load_cached_data=True,
    debug=Config.DEBUG,
    batch_size=Config.BATCH_SIZE,
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        train_csv (str): Path to train metadata CSV.
        val_csv (str): Path to validation metadata CSV.
        test_csv (str): Path to test metadata CSV.
        load_cached_data (bool): Whether to use cached .npz files.
        debug (bool): Whether to run in debug mode (subset of data).
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Define Cache Paths
    train_cache = os.path.join(Config.CACHE_DIR, "train.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test.npz")

    # Process Data
    train_imgs, train_lbls = process_and_cache_data(
        train_df, train_cache, load_cached_data, debug
    )
    val_imgs, val_lbls = process_and_cache_data(
        val_df, val_cache, load_cached_data, debug
    )
    test_imgs, _ = process_and_cache_data(test_df, test_cache, load_cached_data, debug)

    # Initialize Datasets
    train_dataset = WhaleDataset(train_imgs, train_lbls, is_train=True)
    val_dataset = WhaleDataset(val_imgs, val_lbls, is_train=False)
    test_dataset = WhaleDataset(test_imgs, None, is_train=False)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
