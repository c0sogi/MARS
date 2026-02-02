import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import LogMelSpectrogram, InstanceNorm, SpecAugment


def load_audio(file_path):
    """
    Loads audio file using soundfile and pads/crops to the fixed length
    defined in Config.N_SAMPLES.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Audio waveform as a float32 numpy array.
    """
    try:
        wav, sr = sf.read(file_path)
        # Ensure fixed length
        if len(wav) < Config.N_SAMPLES:
            pad_width = Config.N_SAMPLES - len(wav)
            wav = np.pad(wav, (0, pad_width), mode="constant")
        else:
            wav = wav[: Config.N_SAMPLES]
        return wav.astype(np.float32)
    except Exception:
        # Return silent waveform on error
        return np.zeros(Config.N_SAMPLES, dtype=np.float32)


def process_data(metadata_path, cache_name, load_cached=True, max_samples=None):
    """
    Loads metadata, processes audio into spectrograms, and caches the result.
    Implements the caching logic to avoid re-processing on every run.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_name (str): Identifier for the cache file (e.g., 'train', 'val').
        load_cached (bool): If True, attempts to load from disk first.
        max_samples (int, optional): Limit the number of samples (for debugging).

    Returns:
        tuple: (specs, labels, clips) as numpy arrays.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.npz")

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        specs = data["specs"]
        labels = data["labels"]
        clips = data["clips"]

        if max_samples is not None:
            return specs[:max_samples], labels[:max_samples], clips[:max_samples]

        return specs, labels, clips

    # 2. Process from scratch
    df = pd.read_csv(metadata_path)

    if max_samples is not None:
        df = df.iloc[:max_samples]

    # Initialize transforms from library
    mel_transform = LogMelSpectrogram()
    norm_transform = InstanceNorm()

    specs = []
    labels = []
    clips = []

    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        wav = load_audio(full_path)

        # Convert to tensor for transforms (add channel dim: 1, Time)
        wav_tensor = torch.tensor(wav).unsqueeze(0)

        with torch.no_grad():
            # Compute Spectrogram and Normalize
            spec = mel_transform(wav_tensor)
            spec = norm_transform(spec)

        # Remove channel dim for storage (N_MELS, Time)
        spec = spec.squeeze(0).numpy()

        specs.append(spec)
        clips.append(row["clip_name"])

        if "label" in row:
            labels.append(row["label"])
        else:
            labels.append(-1)  # Placeholder for test data

    specs = np.stack(specs)
    labels = np.array(labels)
    clips = np.array(clips)

    # Save to cache
    np.savez(cache_path, specs=specs, labels=labels, clips=clips)

    return specs, labels, clips


def get_transforms(mode="train"):
    """
    Factory function to get the appropriate transforms (augmentation).

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        nn.Module or None: Transform module.
    """
    if mode == "train":
        return SpecAugment(
            time_mask_param=Config.SPECAUG_TIME_MASK,
            freq_mask_param=Config.SPECAUG_FREQ_MASK,
        )
    return None


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Handles 3-channel expansion, augmentation, and pseudo-labels.
    """

    def __init__(
        self, specs, labels, transform=None, pseudo_labels=None, is_test=False
    ):
        """
        Args:
            specs (np.ndarray): Array of spectrograms.
            labels (np.ndarray): Array of ground truth labels.
            transform (callable, optional): Augmentation transform (e.g., SpecAugment).
            pseudo_labels (np.ndarray, optional): Soft labels for self-training.
            is_test (bool): If True, returns dummy targets.
        """
        self.specs = specs
        self.labels = labels
        self.transform = transform
        self.pseudo_labels = pseudo_labels
        self.is_test = is_test

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Load spectrogram
        spec = self.specs[idx]

        # Convert to tensor
        spec = torch.tensor(spec, dtype=torch.float32)

        # Expand to 3 channels for EfficientNet backbone compatibility
        # Input: (F, T) -> Output: (3, F, T)
        spec = spec.unsqueeze(0).repeat(3, 1, 1)

        # Apply augmentation (SpecAugment)
        if self.transform:
            spec = self.transform(spec)

        # Handle Test mode
        if self.is_test:
            return spec, torch.tensor(0, dtype=torch.float32)

        # Handle Pseudo-labels vs Hard labels
        if self.pseudo_labels is not None:
            target = self.pseudo_labels[idx]
        else:
            target = self.labels[idx]

        return spec, torch.tensor(target, dtype=torch.float32)
