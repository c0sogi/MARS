import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from library.config import Config
from library.audio_processor import AudioProcessor


class CachedSpeechDataset(Dataset):
    """
    PyTorch Dataset for loading 3-Channel Multi-Resolution Log-Mel Spectrograms.
    Handles loading from cache via AudioProcessor and applying SpecAugment.
    """

    def __init__(self, metadata_path, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train, val, or test).
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label handling.
        """
        self.mode = mode
        self.processor = AudioProcessor()

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Pre-calculate label indices for speed
        self.label_map = Config.LABEL2ID

        # For test set, labels might be placeholders, but we still map them if possible
        # or handle them in __getitem__

    def __len__(self):
        return len(self.df)

    def _apply_spec_augment(self, spec_tensor):
        """
        Applies SpecAugment (Frequency and Time Masking) to the spectrogram.
        Masks are filled with the minimum value of the spectrogram.

        Args:
            spec_tensor (torch.Tensor): Shape (C, n_mels, time)

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        # Create a copy to avoid modifying the cache in memory if it's shared
        augmented = spec_tensor.clone()

        C, n_mels, time_steps = augmented.shape
        min_val = augmented.min()

        # --- Frequency Masking ---
        # Mask parameter F
        F = Config.FREQ_MASK_PARAM
        f = random.randint(0, F)
        if f > 0:
            f0 = random.randint(0, max(0, n_mels - f))
            augmented[:, f0 : f0 + f, :] = min_val

        # --- Time Masking ---
        # Mask parameter T, strictly < 20% of duration
        T_param = Config.TIME_MASK_PARAM

        # Calculate 20% limit based on actual time steps
        max_time_mask = int(0.2 * time_steps)

        # Effective mask width
        t = random.randint(0, T_param)
        t = min(t, max_time_mask)

        if t > 0:
            t0 = random.randint(0, max(0, time_steps - t))
            augmented[:, :, t0 : t0 + t] = min_val

        return augmented

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        # 1. Load Features (3, 80, T)
        # AudioProcessor handles caching logic (load if exists, else compute & save)
        features_np = self.processor.process_file(filepath, load_cached_data=True)
        features_tensor = torch.from_numpy(features_np).float()

        # 2. Augmentation (Train only)
        if self.mode == "train":
            features_tensor = self._apply_spec_augment(features_tensor)

        # 3. Label Processing
        # Map label string to ID
        # If label is not in map (should not happen with correct metadata), default to unknown or error
        label_id = self.label_map.get(label_str, self.label_map.get("unknown", 0))

        return features_tensor, torch.tensor(label_id, dtype=torch.long)


def get_class_weights(dataset):
    """
    Calculates sample weights for WeightedRandomSampler to balance the dataset.

    Args:
        dataset (CachedSpeechDataset): The training dataset.

    Returns:
        torch.Tensor: A tensor of weights, one for each sample in the dataset.
    """
    # Extract labels from the dataframe directly for speed
    # (avoiding loading all audio files)
    labels = dataset.df["label"].values

    # Count class occurrences
    # We need to map string labels to IDs to ensure order matches Config.LABELS logic if needed,
    # but here we just need counts per unique string to compute weights.

    # Calculate count per class
    class_counts = dataset.df["label"].value_counts().to_dict()

    # Calculate weight per class: 1.0 / count
    class_weights = {k: 1.0 / v for k, v in class_counts.items()}

    # Assign a weight to each sample
    sample_weights = [class_weights[label] for label in labels]

    return torch.tensor(sample_weights, dtype=torch.double)
