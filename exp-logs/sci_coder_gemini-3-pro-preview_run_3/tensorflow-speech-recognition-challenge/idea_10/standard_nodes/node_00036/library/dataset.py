import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from library.config import LABEL_TO_IDX, CACHE_DIR, INPUT_ROOT, SEED
from library.preprocessing import (
    get_cache_filename,
    get_audio_path,
    load_and_pad_waveform,
    compute_multires_melspec,
)
from library.utils import set_seed

# Ensure reproducibility
set_seed(SEED)


class HybridAudioDataset(Dataset):
    """
    Dataset class for the Hybrid 1D-2D Dual-Stream CRNN.
    Handles loading of cached 2D spectrograms and on-the-fly 1D raw waveforms.
    """

    def __init__(
        self,
        metadata_file,
        cache_dir=CACHE_DIR,
        spec_augment=None,
        raw_augment=None,
        is_test=False,
    ):
        """
        Args:
            metadata_file (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            cache_dir (str): Directory where .npy spectrogram files are stored.
            spec_augment (callable, optional): Transform to apply to 2D spectrograms.
            raw_augment (callable, optional): Transform to apply to 1D raw waveforms.
            is_test (bool): Flag indicating if this is the test set (ignores labels).
        """
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.df = pd.read_csv(metadata_file)
        self.cache_dir = cache_dir
        self.spec_augment = spec_augment
        self.raw_augment = raw_augment
        self.is_test = is_test

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            spec_2d (torch.Tensor): Shape (3, n_mels, frames)
            wave_1d (torch.Tensor): Shape (1, num_samples)
            label (int): Class index
        """
        row = self.df.iloc[idx]
        rel_filepath = row["filepath"]
        label_str = row["label"]

        # ---------------------------------------------------------------------
        # Stream 1: 2D Multi-Resolution Spectrogram (Cached)
        # ---------------------------------------------------------------------
        cache_filename = get_cache_filename(rel_filepath)
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # Robust Caching Logic:
        # 1. Try to load from cache
        # 2. If missing, compute, save, then return
        if os.path.exists(cache_path):
            try:
                spec_np = np.load(cache_path)
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}. Recomputing...")
                spec_np = self._compute_and_cache(rel_filepath, cache_path)
        else:
            # Cache miss - compute and save
            spec_np = self._compute_and_cache(rel_filepath, cache_path)

        # Convert to Tensor
        spec_2d = torch.from_numpy(spec_np).float()

        # Apply SpecAugment
        if self.spec_augment is not None:
            spec_2d = self.spec_augment(spec_2d)

        # ---------------------------------------------------------------------
        # Stream 2: 1D Raw Waveform (On-the-fly)
        # ---------------------------------------------------------------------
        full_audio_path = get_audio_path(rel_filepath)

        # load_and_pad_waveform handles loading, resampling, mono conversion, and padding
        # Returns Tensor of shape (1, NUM_SAMPLES)
        if os.path.exists(full_audio_path):
            wave_1d = load_and_pad_waveform(full_audio_path)
        else:
            # Fallback for missing audio files (should be rare given validation)
            from library.config import NUM_SAMPLES

            wave_1d = torch.zeros(1, NUM_SAMPLES)

        # Apply RawAudioAugment
        if self.raw_augment is not None:
            wave_1d = self.raw_augment(wave_1d)

        # ---------------------------------------------------------------------
        # Label
        # ---------------------------------------------------------------------
        if self.is_test:
            # For test set, return a dummy label (e.g., index of 'unknown' or 0)
            # The caller should rely on the dataset index to map back to filename
            label = 0
        else:
            label = LABEL_TO_IDX.get(label_str, LABEL_TO_IDX["unknown"])

        return spec_2d, wave_1d, label

    def _compute_and_cache(self, rel_filepath, cache_path):
        """
        Helper to compute features from raw audio and save to cache.
        """
        full_path = get_audio_path(rel_filepath)
        if os.path.exists(full_path):
            waveform = load_and_pad_waveform(full_path)
        else:
            from library.config import NUM_SAMPLES

            waveform = torch.zeros(1, NUM_SAMPLES)

        features = compute_multires_melspec(waveform)
        np.save(cache_path, features)
        return features

    def get_sample_weights(self):
        """
        Calculates weights for each sample in the dataset to be used with
        WeightedRandomSampler for handling class imbalance.

        Returns:
            list[float]: A list of weights corresponding to each sample in the dataset.
        """
        if self.is_test:
            return None

        # Count class frequencies
        label_counts = self.df["label"].value_counts()

        # Calculate weight for each class: w = 1 / count
        # We use the label string directly to map
        class_weights = {label: 1.0 / count for label, count in label_counts.items()}

        # Map weights to samples
        sample_weights = [class_weights.get(label, 0) for label in self.df["label"]]

        return sample_weights
