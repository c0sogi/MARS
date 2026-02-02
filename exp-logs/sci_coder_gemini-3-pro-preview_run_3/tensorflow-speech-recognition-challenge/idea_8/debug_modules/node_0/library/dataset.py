import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from library.config import Config
from library.feature_extractor import FeatureExtractor


class SpeechCommandDataset(Dataset):
    """
    Dataset class that loads pre-computed 3-Channel Multi-Resolution Spectrograms.
    Supports SpecAugment with strict constraints.
    """

    def __init__(self, df, augment=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'filepath' and 'label'.
            augment (bool): If True, applies SpecAugment.
        """
        self.df = df.reset_index(drop=True)
        self.augment = augment

        # Pre-compute cache paths and label indices for efficiency
        self.file_paths = []
        self.label_indices = []

        for _, row in self.df.iterrows():
            # Use the library function to ensure path consistency
            cache_path = FeatureExtractor._get_cache_path(row["filepath"])
            self.file_paths.append(cache_path)

            # Map label string to integer index
            label_str = row["label"]
            if label_str in Config.LABEL_TO_IDX:
                self.label_indices.append(Config.LABEL_TO_IDX[label_str])
            else:
                # Fallback for safety, though metadata should be clean
                self.label_indices.append(Config.LABEL_TO_IDX["unknown"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.label_indices[idx]

        # 1. Load Data
        if os.path.exists(path):
            # Load cached numpy array
            spec = np.load(path)
        else:
            # Fallback: Compute on-the-fly if cache is missing
            # This ensures the dataset works even if caching was interrupted
            rel_path = self.df.iloc[idx]["filepath"]
            audio_path = os.path.join(Config.INPUT_ROOT, rel_path)
            spec = FeatureExtractor.compute_multires_spec(audio_path)

        # Convert to Tensor (3, 64, 101)
        spec_tensor = torch.from_numpy(spec)

        # 2. Augmentation
        if self.augment:
            spec_tensor = self.spec_augment(spec_tensor)

        return spec_tensor, label

    def spec_augment(self, spec):
        """
        Applies Frequency and Time Masking.
        - Fills masks with the minimum value of the spectrogram.
        - Time mask length is strictly limited by Config.TIME_MASK_PARAM (<20% duration).
        - Applies same mask across all 3 channels to maintain temporal/spectral alignment.
        """
        # spec shape: (Channels=3, Freq=64, Time=101)
        min_val = spec.min()
        C, F, T = spec.shape

        # --- Frequency Masking ---
        f_param = Config.FREQ_MASK_PARAM
        # Random mask length [0, f_param]
        f_mask_len = int(torch.randint(0, f_param + 1, (1,)).item())

        if f_mask_len > 0:
            f_start = int(torch.randint(0, F - f_mask_len + 1, (1,)).item())
            # Apply to all channels
            spec[:, f_start : f_start + f_mask_len, :] = min_val

        # --- Time Masking ---
        t_param = Config.TIME_MASK_PARAM
        # Random mask length [0, t_param]
        # Config.TIME_MASK_PARAM is 20, which is < 20% of 101 time steps.
        t_mask_len = int(torch.randint(0, t_param + 1, (1,)).item())

        if t_mask_len > 0:
            t_start = int(torch.randint(0, T - t_mask_len + 1, (1,)).item())
            # Apply to all channels
            spec[:, :, t_start : t_start + t_mask_len] = min_val

        return spec


def get_weighted_sampler(df):
    """
    Creates a WeightedRandomSampler to balance class distribution in batches.

    Args:
        df (pd.DataFrame): Training metadata.

    Returns:
        WeightedRandomSampler: Sampler with weights inverse to class frequency.
    """
    # 1. Calculate Class Counts
    label_counts = df["label"].value_counts()

    # 2. Calculate Weight per Class (Inverse Frequency)
    weights_per_class = {}
    for label in Config.LABELS:
        count = label_counts.get(label, 0)
        if count > 0:
            weights_per_class[label] = 1.0 / count
        else:
            weights_per_class[label] = 0.0

    # 3. Assign Weight to Each Sample
    sample_weights = [weights_per_class[label] for label in df["label"]]

    # 4. Create Sampler
    # Convert to DoubleTensor for the sampler
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )

    return sampler
