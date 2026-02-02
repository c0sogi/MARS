import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config, compute_spectrograms
from library.utils import seed_everything


class WhaleDataset(Dataset):
    """
    Dataset class for Right Whale Detection.
    Handles dual-stream spectrograms and SpecAugment.
    """

    def __init__(self, x1, x2, y, augment=False):
        """
        Args:
            x1 (np.ndarray): Spectral stream features (High Frequency Resolution).
            x2 (np.ndarray): Temporal stream features (High Temporal Resolution).
            y (np.ndarray): Labels.
            augment (bool): Whether to apply SpecAugment.
        """
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.augment = augment

        # Augmentations
        # Parameters aligned with the reference configuration
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=40)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Convert numpy to tensor
        spec1 = torch.from_numpy(self.x1[idx])
        spec2 = torch.from_numpy(self.x2[idx])
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.augment:
            # Apply SpecAugment to both streams independently
            spec1 = self.freq_mask(spec1)
            spec1 = self.time_mask(spec1)

            spec2 = self.freq_mask(spec2)
            spec2 = self.time_mask(spec2)

        return spec1, spec2, label


def prepare_data(metadata_df, config=Config, cache_name="train", load_cached_data=True):
    """
    Loads audio, computes dual spectrograms, and caches the result to disk.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing file paths and labels.
        config (class): Configuration class with parameters.
        cache_name (str): Name for the cache file (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X1, X2, Y, Clips) numpy arrays.
    """
    # Ensure reproducibility
    seed_everything(config.SEED)

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(config.WORKING_DIR, f"{cache_name}.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}...")
        try:
            data = np.load(cache_path)
            return data["x1"], data["x2"], data["y"], data["clips"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {len(metadata_df)} files for {cache_name}...")
    x1_list, x2_list, y_list, clips_list = [], [], [], []

    for idx, row in metadata_df.iterrows():
        file_path = os.path.join(config.INPUT_DIR, row["file_path"])

        try:
            # Load audio using torchaudio
            # compute_spectrograms expects a tensor input
            waveform, sr = torchaudio.load(file_path)

            # Compute features using the imported function from library.config
            # This returns normalized tensors
            s1, s2 = compute_spectrograms(waveform, config)

            x1_list.append(s1.numpy())
            x2_list.append(s2.numpy())

            # Handle labels (test set might not have them)
            if "label" in row:
                y_list.append(row["label"])
            else:
                y_list.append(-1)  # Dummy for test

            clips_list.append(row["clip_name"])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    if len(x1_list) == 0:
        raise RuntimeError(f"No data processed successfully for {cache_name}.")

    # Convert to numpy arrays
    # Using newaxis to ensure shape (N, 1, F, T) compatible with Conv2d input
    X1 = np.concatenate([x[np.newaxis, ...] for x in x1_list], axis=0)
    X2 = np.concatenate([x[np.newaxis, ...] for x in x2_list], axis=0)
    Y = np.array(y_list, dtype=np.float32)
    Clips = np.array(clips_list)

    # Save to cache
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, x1=X1, x2=X2, y=Y, clips=Clips)

    return X1, X2, Y, Clips
