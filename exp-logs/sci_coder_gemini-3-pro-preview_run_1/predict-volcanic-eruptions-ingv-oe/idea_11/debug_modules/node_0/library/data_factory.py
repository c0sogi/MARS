import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TABULAR_TRAIN_CACHE,
    TABULAR_VAL_CACHE,
    TABULAR_TEST_CACHE,
    VISION_GLOBAL_MAX_CACHE,
    VISION_TRAIN_CACHE_DIR,
    VISION_VAL_CACHE_DIR,
    VISION_TEST_CACHE_DIR,
    USE_LOG_TARGET,
)
from library.feature_engineering import (
    generate_tabular_features,
    compute_global_max,
    generate_vision_dataset,
)
from library.utils import log_transform_target


class VolcanoSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed, normalized spectrograms.

    Attributes:
        metadata (pd.DataFrame): Metadata containing segment_ids and targets.
        data_dir (str): Directory containing the .npy spectrogram files.
        is_test (bool): Flag to indicate if this is the test set (no targets).
    """

    def __init__(self, metadata: pd.DataFrame, data_dir: str, is_test: bool = False):
        self.metadata = metadata
        self.data_dir = data_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        segment_id = int(row["segment_id"])

        # Load pre-processed spectrogram
        # Shape: (10, 224, 224), dtype: float32, Normalized: [0, 1]
        file_path = os.path.join(self.data_dir, f"{segment_id}.npy")

        # Robust loading: if file doesn't exist (shouldn't happen if factory is used), return zeros
        if os.path.exists(file_path):
            spectrogram = np.load(file_path)
        else:
            # Fallback for safety
            spectrogram = np.zeros((10, 224, 224), dtype=np.float32)

        # Convert to Tensor
        x = torch.from_numpy(spectrogram)

        if self.is_test:
            # For test, we might return segment_id or just a dummy target
            # Returning 0.0 as dummy target
            y = torch.tensor(0.0, dtype=torch.float32)
        else:
            # Get target
            target_val = float(row["time_to_eruption"])

            # Apply Log-Scaling if configured
            if USE_LOG_TARGET:
                target_val = log_transform_target(target_val)

            y = torch.tensor(target_val, dtype=torch.float32)

        return x, y


def get_tabular_dataset(
    split: str = "train", load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Generates or loads the tabular dataset for the specified split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        pd.DataFrame: The feature matrix (including targets for train/val).
    """
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
        cache_path = TABULAR_TRAIN_CACHE
    elif split == "val":
        meta_path = VAL_METADATA_PATH
        cache_path = TABULAR_VAL_CACHE
    elif split == "test":
        meta_path = TEST_METADATA_PATH
        cache_path = TABULAR_TEST_CACHE
    else:
        raise ValueError(f"Invalid split: {split}")

    # Generate or Load Features
    df = generate_tabular_features(
        metadata_path=meta_path,
        cache_path=cache_path,
        load_cached_data=load_cached_data,
    )

    return df


def get_vision_dataset(
    split: str = "train", load_cached_data: bool = True
) -> VolcanoSpectrogramDataset:
    """
    Prepares the vision data (calculating global max, generating spectrograms)
    and returns a PyTorch Dataset object.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to use cached intermediate files.

    Returns:
        VolcanoSpectrogramDataset: The dataset ready for DataLoader.
    """
    # 1. Determine Paths
    if split == "train":
        meta_path = TRAIN_METADATA_PATH
        data_dir = VISION_TRAIN_CACHE_DIR
        is_test = False
    elif split == "val":
        meta_path = VAL_METADATA_PATH
        data_dir = VISION_VAL_CACHE_DIR
        is_test = False
    elif split == "test":
        meta_path = TEST_METADATA_PATH
        data_dir = VISION_TEST_CACHE_DIR
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}")

    # 2. Compute Global Max (Always from TRAIN metadata)
    # This ensures normalization is consistent across all splits based on training distribution
    global_max = compute_global_max(
        metadata_path=TRAIN_METADATA_PATH,
        cache_path=VISION_GLOBAL_MAX_CACHE,
        load_cached_data=load_cached_data,
    )

    # 3. Generate Spectrograms for the requested split
    generate_vision_dataset(
        metadata_path=meta_path,
        output_dir=data_dir,
        global_max=global_max,
        load_cached_data=load_cached_data,
    )

    # 4. Load Metadata for Dataset indexing
    df_meta = pd.read_csv(meta_path)

    # 5. Create Dataset
    dataset = VolcanoSpectrogramDataset(
        metadata=df_meta, data_dir=data_dir, is_test=is_test
    )

    return dataset
