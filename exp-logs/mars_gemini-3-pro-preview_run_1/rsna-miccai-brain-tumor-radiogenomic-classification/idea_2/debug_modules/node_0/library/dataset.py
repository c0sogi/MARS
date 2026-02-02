import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    NUM_SLICES,
    IMAGE_SIZE,
    REMOVE_NOISE_MARGIN,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
    SEED,
)
from library.utils import get_sorted_file_paths, load_dicom_slice, set_seed


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specific phase.

    Args:
        phase (str): 'train' or 'valid'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_data_cache(df, cache_path, load_cached_data=True):
    """
    Pre-processes the file paths for the dataset.
    Scans directories, filters noise margins, and caches the result to Parquet.

    Args:
        df (pd.DataFrame): Input metadata containing subject IDs and folder paths.
        cache_path (str): Path to save/load the parquet cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with columns 'flair_paths', 't1wce_paths', 't2w_paths'
                      containing lists of valid file paths.
    """
    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        try:
            df_cached = pd.read_parquet(cache_path, engine="pyarrow")
            return df_cached
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding...")

    # 2. Rebuild cache
    print(f"Building dataset cache (saving to {cache_path})...")

    # We will create new columns for the file lists
    flair_lists = []
    t1wce_lists = []
    t2w_lists = []

    for idx, row in df.iterrows():
        # Construct full paths based on input directory and relative paths in metadata
        flair_dir = os.path.join(INPUT_DIR, row["flair_path"])
        t1wce_dir = os.path.join(INPUT_DIR, row["t1wce_path"])
        t2w_dir = os.path.join(INPUT_DIR, row["t2w_path"])

        # Helper to process one modality
        def process_modality(folder_path):
            paths = get_sorted_file_paths(folder_path)
            if not paths:
                return []

            # Remove noise margin (top and bottom X%)
            n = len(paths)
            if n < 5:  # Keep all if very few slices
                return paths

            margin = int(n * REMOVE_NOISE_MARGIN)
            # Ensure we don't remove everything
            if 2 * margin >= n:
                return paths  # Fallback

            return paths[margin : n - margin]

        flair_lists.append(process_modality(flair_dir))
        t1wce_lists.append(process_modality(t1wce_dir))
        t2w_lists.append(process_modality(t2w_dir))

    # Assign back to dataframe
    df_processed = df.copy()
    df_processed["flair_files"] = flair_lists
    df_processed["t1wce_files"] = t1wce_lists
    df_processed["t2w_files"] = t2w_lists

    # 3. Save cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        df_processed.to_parquet(cache_path, engine="pyarrow")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df_processed


class RSNADataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing file lists (from prepare_data_cache).
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.df = df
        self.transform = transform
        self.num_slices = NUM_SLICES
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get file lists
        # Note: In the parquet/pandas dataframe, these are lists of strings
        flair_files = row["flair_files"]
        t1wce_files = row["t1wce_files"]
        t2w_files = row["t2w_files"]

        # Prepare lists for sampling
        # We handle empty lists by generating black images later if needed,
        # but ideally the dataset cleaning handles this.

        # Uniform Depth Sampling
        # We want exactly NUM_SLICES.
        # We sample indices uniformly from [0, len(files)-1]

        def get_indices(num_available):
            if num_available == 0:
                return []
            if num_available < self.num_slices:
                # If fewer slices than needed, we might need to repeat or just take what we have and pad
                # For simplicity in MIL, we usually just resample with replacement or linspace
                return np.linspace(0, num_available - 1, self.num_slices).astype(int)
            else:
                return np.linspace(0, num_available - 1, self.num_slices).astype(int)

        flair_idxs = get_indices(len(flair_files))
        t1wce_idxs = get_indices(len(t1wce_files))
        t2w_idxs = get_indices(len(t2w_files))

        bag_images = []

        for i in range(self.num_slices):
            # Load FLAIR
            if len(flair_files) > 0:
                f_path = flair_files[flair_idxs[i]]
                img_f = load_dicom_slice(f_path, self.image_size)
            else:
                img_f = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            # Load T1wCE
            if len(t1wce_files) > 0:
                c_path = t1wce_files[t1wce_idxs[i]]
                img_c = load_dicom_slice(c_path, self.image_size)
            else:
                img_c = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            # Load T2w
            if len(t2w_files) > 0:
                t_path = t2w_files[t2w_idxs[i]]
                img_t = load_dicom_slice(t_path, self.image_size)
            else:
                img_t = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            # Stack to (H, W, 3)
            # load_dicom_slice returns (H, W) in range [0, 1]
            img_composite = np.stack([img_f, img_c, img_t], axis=-1)

            # Apply transforms
            # Albumentations expects uint8 or float32.
            # Our images are float32 [0, 1].
            if self.transform:
                res = self.transform(image=img_composite)
                img_tensor = res["image"]  # (3, H, W)
            else:
                # Manual to tensor if no transform provided
                img_tensor = torch.from_numpy(img_composite.transpose(2, 0, 1))

            bag_images.append(img_tensor)

        # Stack bag: (NUM_SLICES, 3, H, W)
        bag_tensor = torch.stack(bag_images)

        # Get target if available
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return bag_tensor, target
        else:
            # Test set might not have targets, return dummy or just ID
            # But standard Dataset usually returns X, y.
            # For inference, we can return a placeholder.
            return bag_tensor, torch.tensor(-1.0, dtype=torch.float32)
