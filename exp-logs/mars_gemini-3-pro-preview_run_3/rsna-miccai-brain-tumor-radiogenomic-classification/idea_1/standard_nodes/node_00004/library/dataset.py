import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import (
    load_dicom,
    normalize_pixels,
    resize_image,
    select_uniform_indices,
)


def process_patient_images(row):
    """
    Reads images for a single patient, processes them, and stacks them.
    Returns a numpy array of shape (C, H, W).
    """
    channels = []

    for mod in Config.MODALITIES:
        # Map modality name to dataframe column name (e.g., FLAIR -> flair_paths)
        col_name = f"{mod.lower()}_paths"
        paths = row[col_name] if col_name in row and row[col_name] is not None else []

        # Select indices of slices to use via uniform sampling Cite solution_lesson_node_00003
        indices = select_uniform_indices(len(paths), Config.NUM_SLICES)

        loaded_slices_count = 0

        for idx in indices:
            rel_path = paths[idx]
            img = load_dicom(rel_path)

            # Resize and Normalize
            img = resize_image(img, Config.IMAGE_SIZE)
            img = normalize_pixels(img)

            if img is None:
                # Fallback: zero array
                img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

            channels.append(img)
            loaded_slices_count += 1

        # Pad with zeros if fewer slices were found/loaded than expected
        while loaded_slices_count < Config.NUM_SLICES:
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
            channels.append(img)
            loaded_slices_count += 1

    # Stack channels. Resulting shape: (C, H, W)
    if len(channels) == 0:
        # Safety fallback
        return np.zeros(
            (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    X_patient = np.array(channels, dtype=np.float32)
    return X_patient


def load_data(df, split_name, load_cached_data=True):
    """
    Loads dataset tensors, utilizing a caching mechanism to store processed arrays.

    Args:
        df (pd.DataFrame): Metadata dataframe containing file paths and targets.
        split_name (str): Identifier for the split (e.g., 'train', 'val', 'test').
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (X, y, ids) where X is image tensor, y is target (or None), ids is list of BraTS21IDs.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    x_path = os.path.join(cache_dir, f"cached_{split_name}_X.npy")
    y_path = os.path.join(cache_dir, f"cached_{split_name}_y.npy")
    ids_path = os.path.join(cache_dir, f"cached_{split_name}_ids.npy")

    has_targets = "MGMT_value" in df.columns

    # 1. Try to load from cache
    if load_cached_data:
        # Check if X and ids exist. If targets are expected, check y too.
        cache_valid = os.path.exists(x_path) and os.path.exists(ids_path)
        if has_targets and not os.path.exists(y_path):
            cache_valid = False

        if cache_valid:
            try:
                X = np.load(x_path)
                ids = np.load(ids_path, allow_pickle=True)
                y = np.load(y_path) if has_targets else None
                return X, y, ids
            except Exception:
                # If load fails, proceed to re-process
                pass

    # 2. Process data from scratch
    X_list = []
    y_list = []
    ids_list = []

    # Iterate through dataframe
    for _, row in df.iterrows():
        # Process Images
        img_tensor = process_patient_images(row)
        X_list.append(img_tensor)

        # Store ID
        ids_list.append(str(row["BraTS21ID"]))

        # Store Target
        if has_targets:
            y_list.append(row["MGMT_value"])

    # Convert to numpy arrays
    if len(X_list) > 0:
        X = np.array(X_list, dtype=np.float32)
        ids = np.array(ids_list)
        y = np.array(y_list, dtype=np.float32) if has_targets else None
    else:
        # Handle empty dataset case
        X = np.zeros(
            (0, Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE),
            dtype=np.float32,
        )
        ids = np.array([])
        y = np.array([], dtype=np.float32) if has_targets else None

    # 3. Save to cache
    np.save(x_path, X)
    np.save(ids_path, ids)
    if has_targets:
        np.save(y_path, y)

    return X, y, ids


class MGMTDataset(Dataset):
    def __init__(self, df, split_name="train", load_cached_data=True, transform=None):
        """
        PyTorch Dataset for MGMT Promoter Methylation prediction.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            split_name (str): Name of the data split (used for caching).
            load_cached_data (bool): Whether to use cached .npy files.
            transform (callable, optional): Optional transform to apply to samples.
        """
        self.transform = transform
        self.X, self.y, self.ids = load_data(df, split_name, load_cached_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve image and convert to torch tensor
        image = self.X[idx]  # Shape (C, H, W)
        image_tensor = torch.from_numpy(image)

        # Apply transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Return tuple (image, target) if target exists, else just image
        if self.y is not None:
            target = self.y[idx]
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return image_tensor, target_tensor
        else:
            return image_tensor

    def get_ids(self):
        """Returns the list of BraTS21IDs corresponding to the dataset indices."""
        return self.ids
