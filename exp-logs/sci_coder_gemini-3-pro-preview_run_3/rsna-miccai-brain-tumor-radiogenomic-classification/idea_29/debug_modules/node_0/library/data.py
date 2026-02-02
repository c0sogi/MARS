import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
from library.config import Config


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Input data of shape (N, C, H, W).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): BraTS21IDs corresponding to the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Data is stored as float32, normalized to [0, 1]
        img = self.X[idx]

        # Convert to tensor
        img_tensor = torch.from_numpy(img)

        # Apply optional transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label

        return img_tensor


def get_sorted_file_paths(file_paths):
    """
    Sorts file paths based on the integer index in the filename (e.g., Image-10.dcm).
    """

    def extract_number(path):
        # Match 'Image-10.dcm' or similar patterns to extract the slice number
        match = re.search(r"(\d+)\.dcm$", path)
        if match:
            return int(match.group(1))
        return 0

    return sorted(file_paths, key=extract_number)


def load_and_process_modality(base_dir, paths, num_slices, img_size):
    """
    Processes a single modality:
    1. Sorts paths by slice index.
    2. Uniformly samples 'num_slices' from the 10%-90% volume depth.
    3. Reads DICOMs, resizes to (img_size, img_size).
    4. Applies View-Adaptive Normalization (min-max on the selected subset).
    """
    # 1. Sort
    sorted_paths = get_sorted_file_paths(paths)
    total_files = len(sorted_paths)

    # Handle empty modality case
    if total_files == 0:
        return np.zeros((num_slices, img_size, img_size), dtype=np.float32)

    # 2. High-Density Uniform Sampling (10% - 90%)
    # We focus on the center of the brain where the tumor is most likely to be
    start_idx = int(total_files * 0.1)
    end_idx = int(total_files * 0.9)

    # Fallback if the range is too small
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = total_files

    # Generate indices
    # We use linspace to pick exactly 'num_slices' indices
    if total_files > 0:
        indices = np.linspace(
            start_idx, max(end_idx - 1, start_idx), num_slices
        ).astype(int)
        selected_paths = [sorted_paths[i] for i in indices]
    else:
        selected_paths = []

    # 3. Read and Resize
    slices = []
    for rel_path in selected_paths:
        full_path = os.path.join(base_dir, rel_path)
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)
            img = cv2.resize(img, (img_size, img_size))
            slices.append(img)
        except Exception:
            # Fallback for corrupt file: use zero slice
            slices.append(np.zeros((img_size, img_size), dtype=np.float32))

    # If we somehow didn't get enough slices (e.g., all files corrupt), pad
    while len(slices) < num_slices:
        slices.append(np.zeros((img_size, img_size), dtype=np.float32))

    stack = np.array(slices, dtype=np.float32)  # Shape: (num_slices, H, W)

    # 4. View-Adaptive Per-Modality Normalization
    # Calculate min/max ONLY on this specific stack of 32 slices
    min_val = np.min(stack)
    max_val = np.max(stack)

    if max_val - min_val > 0:
        stack = (stack - min_val) / (max_val - min_val)
    else:
        stack = stack - min_val  # Should be all zeros if max == min

    return stack


def process_subject(row, input_dir, num_slices, img_size):
    """
    Loads all 4 modalities for a subject and stacks them into a single volume.
    Output shape: (4 * num_slices, img_size, img_size)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    modality_stacks = []

    for mod in modalities:
        # Retrieve paths from dataframe column (e.g., 'flair_paths')
        paths = row.get(f"{mod}_paths", [])
        if paths is None:
            paths = []
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()

        stack = load_and_process_modality(input_dir, paths, num_slices, img_size)
        modality_stacks.append(stack)

    # Stack modalities along channel dimension
    # Result: (128, 224, 224)
    full_volume = np.concatenate(modality_stacks, axis=0)
    return full_volume


def get_datasets(load_cached_data=True):
    """
    Prepares Train, Validation, and Test datasets.
    Implements caching to disk to save time on subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "train_X": os.path.join(cache_dir, "X_train.npy"),
        "train_y": os.path.join(cache_dir, "y_train.npy"),
        "train_ids": os.path.join(cache_dir, "ids_train.npy"),
        "val_X": os.path.join(cache_dir, "X_val.npy"),
        "val_y": os.path.join(cache_dir, "y_val.npy"),
        "val_ids": os.path.join(cache_dir, "ids_val.npy"),
        "test_X": os.path.join(cache_dir, "X_test.npy"),
        "test_ids": os.path.join(cache_dir, "ids_test.npy"),
    }

    # Check if cache exists
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print(f"Loading datasets from cache at {cache_dir}...")
        X_train = np.load(cache_files["train_X"])
        y_train = np.load(cache_files["train_y"])
        ids_train = np.load(cache_files["train_ids"])

        X_val = np.load(cache_files["val_X"])
        y_val = np.load(cache_files["val_y"])
        ids_val = np.load(cache_files["val_ids"])

        X_test = np.load(cache_files["test_X"])
        ids_test = np.load(cache_files["test_ids"])

    else:
        print("Processing datasets from scratch (this may take a while)...")

        # Load Metadata
        df_train = pd.read_parquet(Config.TRAIN_META_PATH)
        df_val = pd.read_parquet(Config.VAL_META_PATH)
        df_test = pd.read_parquet(Config.TEST_META_PATH)

        # Apply Debug sampling if configured
        if Config.DEBUG:
            print(f"DEBUG MODE: limiting to {Config.DEBUG_SAMPLE_SIZE} samples.")
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

        def process_dataframe(df, has_labels=True):
            X_list = []
            y_list = []
            ids_list = []

            total = len(df)
            for idx, row in df.iterrows():
                if idx % 50 == 0:
                    print(f"  Processed {idx}/{total} subjects")

                vol = process_subject(
                    row,
                    Config.INPUT_DIR,
                    Config.NUM_SLICES_PER_MODALITY,
                    Config.IMG_SIZE,
                )

                X_list.append(vol)
                ids_list.append(str(row["BraTS21ID"]))

                if has_labels:
                    y_list.append(row["MGMT_value"])

            X = np.array(X_list, dtype=np.float32)
            ids = np.array(ids_list)
            y = np.array(y_list, dtype=np.float32) if has_labels else None

            return X, y, ids

        print("Processing Training Set...")
        X_train, y_train, ids_train = process_dataframe(df_train, has_labels=True)

        print("Processing Validation Set...")
        X_val, y_val, ids_val = process_dataframe(df_val, has_labels=True)

        print("Processing Test Set...")
        X_test, _, ids_test = process_dataframe(df_test, has_labels=False)

        # Save to cache
        print("Saving processed data to cache...")
        np.save(cache_files["train_X"], X_train)
        np.save(cache_files["train_y"], y_train)
        np.save(cache_files["train_ids"], ids_train)

        np.save(cache_files["val_X"], X_val)
        np.save(cache_files["val_y"], y_val)
        np.save(cache_files["val_ids"], ids_val)

        np.save(cache_files["test_X"], X_test)
        np.save(cache_files["test_ids"], ids_test)

    # Instantiate Datasets
    train_dataset = BraTSDataset(X_train, y_train, ids_train)
    val_dataset = BraTSDataset(X_val, y_val, ids_val)
    test_dataset = BraTSDataset(X_test, None, ids_test)

    return train_dataset, val_dataset, test_dataset
