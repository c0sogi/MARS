import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from library.config import Config
from library.utils import (
    read_dicom_file,
    normalize_min_max,
    get_depth_indices,
    set_seed,
)


class BraTSDataset(Dataset):
    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (np.ndarray): Input data of shape (N, 64, 256, 256).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): BraTS21IDs of shape (N,).
        """
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X[idx] is (64, 256, 256)
        x_data = self.X[idx]

        # Convert to torch tensor
        x_tensor = torch.from_numpy(x_data).float()

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return x_tensor, label
        else:
            # For inference, return the ID to map predictions
            patient_id = self.ids[idx] if self.ids is not None else ""
            return x_tensor, patient_id


def process_patient(row, img_size, num_slices):
    """
    Processes a single patient's data:
    1. Loads full volumes for all 4 modalities.
    2. Normalizes each modality globally.
    3. Samples 16 slices uniformly.
    4. Stacks into a single 2.5D volume (Early Fusion).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    patient_slices = []

    for mod in modalities:
        paths = row.get(f"{mod}_paths", [])

        # Load all available slices for the modality
        volume_imgs = []
        if paths is not None and len(paths) > 0:
            for p in paths:
                img = read_dicom_file(p)
                volume_imgs.append(img)

        # Handle missing data or empty folders
        if len(volume_imgs) == 0:
            # Create a dummy volume with 1 slice to allow code to proceed
            volume = np.zeros((1, img_size, img_size), dtype=np.float32)
        else:
            volume = np.stack(volume_imgs)  # (D, H, W)

        # Global Volumetric Normalization (per modality)
        volume = normalize_min_max(volume)

        # Uniform Sampling (10%-90% depth)
        indices = get_depth_indices(len(volume), num_slices)
        sampled_volume = volume[indices]  # (16, H, W)

        patient_slices.append(sampled_volume)

    # Stack modalities: (4, 16, H, W) -> Transpose to (16, 4, H, W)
    stack = np.stack(patient_slices, axis=1)  # (16, 4, H, W)

    # Flatten to Channels: (16, 4, H, W) -> (64, H, W)
    # Early Fusion: All slices and modalities are stacked in the channel dimension
    volumetric_input = stack.reshape(-1, img_size, img_size)

    return volumetric_input


def prepare_datasets(load_cached_data=True, num_workers=Config.NUM_WORKERS):
    """
    Loads data, processes it into the Dual-Stream format, and caches it.
    Returns Train, Val, and Test datasets.
    """
    set_seed(Config.SEED)

    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths for cached files
    files = {
        "train_X": os.path.join(cache_dir, "cached_train_X.npy"),
        "train_y": os.path.join(cache_dir, "cached_train_y.npy"),
        "train_ids": os.path.join(cache_dir, "cached_train_ids.npy"),
        "val_X": os.path.join(cache_dir, "cached_val_X.npy"),
        "val_y": os.path.join(cache_dir, "cached_val_y.npy"),
        "val_ids": os.path.join(cache_dir, "cached_val_ids.npy"),
        "test_X": os.path.join(cache_dir, "cached_test_X.npy"),
        "test_ids": os.path.join(cache_dir, "cached_test_ids.npy"),
    }

    # Check existence
    all_exist = all(os.path.exists(p) for p in files.values())

    if load_cached_data and all_exist:
        print(f"Loading cached datasets from {cache_dir}...")
        train_X = np.load(files["train_X"])
        train_y = np.load(files["train_y"])
        train_ids = np.load(files["train_ids"])

        val_X = np.load(files["val_X"])
        val_y = np.load(files["val_y"])
        val_ids = np.load(files["val_ids"])

        test_X = np.load(files["test_X"])
        test_ids = np.load(files["test_ids"])

    else:
        print("Processing datasets from scratch (this may take a while)...")

        # Load Metadata
        train_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "val.parquet"))
        test_df = pd.read_parquet(os.path.join(Config.METADATA_DIR, "test.parquet"))

        def process_subset(df, is_test=False):
            # Convert to list of dicts for efficient pickling in Parallel
            records = df.to_dict("records")

            # Process in parallel
            results = Parallel(n_jobs=num_workers, verbose=0)(
                delayed(process_patient)(row, Config.IMG_SIZE, Config.NUM_SLICES_TOTAL)
                for row in records
            )

            X = np.stack(results)  # (N, 64, 256, 256)

            ids = df["BraTS21ID"].values.astype(str)

            if not is_test:
                y = df["MGMT_value"].values.astype(np.float32)
            else:
                y = None

            return X, y, ids

        print("Processing Train...")
        train_X, train_y, train_ids = process_subset(train_df)

        print("Processing Val...")
        val_X, val_y, val_ids = process_subset(val_df)

        print("Processing Test...")
        test_X, _, test_ids = process_subset(test_df, is_test=True)

        print("Saving to cache...")
        np.save(files["train_X"], train_X)
        np.save(files["train_y"], train_y)
        np.save(files["train_ids"], train_ids)

        np.save(files["val_X"], val_X)
        np.save(files["val_y"], val_y)
        np.save(files["val_ids"], val_ids)

        np.save(files["test_X"], test_X)
        np.save(files["test_ids"], test_ids)

    # Initialize Datasets
    train_dataset = BraTSDataset(train_X, train_y, train_ids)
    val_dataset = BraTSDataset(val_X, val_y, val_ids)
    test_dataset = BraTSDataset(test_X, None, test_ids)

    return train_dataset, val_dataset, test_dataset
