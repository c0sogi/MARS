import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from library.config import Config
from library.utils import load_dicom_image


class BraTSDataset(Dataset):
    """
    Dataset class that handles the loading of MRI volumes from DICOM files.
    Implements Uniform Volumetric Sampling and Modality Stacking.
    """

    def __init__(self, df, is_test=False):
        self.df = df
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def _get_sorted_paths(self, paths):
        """
        Sorts file paths based on the integer slice number in the filename.
        Assumes format '.../Image-{number}.dcm'.
        """

        def get_slice_number(path):
            # Extract the number at the end of the filename
            match = re.search(r"Image-(\d+)\.dcm$", path)
            if match:
                return int(match.group(1))
            return 0

        # Sort paths based on the extracted number
        return sorted(paths, key=get_slice_number)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        braTS21ID = row["BraTS21ID"]

        # Retrieve paths for each modality, default to empty list if None
        flair_paths = row["flair_paths"] if row["flair_paths"] is not None else []
        t1w_paths = row["t1w_paths"] if row["t1w_paths"] is not None else []
        t1wce_paths = row["t1wce_paths"] if row["t1wce_paths"] is not None else []
        t2w_paths = row["t2w_paths"] if row["t2w_paths"] is not None else []

        # Sort paths numerically to ensure correct volume ordering
        flair_paths = self._get_sorted_paths(flair_paths)
        t1w_paths = self._get_sorted_paths(t1w_paths)
        t1wce_paths = self._get_sorted_paths(t1wce_paths)
        t2w_paths = self._get_sorted_paths(t2w_paths)

        modalities = [flair_paths, t1w_paths, t1wce_paths, t2w_paths]
        bag = []

        # Uniform Volumetric Sampling
        # Select 32 relative positions between 10% and 90% of the volume depth
        relative_positions = np.linspace(0.1, 0.9, Config.NUM_SLICES)

        for pos in relative_positions:
            slice_channels = []
            for mod_paths in modalities:
                if len(mod_paths) == 0:
                    # Handle missing modality with black image
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                else:
                    # Select slice at the relative depth
                    slice_idx = int(pos * len(mod_paths))
                    # Clamp index to be safe
                    slice_idx = min(slice_idx, len(mod_paths) - 1)

                    img_path = mod_paths[slice_idx]
                    img = load_dicom_image(img_path, Config.IMG_SIZE)

                slice_channels.append(img)

            # Stack modalities for this slice -> Shape: (4, 256, 256)
            slice_tensor = np.stack(slice_channels, axis=0)
            bag.append(slice_tensor)

        # Stack all slices -> Shape: (32, 4, 256, 256)
        X = np.stack(bag, axis=0).astype(np.float32)

        if self.is_test:
            # For test set, return input and ID
            return torch.from_numpy(X), braTS21ID
        else:
            # For train/val, return input and label
            y = row["MGMT_value"]
            return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


def _generate_and_collect_data(dataset):
    """
    Iterates over the dataset using a DataLoader to leverage multiprocessing,
    collecting all data into numpy arrays.
    """
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
    )

    all_X = []
    all_y = []

    print(f"Processing {len(dataset)} samples with {Config.NUM_WORKERS} workers...")

    for batch in loader:
        X, y = batch
        all_X.append(X.numpy())

        if dataset.is_test:
            # y is a tuple of IDs (strings)
            all_y.extend(y)
        else:
            # y is a tensor of labels
            all_y.append(y.numpy())

    # Concatenate all batches
    all_X = np.concatenate(all_X, axis=0)

    if dataset.is_test:
        all_y = np.array(all_y)
    else:
        all_y = np.concatenate(all_y, axis=0)

    return all_X, all_y


def get_train_val_datasets(load_cached_data=True):
    """
    Returns TensorDatasets for training and validation.
    Uses caching to avoid re-processing DICOMs on subsequent runs.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Check if cache exists
    cache_exists = (
        os.path.exists(Config.CACHE_TRAIN_X)
        and os.path.exists(Config.CACHE_TRAIN_Y)
        and os.path.exists(Config.CACHE_VAL_X)
        and os.path.exists(Config.CACHE_VAL_Y)
    )

    if load_cached_data and cache_exists:
        print("Loading cached train/val data...")
        train_X = np.load(Config.CACHE_TRAIN_X)
        train_y = np.load(Config.CACHE_TRAIN_Y)
        val_X = np.load(Config.CACHE_VAL_X)
        val_y = np.load(Config.CACHE_VAL_Y)
    else:
        print("Generating train/val data from scratch...")
        # Load metadata
        train_df = pd.read_parquet(Config.TRAIN_META_PATH)
        val_df = pd.read_parquet(Config.VAL_META_PATH)

        # Debugging: subset data if requested
        if Config.DEBUG:
            print(f"DEBUG mode: Limiting to {Config.DEBUG_SAMPLE_SIZE} samples.")
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Create raw datasets
        train_ds_raw = BraTSDataset(train_df, is_test=False)
        val_ds_raw = BraTSDataset(val_df, is_test=False)

        # Process and collect data
        train_X, train_y = _generate_and_collect_data(train_ds_raw)
        val_X, val_y = _generate_and_collect_data(val_ds_raw)

        # Save to cache
        print("Saving data to cache...")
        np.save(Config.CACHE_TRAIN_X, train_X)
        np.save(Config.CACHE_TRAIN_Y, train_y)
        np.save(Config.CACHE_VAL_X, val_X)
        np.save(Config.CACHE_VAL_Y, val_y)

    # Create efficient TensorDatasets
    train_dataset = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y))

    return train_dataset, val_dataset


class TestTensorDataset(Dataset):
    """Simple wrapper to return X and ID for the test set."""

    def __init__(self, X, ids):
        self.X = X
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.ids[idx]


def get_test_dataset(load_cached_data=True):
    """
    Returns a dataset for the test set.
    Uses caching to avoid re-processing.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_exists = os.path.exists(Config.CACHE_TEST_X) and os.path.exists(
        Config.CACHE_TEST_IDS
    )

    if load_cached_data and cache_exists:
        print("Loading cached test data...")
        test_X = np.load(Config.CACHE_TEST_X)
        test_ids = np.load(Config.CACHE_TEST_IDS, allow_pickle=True)
    else:
        print("Generating test data from scratch...")
        test_df = pd.read_parquet(Config.TEST_META_PATH)

        if Config.DEBUG:
            print(f"DEBUG mode: Limiting to {Config.DEBUG_SAMPLE_SIZE} samples.")
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        test_ds_raw = BraTSDataset(test_df, is_test=True)
        test_X, test_ids = _generate_and_collect_data(test_ds_raw)

        print("Saving test data to cache...")
        np.save(Config.CACHE_TEST_X, test_X)
        np.save(Config.CACHE_TEST_IDS, test_ids)

    return TestTensorDataset(test_X, test_ids)
