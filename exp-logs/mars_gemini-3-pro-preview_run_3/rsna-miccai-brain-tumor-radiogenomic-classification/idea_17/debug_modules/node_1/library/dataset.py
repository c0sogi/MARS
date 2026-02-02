import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_dicom_volume, generate_strided_indices


def process_patient_volume(row):
    """
    Process a single patient: load 4 modalities, normalize, extract strided views.
    Returns: numpy array of shape (2, 64, 256, 256)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    views_a = []
    views_b = []

    for mod in modalities:
        # Get paths from dataframe row
        paths = row[f"{mod}_paths"]
        if paths is None:
            paths = []

        # 1. Load Volume (Global Volumetric Normalization happens here)
        # Shape: (Depth, 256, 256)
        volume = load_dicom_volume(paths)

        # 2. Generate Indices
        # Note: volume.shape[0] equals the number of files processed (with black slices for errors)
        num_files = volume.shape[0]
        indices = generate_strided_indices(num_files)

        # 3. Extract Slices for View A and View B
        if num_files == 0:
            # Handle empty volume case with zeros
            slice_shape = (Config.SLICES_PER_VIEW, Config.IMG_SIZE, Config.IMG_SIZE)
            mod_view_a = np.zeros(slice_shape, dtype=np.float32)
            mod_view_b = np.zeros(slice_shape, dtype=np.float32)
        else:
            # View A
            idx_a = indices["view_a"]
            if len(idx_a) > 0:
                mod_view_a = volume[idx_a]
            else:
                mod_view_a = np.zeros(
                    (Config.SLICES_PER_VIEW, Config.IMG_SIZE, Config.IMG_SIZE),
                    dtype=np.float32,
                )

            # View B
            idx_b = indices["view_b"]
            if len(idx_b) > 0:
                mod_view_b = volume[idx_b]
            else:
                mod_view_b = np.zeros(
                    (Config.SLICES_PER_VIEW, Config.IMG_SIZE, Config.IMG_SIZE),
                    dtype=np.float32,
                )

        views_a.append(mod_view_a)
        views_b.append(mod_view_b)

    # Stack modalities along the channel dimension
    # Each view list contains 4 arrays of shape (16, 256, 256)
    # Concatenating gives (64, 256, 256)
    tensor_a = np.concatenate(views_a, axis=0)
    tensor_b = np.concatenate(views_b, axis=0)

    # Stack views: (2, 64, 256, 256)
    return np.stack([tensor_a, tensor_b], axis=0)


def load_and_cache_data(df, name, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    Strictly follows the caching logic requirement.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X = os.path.join(cache_dir, f"cached_{name}_X.npy")
    path_y = os.path.join(cache_dir, f"cached_{name}_y.npy")
    path_ids = os.path.join(cache_dir, f"cached_{name}_ids.npy")

    # 1. Try to load cached data
    if (
        load_cached_data
        and os.path.exists(path_X)
        and os.path.exists(path_y)
        and os.path.exists(path_ids)
    ):
        print(f"Loading cached {name} data from {cache_dir}...")
        try:
            X = np.load(path_X)
            y = np.load(path_y)
            ids = np.load(path_ids, allow_pickle=True)
            return X, y, ids
        except Exception as e:
            print(f"Error loading cache: {e}. Reprocessing...")

    # 2. Process from scratch if cache missing or load failed
    print(f"Processing {name} data (Cache miss or force reload)...")

    X_list = []
    y_list = []
    ids_list = []

    has_target = "MGMT_value" in df.columns

    for idx, row in df.iterrows():
        # Process Volume -> (2, 64, 256, 256)
        x_np = process_patient_volume(row)
        X_list.append(x_np)

        # Process ID
        ids_list.append(str(row["BraTS21ID"]))

        # Process Target
        if has_target:
            y_list.append(row["MGMT_value"])
        else:
            y_list.append(-1.0)  # Dummy for test set

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list, dtype=str)

    # 3. Save to cache
    print(f"Saving {name} data to cache...")
    np.save(path_X, X)
    np.save(path_y, y)
    np.save(path_ids, ids)

    return X, y, ids


class SSVEDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.mode = mode

        # Determine Metadata Path
        if mode == "train":
            meta_path = Config.TRAIN_META_PATH
        elif mode == "val":
            meta_path = Config.VAL_META_PATH
        elif mode == "test":
            meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Load Metadata DataFrame
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_parquet(meta_path)

        # Load Data (Cached or Processed)
        self.X, self.y, self.ids = load_and_cache_data(df, mode, load_cached_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # self.X shape: (N, 2, 64, 256, 256)
        # self.y shape: (N,)

        if self.mode == "train":
            # Stochastic View Selection
            # Randomly select 0 (View A) or 1 (View B)
            view_idx = np.random.randint(0, 2)

            img = self.X[idx][view_idx]  # Shape: (64, 256, 256)
            target = self.y[idx]

            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )

        else:
            # Validation / Test
            # Return both views for Multi-View Ensemble
            # Output shape: (2, 64, 256, 256)
            img = self.X[idx]
            target = self.y[idx]

            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )

    def get_ids(self):
        """Returns the list of BraTS21IDs in the dataset."""
        return self.ids
