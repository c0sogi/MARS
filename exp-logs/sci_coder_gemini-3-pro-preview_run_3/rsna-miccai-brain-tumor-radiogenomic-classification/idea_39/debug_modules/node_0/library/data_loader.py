import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def load_dicom_slice(path):
    """
    Reads a DICOM file and returns the pixel array as a float32 numpy array.
    Returns None if the file cannot be read.
    """
    try:
        # force=True allows reading files even if they are missing the header preamble
        ds = pydicom.dcmread(path, force=True)
        img = ds.pixel_array.astype(np.float32)
        return img
    except Exception:
        return None


def process_patient_volume(row, input_dir):
    """
    Processes a single patient's MRI data according to SSF-Net requirements:
    1. Sorts files by integer index.
    2. Samples 32 slices uniformly from 10-90% depth.
    3. Normalizes based on the statistics of the selected subset.
    4. Splits into Even and Odd streams.
    5. Stacks by modality.

    Returns:
        X_even (np.ndarray): Shape (64, 224, 224)
        X_odd (np.ndarray): Shape (64, 224, 224)
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    streams = {"even": [], "odd": []}

    target_size = (Config.IMG_SIZE, Config.IMG_SIZE)
    total_slices = Config.TOTAL_SLICES  # 32

    for mod in modalities:
        # Retrieve paths from metadata
        paths = row.get(f"{mod}_paths", [])
        if paths is None:
            paths = []

        # 1. External Integer Sorting
        try:
            paths = sorted(
                paths,
                key=lambda x: int(os.path.basename(x).split("-")[-1].split(".")[0]),
            )
        except Exception:
            paths = sorted(paths)

        # 2. High-Density Uniform Sampling
        depth = len(paths)
        if depth == 0:
            # Handle missing modality
            mod_volume = np.zeros((total_slices, *target_size), dtype=np.float32)
        else:
            # Define 10%-90% range
            start_idx = int(depth * 0.1)
            end_idx = int(depth * 0.9)

            # Fallback for small volumes
            if end_idx <= start_idx:
                start_idx = 0
                end_idx = depth

            # Generate uniform indices
            indices = np.linspace(
                start_idx, max(start_idx, end_idx - 1), total_slices, dtype=int
            )

            selected_slices = []
            for idx in indices:
                full_path = os.path.join(input_dir, paths[idx])
                img = load_dicom_slice(full_path)

                if img is None:
                    img = np.zeros(target_size, dtype=np.float32)
                else:
                    # Resize to 224x224
                    if img.shape != target_size:
                        img = cv2.resize(
                            img, target_size, interpolation=cv2.INTER_CUBIC
                        )

                selected_slices.append(img)

            mod_volume = np.array(selected_slices, dtype=np.float32)

            # 3. Subset-Adaptive Per-Modality Normalization
            v_min = mod_volume.min()
            v_max = mod_volume.max()
            if v_max - v_min > 0:
                mod_volume = (mod_volume - v_min) / (v_max - v_min)
            else:
                mod_volume[:] = 0.0

        # 4. Deterministic Strided Splitting
        # Even indices: 0, 2, ... | Odd indices: 1, 3, ...
        even_slices = mod_volume[0::2]  # 16 slices
        odd_slices = mod_volume[1::2]  # 16 slices

        streams["even"].append(even_slices)
        streams["odd"].append(odd_slices)

    # 5. Modality-Grouped Stacking
    # Concatenate along the channel dimension (axis 0)
    # Result: [FLAIR(16), T1w(16), T1wCE(16), T2w(16)] -> (64, 224, 224)
    X_even = np.concatenate(streams["even"], axis=0)
    X_odd = np.concatenate(streams["odd"], axis=0)

    return X_even, X_odd


def load_and_cache_data(meta_path, cache_name, load_cached_data=True):
    """
    Loads dataset from metadata. Uses caching to speed up subsequent runs.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_X_even = os.path.join(cache_dir, f"{cache_name}_X_even.npy")
    path_X_odd = os.path.join(cache_dir, f"{cache_name}_X_odd.npy")
    path_y = os.path.join(cache_dir, f"{cache_name}_y.npy")
    path_ids = os.path.join(cache_dir, f"{cache_name}_ids.npy")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(path_X_even)
        and os.path.exists(path_X_odd)
        and os.path.exists(path_ids)
    ):
        # print(f"Loading {cache_name} data from cache...")
        X_even = np.load(path_X_even)
        X_odd = np.load(path_X_odd)
        ids = np.load(path_ids)
        y = np.load(path_y) if os.path.exists(path_y) else None
        return X_even, X_odd, y, ids

    # Process from scratch
    # print(f"Processing {cache_name} data from scratch...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_parquet(meta_path)

    X_even_list = []
    X_odd_list = []
    y_list = []
    ids_list = []

    has_labels = "MGMT_value" in df.columns

    for _, row in df.iterrows():
        pid = row["BraTS21ID"]
        xe, xo = process_patient_volume(row, Config.INPUT_DIR)

        X_even_list.append(xe)
        X_odd_list.append(xo)
        ids_list.append(pid)

        if has_labels:
            y_list.append(row["MGMT_value"])

    X_even = np.array(X_even_list, dtype=np.float32)
    X_odd = np.array(X_odd_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    np.save(path_X_even, X_even)
    np.save(path_X_odd, X_odd)
    np.save(path_ids, ids)

    if has_labels:
        y = np.array(y_list, dtype=np.float32)
        np.save(path_y, y)
    else:
        y = None

    return X_even, X_odd, y, ids


class BraTSDataset(Dataset):
    def __init__(self, X_even, X_odd, y=None, ids=None):
        self.X_even = X_even
        self.X_odd = X_odd
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X_even)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        even = torch.from_numpy(self.X_even[idx])
        odd = torch.from_numpy(self.X_odd[idx])

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return (even, odd), label
        else:
            # Return dummy label for test set
            return (even, odd), torch.tensor(0.0, dtype=torch.float32)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Generates DataLoaders for Train, Validation, and Test sets.
    """
    seed_everything(Config.SEED)

    # Train Loader
    X_tr_e, X_tr_o, y_tr, _ = load_and_cache_data(
        Config.TRAIN_META_PATH, "train", load_cached_data
    )
    train_dataset = BraTSDataset(X_tr_e, X_tr_o, y_tr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Validation Loader
    X_val_e, X_val_o, y_val, _ = load_and_cache_data(
        Config.VAL_META_PATH, "val", load_cached_data
    )
    val_dataset = BraTSDataset(X_val_e, X_val_o, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Test Loader
    X_te_e, X_te_o, _, te_ids = load_and_cache_data(
        Config.TEST_META_PATH, "test", load_cached_data
    )
    test_dataset = BraTSDataset(X_te_e, X_te_o, ids=te_ids)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
