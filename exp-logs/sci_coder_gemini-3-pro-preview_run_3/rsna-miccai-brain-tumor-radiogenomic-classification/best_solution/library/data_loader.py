import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from library.utils import seed_everything

# Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_opt"
IMG_SIZE = 224
NUM_SLICES_FULL = 16
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]


class BraTSDataset(Dataset):
    def __init__(self, images, labels=None, ids=None, mode="train"):
        """
        Args:
            images (np.ndarray): Shape (N, 4, 32, H, W)
            labels (np.ndarray): Shape (N,)
            ids (np.ndarray): Shape (N,)
            mode (str): 'train', 'val', or 'test'
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Shape: (4, 16, H, W)
        vol = self.images[idx]

        # Flatten channels: (4 * 16, H, W) -> (64, H, W)
        # We stack depth-wise per modality: [M1_S1, M1_S2... M2_S1...]
        vol = vol.reshape(-1, IMG_SIZE, IMG_SIZE)

        # Convert to float tensor
        vol_tensor = torch.from_numpy(vol).float()

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return vol_tensor, label
        else:
            return vol_tensor


def load_dicom_volume(paths):
    """
    Loads a volume of MRI slices from a list of paths.
    Applies sorting, uniform sampling (32 slices), resizing, and normalization.
    """
    # 1. Sort paths numerically by instance number (Image-X.dcm)
    try:
        paths = sorted(paths, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    except Exception:
        paths = sorted(paths)

    # 2. Uniform Sampling (10% - 90% depth range)
    n_files = len(paths)
    if n_files == 0:
        return np.zeros((NUM_SLICES_FULL, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    if n_files < NUM_SLICES_FULL:
        # If fewer files than needed, resample indices linearly
        indices = np.linspace(0, n_files - 1, NUM_SLICES_FULL).astype(int)
    else:
        # Exclude top/bottom 10% to focus on brain/tumor center
        start = int(n_files * 0.1)
        end = int(n_files * 0.9)
        if end <= start:
            start = 0
            end = n_files
        indices = np.linspace(start, end - 1, NUM_SLICES_FULL).astype(int)

    selected_paths = [paths[i] for i in indices]

    volume = []
    for p in selected_paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array
        except Exception:
            # Fallback for corrupt files
            img = np.zeros((IMG_SIZE, IMG_SIZE))

        # Resize to fixed resolution
        if img.shape != (IMG_SIZE, IMG_SIZE):
            try:
                img = cv2.resize(
                    img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )
            except Exception:
                img = np.zeros((IMG_SIZE, IMG_SIZE))

        volume.append(img)

    volume = np.array(volume, dtype=np.float32)  # (32, H, W)

    # 3. Min-Max Normalization per volume
    v_min = volume.min()
    v_max = volume.max()
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    return volume


def get_processed_data(df, mode, load_cached_data):
    """
    Loads data from cache or processes it from scratch.
    Returns: X (images), y (labels), ids
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_X = os.path.join(CACHE_DIR, f"cached_{mode}_X.npy")
    cache_y = os.path.join(CACHE_DIR, f"cached_{mode}_y.npy")
    cache_ids = os.path.join(CACHE_DIR, f"cached_{mode}_ids.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        print(f"Loading cached {mode} data from {CACHE_DIR}...")
        X = np.load(cache_X)
        ids = np.load(cache_ids, allow_pickle=True)

        if mode != "test":
            if os.path.exists(cache_y):
                y = np.load(cache_y)
                return X, y, ids
        else:
            return X, None, ids

    # Process from scratch
    print(f"Processing {mode} data from scratch...")
    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        pid = row["BraTS21ID"]

        # Load 4 modalities
        channels = []
        for mod in MODALITIES:  # ["flair", "t1w", "t1wce", "t2w"]
            col_name = f"{mod}_paths"
            paths = row[col_name]
            if paths is None:
                paths = []

            # Load volume for this modality
            vol = load_dicom_volume(paths)
            channels.append(vol)

        # Stack modalities: (4, 32, H, W)
        X_patient = np.stack(channels, axis=0)
        X_list.append(X_patient)
        ids_list.append(pid)

        if mode != "test":
            y_list.append(row["MGMT_value"])

    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    print(f"Saving {mode} data to cache...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)

    if mode != "test":
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y, y)
        return X, y, ids

    return X, None, ids


def get_dataloaders(batch_size=8, load_cached_data=True):
    """
    Constructs DataLoaders for Train, Val, and Test sets.
    """
    seed_everything(42)

    # Load Metadata
    train_df = pd.read_parquet("./metadata/train.parquet")
    val_df = pd.read_parquet("./metadata/val.parquet")
    test_df = pd.read_parquet("./metadata/test.parquet")

    # Process Data (Load/Cache)
    X_train, y_train, ids_train = get_processed_data(
        train_df, "train", load_cached_data
    )
    X_val, y_val, ids_val = get_processed_data(val_df, "val", load_cached_data)
    X_test, _, ids_test = get_processed_data(test_df, "test", load_cached_data)

    # Create Datasets
    train_ds = BraTSDataset(X_train, y_train, ids_train, mode="train")
    val_ds = BraTSDataset(X_val, y_val, ids_val, mode="val")
    test_ds = BraTSDataset(X_test, None, ids_test, mode="test")

    # Create Loaders
    # Pin memory and num_workers for efficiency
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, ids_test
