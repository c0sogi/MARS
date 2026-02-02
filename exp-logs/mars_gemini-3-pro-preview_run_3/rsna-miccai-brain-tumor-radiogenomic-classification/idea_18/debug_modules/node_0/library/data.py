import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.utils import set_seed, Logger

# ==========================================
# Constants & Configuration
# ==========================================
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_18"
METADATA_DIR = "./metadata"
IMG_SIZE = 256
NUM_SLICES_TOTAL = 32
NUM_SLICES_VIEW = 16
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]


# ==========================================
# Dataset Classes
# ==========================================
class BraTSDataset(Dataset):
    """
    Dataset for Training and Validation.
    Each sample is a single view (Even or Odd) of the patient's MRI.
    Input shape: (64, 256, 256)
    """

    def __init__(self, X, y, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X[idx] is a (64, 256, 256) float32 numpy array
        img = self.X[idx]
        label = self.y[idx]

        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor


class BraTSTestDataset(Dataset):
    """
    Dataset for Testing.
    Returns both views (Even and Odd) for ensemble inference.
    """

    def __init__(self, X_even, X_odd, ids):
        self.X_even = X_even
        self.X_odd = X_odd
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_even = torch.from_numpy(self.X_even[idx]).float()
        img_odd = torch.from_numpy(self.X_odd[idx]).float()
        patient_id = self.ids[idx]

        return img_even, img_odd, patient_id


# ==========================================
# Preprocessing Functions
# ==========================================
def load_dicom_slice(path):
    """Loads a single DICOM file and returns the pixel array."""
    try:
        full_path = os.path.join(INPUT_DIR, path)
        dcm = pydicom.dcmread(full_path)
        img = dcm.pixel_array.astype(np.float32)
        return img
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def resize_slice(img):
    """Resizes image to IMG_SIZE x IMG_SIZE."""
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def normalize_modality_volume(vol):
    """
    Normalizes a 3D volume (D, H, W) using global min/max.
    """
    v_min = np.min(vol)
    v_max = np.max(vol)
    if v_max - v_min > 0:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol)
    return vol


def get_uniform_indices(total_slices, num_selected):
    """
    Selects indices uniformly from the 10%-90% range.
    """
    if total_slices < num_selected:
        start = 0
        end = total_slices - 1
    else:
        start = int(total_slices * 0.10)
        end = int(total_slices * 0.90)
        if end - start < num_selected:
            start = 0
            end = total_slices - 1

    indices = np.linspace(start, end, num_selected, dtype=int)
    indices = np.clip(indices, 0, total_slices - 1)
    return indices


def process_patient(row):
    """
    Loads data for a single patient, generates View A and View B.
    Returns: (view_a, view_b) where each is (64, 256, 256).
    """
    views_a = []
    views_b = []

    for mod in MODALITIES:
        path_col = f"{mod.lower()}_paths"
        paths = row[path_col]

        if paths is None or len(paths) == 0:
            vol_a = np.zeros((NUM_SLICES_VIEW, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            vol_b = np.zeros((NUM_SLICES_VIEW, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            views_a.append(vol_a)
            views_b.append(vol_b)
            continue

        # Sort paths numerically based on Image-XXX.dcm
        def extract_num(p):
            base = os.path.basename(p)
            name = os.path.splitext(base)[0]
            try:
                return int(name.split("-")[1])
            except:
                return 0

        sorted_paths = sorted(paths, key=extract_num)

        # Load and resize
        slices = [resize_slice(load_dicom_slice(p)) for p in sorted_paths]
        volume = np.array(slices)  # (D, 256, 256)

        # Normalize
        volume = normalize_modality_volume(volume)

        # Select 32 Indices
        total_slices = len(volume)
        indices = get_uniform_indices(total_slices, NUM_SLICES_TOTAL)
        selected_volume = volume[indices]  # (32, 256, 256)

        # Split into Views
        # View A: 0, 2, ... 30
        # View B: 1, 3, ... 31
        view_a_indices = np.arange(0, NUM_SLICES_TOTAL, 2)
        view_b_indices = np.arange(1, NUM_SLICES_TOTAL, 2)

        vol_a = selected_volume[view_a_indices]  # (16, 256, 256)
        vol_b = selected_volume[view_b_indices]  # (16, 256, 256)

        views_a.append(vol_a)
        views_b.append(vol_b)

    # Stack Modalities: (4, 16, 256, 256) -> (64, 256, 256)
    final_view_a = np.concatenate(views_a, axis=0)
    final_view_b = np.concatenate(views_b, axis=0)

    return final_view_a, final_view_b


def process_dataset(df, split_name, load_cached_data=True):
    """
    Processes the dataframe into X and y arrays, with caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_X_path = os.path.join(CACHE_DIR, f"X_{split_name}.npy")
    cache_y_path = os.path.join(CACHE_DIR, f"y_{split_name}.npy")
    cache_ids_path = os.path.join(CACHE_DIR, f"ids_{split_name}.npy")

    is_test = split_name == "test"

    if is_test:
        cache_X_even_path = os.path.join(CACHE_DIR, f"X_test_even.npy")
        cache_X_odd_path = os.path.join(CACHE_DIR, f"X_test_odd.npy")

        if (
            load_cached_data
            and os.path.exists(cache_X_even_path)
            and os.path.exists(cache_X_odd_path)
            and os.path.exists(cache_ids_path)
        ):
            print(f"Loading cached {split_name} data...")
            X_even = np.load(cache_X_even_path)
            X_odd = np.load(cache_X_odd_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return X_even, X_odd, ids
    else:
        if (
            load_cached_data
            and os.path.exists(cache_X_path)
            and os.path.exists(cache_y_path)
            and os.path.exists(cache_ids_path)
        ):
            print(f"Loading cached {split_name} data...")
            X = np.load(cache_X_path)
            y = np.load(cache_y_path)
            ids = np.load(cache_ids_path, allow_pickle=True)
            return X, y, ids

    print(f"Processing {split_name} data from scratch...")

    X_list = []
    y_list = []
    ids_list = []

    X_even_list = []
    X_odd_list = []

    total = len(df)
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"Processing {idx}/{total}")

        view_a, view_b = process_patient(row)
        pid = row["BraTS21ID"]

        if is_test:
            X_even_list.append(view_a)
            X_odd_list.append(view_b)
            ids_list.append(pid)
        else:
            target = row["MGMT_value"]
            # Add View A
            X_list.append(view_a)
            y_list.append(target)
            ids_list.append(pid)

            # Add View B
            X_list.append(view_b)
            y_list.append(target)
            ids_list.append(pid)

    if is_test:
        X_even = np.array(X_even_list, dtype=np.float32)
        X_odd = np.array(X_odd_list, dtype=np.float32)
        ids = np.array(ids_list)

        np.save(cache_X_even_path, X_even)
        np.save(cache_X_odd_path, X_odd)
        np.save(cache_ids_path, ids)
        return X_even, X_odd, ids
    else:
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        ids = np.array(ids_list)

        np.save(cache_X_path, X)
        np.save(cache_y_path, y)
        np.save(cache_ids_path, ids)
        return X, y, ids


def get_dataloaders(
    batch_size=16, num_workers=2, load_cached_data=True, debug_limit=None
):
    """
    Main entry point to get PyTorch DataLoaders.
    """
    logger = Logger()
    logger.section("Data Preparation")

    # Load Metadata
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # Handle Debug Mode
    split_suffix = ""
    if debug_limit:
        logger.log(f"Debug mode: limiting datasets to {debug_limit} samples.")
        train_df = train_df.head(debug_limit)
        val_df = val_df.head(debug_limit)
        test_df = test_df.head(debug_limit)
        split_suffix = "_debug"
        # In debug mode, we typically don't want to load the full cache,
        # but we can cache the debug subset separately.

    # Process Data
    # We pass modified split names to avoid overwriting full cache with debug cache
    X_train, y_train, ids_train = process_dataset(
        train_df, "train" + split_suffix, load_cached_data
    )
    X_val, y_val, ids_val = process_dataset(
        val_df, "val" + split_suffix, load_cached_data
    )
    X_test_even, X_test_odd, ids_test = process_dataset(
        test_df, "test" + split_suffix, load_cached_data
    )

    logger.log(f"Train shape: {X_train.shape}")
    logger.log(f"Val shape: {X_val.shape}")
    logger.log(f"Test shape: {X_test_even.shape} (x2 views)")

    # Create Datasets
    train_dataset = BraTSDataset(X_train, y_train, ids_train)
    val_dataset = BraTSDataset(X_val, y_val, ids_val)
    test_dataset = BraTSTestDataset(X_test_even, X_test_odd, ids_test)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
