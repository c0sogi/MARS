import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
import glob
from library.config import Config
from library.utils import read_dicom_robust, resize_volume, normalize_minmax

# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class BraTSDataset(Dataset):
    def __init__(self, data, labels=None, ids=None, is_train=False):
        """
        Args:
            data (np.ndarray): Shape (N, 2, 12, H, W).
                               Dim 1 is views: 0=Bulk, 1=Core.
            labels (np.ndarray, optional): Shape (N,). Binary targets.
            ids (np.ndarray, optional): Shape (N,). BraTS21IDs for test set.
            is_train (bool): Whether to apply augmentations.
        """
        self.data = data
        self.labels = labels
        self.ids = ids
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract views: (12, H, W)
        # data[idx] is (2, 12, H, W)
        view_bulk = torch.from_numpy(self.data[idx, 0])
        view_core = torch.from_numpy(self.data[idx, 1])

        # Apply synchronized augmentations
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                view_bulk = TF.hflip(view_bulk)
                view_core = TF.hflip(view_core)

            # Random Vertical Flip
            if random.random() > 0.5:
                view_bulk = TF.vflip(view_bulk)
                view_core = TF.vflip(view_core)

            # Random Rotation
            angle = random.uniform(-Config.ROTATION_DEGREES, Config.ROTATION_DEGREES)
            view_bulk = TF.rotate(view_bulk, angle)
            view_core = TF.rotate(view_core, angle)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return view_bulk, view_core, label
        elif self.ids is not None:
            subject_id = self.ids[idx]
            return view_bulk, view_core, subject_id
        else:
            return view_bulk, view_core


# ------------------------------------------------------------------------------
# Data Processing Logic
# ------------------------------------------------------------------------------


def get_sorted_files(dir_path):
    """Returns sorted list of .dcm files in a directory."""
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort by the integer number in the filename (e.g., Image-123.dcm)
    # Assuming format Image-N.dcm
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except:
        files.sort()  # Fallback
    return files


def load_patient_views(row):
    """
    Generates the Bulk and Core views for a single patient.
    Returns:
        np.ndarray: Shape (2, 12, 224, 224)
    """
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    paths = {m: os.path.join(Config.INPUT_DIR, row[f"path_{m}"]) for m in modalities}

    # Get file lists for all modalities
    file_lists = {m: get_sorted_files(paths[m]) for m in modalities}

    # Pre-check: if any modality is missing files, return zeros
    if any(len(l) == 0 for l in file_lists.values()):
        return np.zeros(
            (2, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # --------------------------------------------------------------------------
    # 1. Determine Anchors
    # --------------------------------------------------------------------------

    # --- View 1: Bulk (FLAIR) ---
    flair_files = file_lists["FLAIR"]
    n_flair = len(flair_files)
    start_f = int(n_flair * Config.DEPTH_MIN)
    end_f = int(n_flair * Config.DEPTH_MAX)

    best_flair_idx = n_flair // 2
    max_intensity_sum = -1.0

    # Scan FLAIR to find max intensity sum (Bulk)
    # Optimization: Check every 2nd slice to save time, it's robust enough for "Bulk"
    for i in range(start_f, end_f, 2):
        f_path = os.path.join(paths["FLAIR"], flair_files[i])
        img = read_dicom_robust(f_path)
        s = np.sum(img)
        if s > max_intensity_sum:
            max_intensity_sum = s
            best_flair_idx = i

    # --- View 2: Core (T1wCE) ---
    ce_files = file_lists["T1wCE"]
    n_ce = len(ce_files)
    start_c = int(n_ce * Config.DEPTH_MIN)
    end_c = int(n_ce * Config.DEPTH_MAX)

    best_ce_idx = n_ce // 2
    max_pixel_val = -1.0

    # Scan T1wCE to find max pixel intensity (Core)
    # Scan every slice for precision on small tumors
    for i in range(start_c, end_c):
        f_path = os.path.join(paths["T1wCE"], ce_files[i])
        img = read_dicom_robust(f_path)
        # Use max pixel value
        m = np.max(img)
        if m > max_pixel_val:
            max_pixel_val = m
            best_ce_idx = i

    # --------------------------------------------------------------------------
    # 2. Build Volumes
    # --------------------------------------------------------------------------

    def build_view(anchor_modality, anchor_idx, total_slices_anchor):
        """
        Constructs a 12-channel volume based on an anchor index in a specific modality.
        Uses relative depth mapping for other modalities.
        """
        view_channels = []

        # Define relative depth of the anchor
        if total_slices_anchor > 0:
            rel_depth = anchor_idx / total_slices_anchor
        else:
            rel_depth = 0.5

        # Order: FLAIR(3), T1w(3), T1wCE(3), T2w(3) -> Groups=4 compatible
        for mod in modalities:
            m_files = file_lists[mod]
            n_m = len(m_files)

            if n_m == 0:
                # Missing modality fallback
                for _ in range(Config.NUM_SLICES):
                    view_channels.append(
                        np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
                    )
                continue

            # Map anchor to this modality
            center_idx = int(rel_depth * n_m)

            # Select 3 slices: center - stride, center, center + stride
            indices = [
                center_idx - Config.STRIDE,
                center_idx,
                center_idx + Config.STRIDE,
            ]

            # Edge Clamping
            indices = [max(0, min(idx, n_m - 1)) for idx in indices]

            for idx in indices:
                f_path = os.path.join(paths[mod], m_files[idx])
                img = read_dicom_robust(f_path)
                img = resize_volume(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                img = normalize_minmax(img)
                view_channels.append(img)

        return np.stack(view_channels, axis=0)  # (12, 224, 224)

    # Build Bulk View (anchored on FLAIR)
    vol_bulk = build_view("FLAIR", best_flair_idx, n_flair)

    # Build Core View (anchored on T1wCE)
    vol_core = build_view("T1wCE", best_ce_idx, n_ce)

    # Stack views: (2, 12, 224, 224)
    return np.stack([vol_bulk, vol_core], axis=0)


def process_dataset(df, desc="Data"):
    """
    Iterates through dataframe, loads data, and returns stacked numpy arrays.
    """
    print(f"Processing {desc} ({len(df)} subjects)...")
    data_list = []

    for _, row in df.iterrows():
        # Returns (2, 12, H, W)
        patient_data = load_patient_views(row)
        data_list.append(patient_data)

    return np.array(data_list, dtype=np.float32)


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    Handles caching of pre-processed numpy arrays.
    """

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # --------------------------------------------------------------------------
    # 2. Train Data
    # --------------------------------------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.CACHE_TRAIN_DATA)
        and os.path.exists(Config.CACHE_TRAIN_LABELS)
    ):
        print("Loading cached training data...")
        X_train = np.load(Config.CACHE_TRAIN_DATA)
        y_train = np.load(Config.CACHE_TRAIN_LABELS)
    else:
        print("Generating training data from scratch...")
        X_train = process_dataset(df_train, "Train")
        y_train = df_train["MGMT_value"].values.astype(np.float32)

        np.save(Config.CACHE_TRAIN_DATA, X_train)
        np.save(Config.CACHE_TRAIN_LABELS, y_train)
        print("Training data cached.")

    # --------------------------------------------------------------------------
    # 3. Validation Data
    # --------------------------------------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.CACHE_VAL_DATA)
        and os.path.exists(Config.CACHE_VAL_LABELS)
    ):
        print("Loading cached validation data...")
        X_val = np.load(Config.CACHE_VAL_DATA)
        y_val = np.load(Config.CACHE_VAL_LABELS)
    else:
        print("Generating validation data from scratch...")
        X_val = process_dataset(df_val, "Val")
        y_val = df_val["MGMT_value"].values.astype(np.float32)

        np.save(Config.CACHE_VAL_DATA, X_val)
        np.save(Config.CACHE_VAL_LABELS, y_val)
        print("Validation data cached.")

    # --------------------------------------------------------------------------
    # 4. Test Data
    # --------------------------------------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.CACHE_TEST_DATA)
        and os.path.exists(Config.CACHE_TEST_IDS)
    ):
        print("Loading cached test data...")
        X_test = np.load(Config.CACHE_TEST_DATA)
        ids_test = np.load(Config.CACHE_TEST_IDS)
    else:
        print("Generating test data from scratch...")
        X_test = process_dataset(df_test, "Test")
        ids_test = df_test["BraTS21ID"].values

        np.save(Config.CACHE_TEST_DATA, X_test)
        np.save(Config.CACHE_TEST_IDS, ids_test)
        print("Test data cached.")

    # --------------------------------------------------------------------------
    # 5. Create Datasets and Loaders
    # --------------------------------------------------------------------------
    train_dataset = BraTSDataset(X_train, labels=y_train, is_train=True)
    val_dataset = BraTSDataset(X_val, labels=y_val, is_train=False)
    test_dataset = BraTSDataset(X_test, ids=ids_test, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
