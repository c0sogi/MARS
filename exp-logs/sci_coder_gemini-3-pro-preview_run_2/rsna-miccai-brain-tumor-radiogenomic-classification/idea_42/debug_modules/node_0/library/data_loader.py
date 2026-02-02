import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import re
from library.config import Config
from library.utils import save_to_cache, load_from_cache

# ------------------------------------------------------------------------------
# Helper Functions: File I/O and Processing
# ------------------------------------------------------------------------------


def get_slice_id(filename):
    """
    Extracts the integer slice ID from a filename (e.g., 'Image-123.dcm' -> 123).
    """
    m = re.search(r"Image-(\d+)\.dcm", filename)
    if m:
        return int(m.group(1))
    return -1


def read_dicom_tiered(path):
    """
    Reads a DICOM file with a tiered approach:
    1. Try OpenCV (standard).
    2. Fallback to raw binary tail-read if OpenCV fails (handling missing codecs).

    Performs resizing to Config.IMG_SIZE and returns a float32 array.
    """
    img = None

    # Tier 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Tier 2: Binary Fallback
    if img is None:
        try:
            file_size = os.path.getsize(path)
            # Heuristic for resolution based on file size
            # 512x512 uint16 = 524,288 bytes
            # 256x256 uint16 = 131,072 bytes
            if file_size >= 524288:
                shape = (512, 512)
                offset = 512 * 512 * 2
            elif file_size >= 131072:
                shape = (256, 256)
                offset = 256 * 256 * 2
            else:
                # Unknown size, return zeros
                return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

            with open(path, "rb") as f:
                f.seek(-offset, os.SEEK_END)
                b = f.read(offset)
                img = np.frombuffer(b, dtype=np.uint16).reshape(shape)
        except Exception:
            # Final fallback
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Preprocessing
    img = img.astype(np.float32)

    # Resize if necessary
    if img.shape[0] != Config.IMG_SIZE or img.shape[1] != Config.IMG_SIZE:
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

    return img


def get_sorted_files(folder):
    """Returns a list of DICOM files in a folder, sorted by their slice ID."""
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".dcm")]
    files.sort(key=lambda x: get_slice_id(x))
    return files


def select_candidates(flair_path):
    """
    Analyzes FLAIR slices to find Top-3 local maxima in intensity sum.
    Returns a list of explicit Slice IDs (integers).
    """
    files = get_sorted_files(flair_path)
    if not files:
        # Fallback if directory empty (should be caught by metadata check)
        return [0] * Config.NUM_CANDIDATES

    # 1. Calculate Integrals (Sum of Intensity)
    # We read all files to build the Z-axis profile
    integrals = []
    file_map = {}  # List Index -> Slice ID

    n_slices = len(files)
    start_idx = int(n_slices * Config.SEARCH_MIN_DEPTH)
    end_idx = int(n_slices * Config.SEARCH_MAX_DEPTH)

    for i, f in enumerate(files):
        file_map[i] = get_slice_id(f)

        # Skip if out of bounds (optimization: don't read)
        if i < start_idx or i > end_idx:
            integrals.append(0)
            continue

        path = os.path.join(flair_path, f)
        img = read_dicom_tiered(path)
        val = np.sum(img)
        integrals.append(val)

    integrals = np.array(integrals)

    # 2. Find Candidates (Local Maxima)
    # Sort indices by intensity descending
    sorted_indices = np.argsort(integrals)[::-1]

    selected_indices = []
    for idx in sorted_indices:
        if integrals[idx] == 0:
            continue  # Skip masked/empty

        # Distance constraint
        is_far = True
        for s in selected_indices:
            if abs(idx - s) < Config.CANDIDATE_MIN_DIST:
                is_far = False
                break

        if is_far:
            selected_indices.append(idx)

        if len(selected_indices) >= Config.NUM_CANDIDATES:
            break

    # 3. Fallback (Duplicate if not enough candidates)
    while len(selected_indices) < Config.NUM_CANDIDATES:
        if selected_indices:
            selected_indices.append(selected_indices[0])
        else:
            # If absolutely no valid slices, pick middle
            mid = n_slices // 2
            selected_indices.append(mid)

    # Return explicit Slice IDs
    return [file_map.get(i, 0) for i in selected_indices]


def process_patient(row, input_dir):
    """
    Process a single patient:
    1. Select candidates via FLAIR.
    2. Build multi-channel stacks for each candidate.
    3. Normalize.

    Returns: Tensor of shape (NUM_CANDIDATES, NUM_CHANNELS, H, W)
    """
    # Paths
    paths = {
        "FLAIR": os.path.join(input_dir, row["path_FLAIR"]),
        "T1w": os.path.join(input_dir, row["path_T1w"]),
        "T1wCE": os.path.join(input_dir, row["path_T1wCE"]),
        "T2w": os.path.join(input_dir, row["path_T2w"]),
    }

    # 1. Select Candidates (using FLAIR)
    candidate_ids = select_candidates(paths["FLAIR"])

    # 2. Map available files for all modalities
    # Modality -> {SliceID: Filename}
    mod_file_maps = {}
    mod_order = ["FLAIR", "T1w", "T1wCE", "T2w"]

    for mod in mod_order:
        files = get_sorted_files(paths[mod])
        mod_file_maps[mod] = {get_slice_id(f): f for f in files}

    # 3. Build Tensor
    # Shape: (3, 12, 224, 224)
    patient_tensor = np.zeros(
        (Config.NUM_CANDIDATES, Config.NUM_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE),
        dtype=np.float32,
    )

    for c_idx, anchor_id in enumerate(candidate_ids):
        # Offsets: [Anchor-5, Anchor, Anchor+5]
        offsets = [-Config.STACK_STRIDE, 0, Config.STACK_STRIDE]

        channel_ptr = 0
        for mod in mod_order:
            f_map = mod_file_maps[mod]
            available_ids = sorted(list(f_map.keys()))

            if not available_ids:
                # Missing modality (rare), fill zeros
                channel_ptr += len(offsets)
                continue

            for off in offsets:
                target_id = anchor_id + off

                # Edge Clamping: Find nearest available ID
                if target_id in f_map:
                    read_id = target_id
                else:
                    # Find nearest ID
                    read_id = min(available_ids, key=lambda x: abs(x - target_id))

                # Read Image
                file_path = os.path.join(paths[mod], f_map[read_id])
                img = read_dicom_tiered(file_path)

                # Independent Per-Channel Min-Max Scaling [0, 1]
                mi, ma = img.min(), img.max()
                if ma - mi > 1e-6:
                    img = (img - mi) / (ma - mi)
                else:
                    img = np.zeros_like(img)

                patient_tensor[c_idx, channel_ptr, :, :] = img
                channel_ptr += 1

    return patient_tensor


def generate_dataset_arrays(metadata_df, cache_prefix, load_cached_data=True):
    """
    Generates or loads the full dataset arrays (X, y, ids).
    Uses caching to avoid re-processing.
    """
    data_file = f"{cache_prefix}_data.npy"
    labels_file = f"{cache_prefix}_labels.npy"
    ids_file = f"{cache_prefix}_ids.npy"

    # 1. Try Load
    if load_cached_data:
        X = load_from_cache(data_file)
        y = load_from_cache(labels_file)
        ids = load_from_cache(ids_file)

        if (
            X is not None
            and (y is not None or "test" in cache_prefix)
            and ids is not None
        ):
            print(f"Loaded {cache_prefix} data from cache.")
            return X, y, ids

    print(
        f"Processing {cache_prefix} data from scratch ({len(metadata_df)} subjects)..."
    )

    # 2. Process in Parallel
    # Convert DataFrame to list of dicts for joblib
    rows = metadata_df.to_dict("records")

    results = Parallel(n_jobs=Config.NUM_WORKERS, verbose=0)(
        delayed(process_patient)(row, Config.INPUT_DIR) for row in rows
    )

    # 3. Aggregate
    X = np.array(results, dtype=np.float32)

    # Extract labels if available
    if "MGMT_value" in metadata_df.columns:
        y = metadata_df["MGMT_value"].values.astype(np.float32)
    else:
        y = np.zeros(len(metadata_df), dtype=np.float32)  # Dummy for test

    ids = metadata_df["BraTS21ID"].values.astype(np.int64)

    # 4. Save to Cache
    save_to_cache(X, data_file)
    save_to_cache(y, labels_file)
    save_to_cache(ids, ids_file)

    return X, y, ids


# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------


class MILDataset(Dataset):
    def __init__(self, X, y, ids, transform=None):
        """
        Args:
            X: (N, 3, 12, 224, 224) float32 tensor
            y: (N,) float32 labels
            ids: (N,) int64 IDs
            transform: Albumentations or similar (optional)
        """
        self.X = X
        self.y = y
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Extract bag of instances: (3, 12, 224, 224)
        bag = self.X[idx]
        label = self.y[idx]
        subject_id = self.ids[idx]

        # Apply transforms if any
        # Note: Transforms usually expect (H, W, C).
        # Here we have (Candidates, Channels, H, W).
        # We apply transform per candidate if needed.
        # For this implementation, we assume minimal augmentation handled in training loop or here.
        # We return tensors directly.

        return (
            torch.from_numpy(bag),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(subject_id, dtype=torch.long),
        )


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def get_train_val_datasets(load_cached=True):
    """
    Loads training and validation datasets using metadata.
    Returns: (train_dataset, val_dataset)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Generate/Load Arrays
    X_train, y_train, ids_train = generate_dataset_arrays(
        train_df, "train", load_cached
    )
    X_val, y_val, ids_val = generate_dataset_arrays(val_df, "val", load_cached)

    # Create Datasets
    train_dataset = MILDataset(X_train, y_train, ids_train)
    val_dataset = MILDataset(X_val, y_val, ids_val)

    return train_dataset, val_dataset


def get_test_dataset(load_cached=True):
    """
    Loads test dataset using metadata.
    Returns: test_dataset
    """
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Generate/Load Arrays (y will be dummy zeros)
    X_test, y_test, ids_test = generate_dataset_arrays(test_df, "test", load_cached)

    test_dataset = MILDataset(X_test, y_test, ids_test)

    return test_dataset
