import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from library.utils import extract_image_id, set_seed

# ==========================================
# Constants & Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_opt_16"
IMG_SIZE = 224
NUM_SLICES_PER_MODALITY = 16
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]
TOTAL_CHANNELS = NUM_SLICES_PER_MODALITY * len(MODALITIES)  # 16 * 4 = 64


class BraTSDataset(Dataset):
    """
    A lightweight Dataset wrapper for pre-processed tensors.
    """

    def __init__(self, X, y=None, ids=None):
        """
        Args:
            X (torch.Tensor): Input images of shape (N, 128, 224, 224).
            y (torch.Tensor, optional): Target labels of shape (N, 1).
            ids (np.ndarray, optional): BraTS21IDs for identification (used in test).
        """
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            # For test set, return input and ID
            return self.X[idx], self.ids[idx]


def load_and_process_modality(file_paths):
    """
    Loads DICOM files for a single modality, normalizes volumetrically,
    and samples slices uniformly.

    Returns:
        np.ndarray: Shape (32, 224, 224), float32, normalized [0, 1].
    """
    # 1. Robust Loading & Sorting
    if len(file_paths) == 0:
        return np.zeros((NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # Sort paths using robust integer extraction from filename
    path_id_pairs = []
    for p in file_paths:
        fid = extract_image_id(os.path.basename(p))
        path_id_pairs.append((fid, p))

    path_id_pairs.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in path_id_pairs]

    # Read DICOMs
    slices = []
    for p in sorted_paths:
        full_path = os.path.join(INPUT_DIR, p)
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array.astype(np.float32)
            slices.append(img)
        except Exception:
            continue

    if not slices:
        return np.zeros((NUM_SLICES_PER_MODALITY, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    volume = np.stack(slices)  # (Depth, H, W)

    # 2. Per-Modality Volumetric Normalization
    v_min = np.min(volume)
    v_max = np.max(volume)
    if v_max - v_min > 0:
        volume = (volume - v_min) / (v_max - v_min)
    else:
        volume = np.zeros_like(volume)

    # 3. High-Density Uniform Sampling (10% - 90%)
    depth = volume.shape[0]
    start_idx = int(depth * 0.1)
    end_idx = int(depth * 0.9)

    # Handle edge cases with very few slices
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = depth

    if end_idx > start_idx:
        indices = np.linspace(start_idx, end_idx - 1, NUM_SLICES_PER_MODALITY)
    else:
        # Fallback if depth is extremely small, just repeat or linspace full
        indices = np.linspace(0, depth - 1, NUM_SLICES_PER_MODALITY)

    indices = np.clip(indices.astype(int), 0, depth - 1)
    selected_slices = volume[indices]  # (32, H_orig, W_orig)

    # 4. Resize
    resized_stack = []
    for i in range(len(selected_slices)):
        # cv2.resize expects (Width, Height)
        res = cv2.resize(
            selected_slices[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
        )
        resized_stack.append(res)

    return np.stack(resized_stack)  # (32, 224, 224)


def process_patient(row):
    """
    Processes all modalities for a single patient and stacks them.
    Returns:
        np.ndarray: Shape (128, 224, 224)
    """
    modality_chunks = []
    for mod in MODALITIES:
        col_name = f"{mod}_paths"
        paths = row[col_name] if row[col_name] is not None else []
        chunk = load_and_process_modality(paths)
        modality_chunks.append(chunk)

    # Stack: [FLAIR, T1w, T1wCE, T2w] -> (128, 224, 224)
    full_tensor = np.concatenate(modality_chunks, axis=0)
    return full_tensor


def generate_dataset_arrays(df, desc="dataset"):
    """
    Iterates through the dataframe to generate full X and y arrays.
    """
    X_list = []
    y_list = []
    ids_list = []

    print(f"Processing {desc} ({len(df)} samples)...")

    for idx, row in df.iterrows():
        try:
            tensor = process_patient(row)
            X_list.append(tensor)

            # Handle Target
            if "MGMT_value" in row:
                y_list.append(row["MGMT_value"])
            else:
                y_list.append(-1.0)  # Placeholder for test

            ids_list.append(row["BraTS21ID"])
        except Exception as e:
            print(f"Error processing {row['BraTS21ID']}: {e}")
            # Fallback: Zero tensor
            X_list.append(
                np.zeros((TOTAL_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            )
            y_list.append(0.0)
            ids_list.append(row["BraTS21ID"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    ids = np.array(ids_list)

    return X, y, ids


def get_dataloaders(batch_size=16, load_cached_data=True):
    """
    Main interface. Manages caching and returns PyTorch DataLoaders.
    """
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Define Cache Paths
    cache_files = {
        "train": {
            "X": os.path.join(CACHE_DIR, "X_train.npy"),
            "y": os.path.join(CACHE_DIR, "y_train.npy"),
            "ids": os.path.join(CACHE_DIR, "ids_train.npy"),
        },
        "val": {
            "X": os.path.join(CACHE_DIR, "X_val.npy"),
            "y": os.path.join(CACHE_DIR, "y_val.npy"),
            "ids": os.path.join(CACHE_DIR, "ids_val.npy"),
        },
        "test": {
            "X": os.path.join(CACHE_DIR, "X_test.npy"),
            "ids": os.path.join(CACHE_DIR, "ids_test.npy"),
        },
    }

    # 2. Load Metadata
    train_df = pd.read_parquet(os.path.join(METADATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(METADATA_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(METADATA_DIR, "test.parquet"))

    # 3. Prepare Train Data
    if load_cached_data and os.path.exists(cache_files["train"]["X"]):
        print("Loading cached Train data...")
        X_train = np.load(cache_files["train"]["X"])
        y_train = np.load(cache_files["train"]["y"])
    else:
        X_train, y_train, ids_train = generate_dataset_arrays(train_df, "Train")
        np.save(cache_files["train"]["X"], X_train)
        np.save(cache_files["train"]["y"], y_train)
        np.save(cache_files["train"]["ids"], ids_train)

    train_dataset = BraTSDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)
    )

    # 4. Prepare Val Data
    if load_cached_data and os.path.exists(cache_files["val"]["X"]):
        print("Loading cached Val data...")
        X_val = np.load(cache_files["val"]["X"])
        y_val = np.load(cache_files["val"]["y"])
    else:
        X_val, y_val, ids_val = generate_dataset_arrays(val_df, "Val")
        np.save(cache_files["val"]["X"], X_val)
        np.save(cache_files["val"]["y"], y_val)
        np.save(cache_files["val"]["ids"], ids_val)

    val_dataset = BraTSDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1)
    )

    # 5. Prepare Test Data
    if load_cached_data and os.path.exists(cache_files["test"]["X"]):
        print("Loading cached Test data...")
        X_test = np.load(cache_files["test"]["X"])
        ids_test = np.load(cache_files["test"]["ids"])
    else:
        X_test, _, ids_test = generate_dataset_arrays(test_df, "Test")
        np.save(cache_files["test"]["X"], X_test)
        np.save(cache_files["test"]["ids"], ids_test)

    test_dataset = BraTSDataset(torch.from_numpy(X_test), y=None, ids=ids_test)

    # 6. Create DataLoaders
    # Pin memory enables faster transfer to CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
