import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom

from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMAGE_SIZE,
    NUM_SLICES_PER_MODALITY,
    INPUT_CHANNELS,
    NUM_MODALITIES,
)


class SSBHDDataset(Dataset):
    """
    Dataset class for Stabilized Semantic-Block High-Density Network.
    Serves pre-processed tensors from memory/cache.
    """

    def __init__(self, X, y=None, ids=None):
        self.X = X
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (N, 128, 224, 224)
        img = self.X[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32)

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            # For inference, return ID as well to track predictions
            return img_tensor, self.ids[idx]


def load_dicom_volume(rel_paths):
    """
    Loads a volume of DICOM slices, sorts them by InstanceNumber.
    Returns a 3D numpy array (Depth, Height, Width).
    """
    if not rel_paths:
        return None

    slices = []
    for rel_path in rel_paths:
        full_path = os.path.join(INPUT_DIR, rel_path)
        if not os.path.exists(full_path):
            continue

        try:
            dcm = pydicom.dcmread(full_path)
            # Tag (0020, 0013) is Instance Number.
            # Crucial for spatial coherence.
            if hasattr(dcm, "InstanceNumber"):
                inst_num = int(dcm.InstanceNumber)
            else:
                # Fallback: try to infer from filename (e.g., Image-123.dcm)
                try:
                    base = os.path.basename(full_path)
                    inst_num = int(base.split("-")[1].split(".")[0])
                except:
                    inst_num = 0

            arr = dcm.pixel_array
            slices.append((inst_num, arr))
        except Exception:
            # Skip corrupted files to prevent silent failures later
            continue

    if not slices:
        return None

    # Sort by Instance Number
    slices.sort(key=lambda x: x[0])

    # Stack into volume
    vol = np.stack([s[1] for s in slices])
    return vol


def process_patient(row):
    """
    Processes a single patient:
    1. Loads 4 modalities.
    2. Global Volumetric Normalization (min/max over all modalities).
    3. Uniform Sampling (32 slices, 10-90% depth).
    4. Resizing to 224x224.
    5. Stacking into (128, 224, 224).
    """
    modalities = ["flair", "t1w", "t1wce", "t2w"]
    loaded_vols = []

    # 1. Load volumes
    for mod in modalities:
        paths = row.get(f"{mod}_paths", [])
        vol = load_dicom_volume(paths)
        loaded_vols.append(vol)

    # 2. Global Normalization
    # Compute global min/max across all valid modalities for this patient
    valid_arrays = [v for v in loaded_vols if v is not None]

    if not valid_arrays:
        # Fallback: return zeros if no data found
        return np.zeros((INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    global_min = min(v.min() for v in valid_arrays)
    global_max = max(v.max() for v in valid_arrays)
    range_val = global_max - global_min
    if range_val < 1e-6:
        range_val = 1.0

    # 3. Process each modality
    processed_blocks = []
    for vol in loaded_vols:
        # Create empty block
        block = np.zeros(
            (NUM_SLICES_PER_MODALITY, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )

        if vol is not None and vol.size > 0:
            # Normalize
            vol_norm = (vol.astype(np.float32) - global_min) / range_val

            # Sampling: 10% to 90% depth
            depth = vol_norm.shape[0]
            start = int(depth * 0.1)
            end = int(depth * 0.9)
            if end <= start:
                start = 0
                end = depth

            # Generate indices
            if depth > 0:
                indices = np.linspace(start, end - 1, NUM_SLICES_PER_MODALITY)
                indices = np.round(indices).astype(int)
                indices = np.clip(indices, 0, depth - 1)

                # Extract and Resize
                for i, idx in enumerate(indices):
                    slc = vol_norm[idx]
                    if slc.shape != (IMAGE_SIZE, IMAGE_SIZE):
                        slc = cv2.resize(
                            slc,
                            (IMAGE_SIZE, IMAGE_SIZE),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    block[i] = slc

        processed_blocks.append(block)

    # 4. Stack
    # [ (32, 224, 224), ... ] -> (128, 224, 224)
    final_tensor = np.concatenate(processed_blocks, axis=0)
    return final_tensor


def get_dataset(metadata_path, dataset_type="train", load_cached_data=True):
    """
    Factory function to get the dataset. Handles caching logic.
    dataset_type: 'train', 'val', or 'test'
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache paths
    cache_X = os.path.join(WORKING_DIR, f"cached_{dataset_type}_X.npy")
    cache_y = os.path.join(WORKING_DIR, f"cached_{dataset_type}_y.npy")
    cache_ids = os.path.join(WORKING_DIR, f"cached_{dataset_type}_ids.npy")

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_X) and os.path.exists(cache_ids):
        # Check y existence based on type
        has_y = os.path.exists(cache_y)
        if (dataset_type in ["train", "val"] and has_y) or (dataset_type == "test"):
            print(f"Loading cached {dataset_type} data from {WORKING_DIR}...")
            X = np.load(cache_X)
            ids = np.load(cache_ids, allow_pickle=True)
            y = np.load(cache_y) if has_y else None
            return SSBHDDataset(X, y, ids)

    # Process from scratch
    print(f"Processing {dataset_type} data from scratch (this may take a while)...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_parquet(metadata_path)

    X_list = []
    y_list = []
    ids_list = []

    total = len(df)
    for idx, row in df.iterrows():
        # Print progress periodically
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} patients")

        tensor = process_patient(row)
        X_list.append(tensor)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    # Convert to numpy
    X = np.array(X_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save cache
    print(f"Saving {dataset_type} cache to {WORKING_DIR}...")
    np.save(cache_X, X)
    np.save(cache_ids, ids)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
        np.save(cache_y, y)
        return SSBHDDataset(X, y, ids)
    else:
        return SSBHDDataset(X, None, ids)
