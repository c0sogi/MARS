import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from typing import List, Tuple, Optional, Dict
import logging

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("Data_Loader")


def read_raw_dicom_bytes(path: str) -> Optional[np.ndarray]:
    """
    Fallback reader for DICOM files when OpenCV fails.
    Assumes uncompressed pixel data located at the end of the file.
    Supports 512x512 and 256x256 resolutions (uint16).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()

        file_size = len(data)

        # Heuristic for 512x512 uint16
        size_512 = 512 * 512 * 2
        # Allow for header size (approx 1-15KB usually)
        if file_size >= size_512 + 128 and file_size < size_512 + 20000:
            pixel_data = data[-size_512:]
            img = np.frombuffer(pixel_data, dtype=np.uint16).reshape(512, 512)
            return img

        # Heuristic for 256x256 uint16
        size_256 = 256 * 256 * 2
        if file_size >= size_256 + 128 and file_size < size_256 + 20000:
            pixel_data = data[-size_256:]
            img = np.frombuffer(pixel_data, dtype=np.uint16).reshape(256, 256)
            return img

        return None
    except Exception:
        return None


def read_dicom(path: str) -> np.ndarray:
    """
    Reads a DICOM file. Prioritizes OpenCV, falls back to raw binary read.
    Returns a numpy array (H, W) or raises FileNotFoundError/ValueError.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Priority 1: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # Priority 2: Raw Binary Tail-Read
    img = read_raw_dicom_bytes(path)
    if img is not None:
        return img

    raise ValueError(f"Could not read DICOM file: {path}")


def get_slice_id(filename: str) -> int:
    """
    Extracts explicit slice ID from filename (e.g., 'Image-10.dcm' -> 10).
    """
    try:
        # Remove extension
        name = os.path.splitext(filename)[0]
        # Split by '-' and take the last part
        return int(name.split("-")[-1])
    except Exception:
        return -1


def compute_roi_anchor(flair_path: str) -> int:
    """
    Determines the anchor slice ID based on maximum sum of intensity
    in the FLAIR modality within the 15-85% depth range.
    """
    files = sorted([f for f in os.listdir(flair_path) if f.endswith(".dcm")])
    if not files:
        raise ValueError(f"No DICOM files found in {flair_path}")

    # Map ID to filename
    id_map = {}
    for f in files:
        sid = get_slice_id(f)
        if sid != -1:
            id_map[sid] = f

    sorted_ids = sorted(id_map.keys())
    num_slices = len(sorted_ids)

    start_idx = int(num_slices * Config.ROI_DEPTH_MIN)
    end_idx = int(num_slices * Config.ROI_DEPTH_MAX)

    # Safety check for very small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    candidate_ids = sorted_ids[start_idx:end_idx]

    max_intensity = -1.0
    best_id = (
        candidate_ids[len(candidate_ids) // 2]
        if candidate_ids
        else sorted_ids[num_slices // 2]
    )

    for sid in candidate_ids:
        try:
            fpath = os.path.join(flair_path, id_map[sid])
            img = read_dicom(fpath)
            intensity = np.sum(img)
            if intensity > max_intensity:
                max_intensity = intensity
                best_id = sid
        except Exception:
            continue

    return best_id


def process_subject(row: pd.Series) -> Optional[np.ndarray]:
    """
    Loads and stacks MRI data for a single subject.
    Returns: (12, 224, 224) float32 tensor or None if failed.
    """
    try:
        # 1. Determine Anchor from FLAIR
        flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
        anchor_id = compute_roi_anchor(flair_dir)

        # 2. Define target slice IDs
        offsets = [-Config.STRIDE, 0, Config.STRIDE]
        target_ids = [anchor_id + off for off in offsets]

        channels = []

        # 3. Iterate Modalities (FLAIR, T1w, T1wCE, T2w)
        for mod in Config.MODALITIES:
            mod_dir = os.path.join(Config.INPUT_DIR, row[f"path_{mod}"])
            files = [f for f in os.listdir(mod_dir) if f.endswith(".dcm")]

            # Map IDs
            id_map = {}
            for f in files:
                sid = get_slice_id(f)
                if sid != -1:
                    id_map[sid] = f

            available_ids = sorted(id_map.keys())
            if not available_ids:
                raise ValueError(f"No slices for {mod}")

            min_id, max_id = available_ids[0], available_ids[-1]

            # 4. Extract Slices
            for tid in target_ids:
                # Edge Clamping
                if tid < min_id:
                    read_id = min_id
                elif tid > max_id:
                    read_id = max_id
                else:
                    # Find closest if exact missing (though usually continuous)
                    # Using closest available logic
                    read_id = min(available_ids, key=lambda x: abs(x - tid))

                img_path = os.path.join(mod_dir, id_map[read_id])
                img = read_dicom(img_path)

                # Resize
                # Cite solution_lesson_node_00102: Use INTER_AREA for downsampling to preserve texture
                img = cv2.resize(
                    img,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                img = img.astype(np.float32)

                # Normalize (Min-Max)
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)

                channels.append(img)

        # Stack: (12, H, W)
        volume = np.stack(channels, axis=0)
        return volume

    except Exception as e:
        # logger.warning(f"Failed to process subject {row.get('BraTS21ID', 'Unknown')}: {e}")
        return None


def generate_cache(
    metadata_path: str,
    cache_data_path: str,
    cache_labels_path: Optional[str] = None,
    load_cached_data: bool = True,
    debug_limit: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generates or loads cached dataset arrays.
    Implements Circuit Breaker logic.
    """
    # Try loading cache
    if load_cached_data and os.path.exists(cache_data_path):
        logger.info(f"Loading cache from {cache_data_path}")
        data = np.load(cache_data_path)
        labels = None
        if cache_labels_path and os.path.exists(cache_labels_path):
            labels = np.load(cache_labels_path)
        return data, labels

    logger.info(f"Generating cache for {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if debug_limit:
        df = df.head(debug_limit)

    data_list = []
    labels_list = []
    failed_count = 0

    for idx, row in df.iterrows():
        vol = process_subject(row)

        if vol is not None:
            data_list.append(vol)
            if "MGMT_value" in row:
                labels_list.append(row["MGMT_value"])
        else:
            failed_count += 1

    # Circuit Breaker
    total_attempted = len(df)
    failure_rate = failed_count / total_attempted if total_attempted > 0 else 0.0

    logger.info(
        f"Processed {total_attempted} subjects. Failed: {failed_count} ({failure_rate:.2%})"
    )

    if failure_rate > Config.CIRCUIT_BREAKER_THRESHOLD:
        raise RuntimeError(
            f"Circuit Breaker Triggered! Failure rate {failure_rate:.2%} exceeds threshold {Config.CIRCUIT_BREAKER_THRESHOLD}. "
            "Pipeline halted to prevent training on corrupt data."
        )

    data_arr = np.array(data_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32) if labels_list else None

    # Save cache
    os.makedirs(os.path.dirname(cache_data_path), exist_ok=True)
    np.save(cache_data_path, data_arr)
    if labels_arr is not None and cache_labels_path:
        np.save(cache_labels_path, labels_arr)

    return data_arr, labels_arr


class MemoryCachedDataset(Dataset):
    """
    Dataset that holds all data in RAM.
    Applies Albumentations augmentations on-the-fly.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[A.Compose] = None,
    ):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is (C, H, W)
        img = self.data[idx]

        if self.transform:
            # Transpose to (H, W, C) for Albumentations
            img_t = np.transpose(img, (1, 2, 0))
            augmented = self.transform(image=img_t)["image"]
            # Transpose back to (C, H, W)
            img = np.transpose(augmented, (2, 0, 1))

        # Convert to torch tensor
        img_tensor = torch.from_numpy(img).float()

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label

        return img_tensor


def get_dataloaders(
    load_cached_data: bool = True, debug_limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """

    # 1. Define Augmentations
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=Config.AUG_ROTATION_DEG, border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
        ]
    )

    # 2. Process/Load Train Data
    train_data, train_labels = generate_cache(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_DATA,
        Config.CACHE_TRAIN_LABELS,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Process/Load Val Data
    val_data, val_labels = generate_cache(
        Config.VAL_METADATA_PATH,
        Config.CACHE_VAL_DATA,
        Config.CACHE_VAL_LABELS,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 4. Process/Load Test Data
    test_data, _ = generate_cache(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_DATA,
        None,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 5. Create Datasets
    train_dataset = MemoryCachedDataset(
        train_data, train_labels, transform=train_transform
    )
    val_dataset = MemoryCachedDataset(val_data, val_labels, transform=None)
    test_dataset = MemoryCachedDataset(test_data, None, transform=None)

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
