import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library import config, utils

# Initialize Logger
logger = utils.get_logger("DATA_LOADER")


def load_dicom(path):
    """
    Loads a DICOM file. Tries OpenCV first, falls back to binary tail-read.
    Resizes to config.IMG_SIZE (224x224).
    Returns float32 numpy array.
    """
    img = None
    try:
        # Attempt 1: OpenCV
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        pass

    if img is None:
        # Attempt 2: Fallback Binary Tail-Read
        # DICOM pixel data is usually at the end. We assume standard resolutions.
        try:
            file_size = os.path.getsize(path)

            # Check for 512x512 (uint16 = 2 bytes)
            dim = 512
            byte_size = dim * dim * 2

            if file_size >= byte_size:
                with open(path, "rb") as f:
                    f.seek(-byte_size, 2)
                    buf = f.read()
                    img = np.frombuffer(buf, dtype=np.uint16).reshape(dim, dim)
            else:
                # Check for 256x256
                dim = 256
                byte_size = dim * dim * 2
                if file_size >= byte_size:
                    with open(path, "rb") as f:
                        f.seek(-byte_size, 2)
                        buf = f.read()
                        img = np.frombuffer(buf, dtype=np.uint16).reshape(dim, dim)
        except Exception:
            pass

    if img is None:
        # Failure case: Return black image
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    # Resize to target dimension using Area interpolation (best for downsampling)
    if img.shape[0] != config.IMG_SIZE or img.shape[1] != config.IMG_SIZE:
        img = cv2.resize(
            img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )

    return img.astype(np.float32)


def get_roi_anchor(flair_path):
    """
    Determines the anchor slice ID based on FLAIR modality.
    1. Sorts files by Image-{ID}.
    2. Restricts to 15-85% depth.
    3. Finds slice with max intensity sum (raw pixel values).

    Returns:
        tuple: (best_id, min_id, max_id) or None if failed.
    """
    if not os.path.exists(flair_path):
        return None

    files = [
        f
        for f in os.listdir(flair_path)
        if f.startswith("Image-") and f.endswith(".dcm")
    ]
    if not files:
        return None

    # Parse IDs from filenames: Image-{id}.dcm
    try:
        file_ids = [int(f.split("-")[1].split(".")[0]) for f in files]
    except Exception:
        return None

    file_ids.sort()

    if not file_ids:
        return None

    # Determine depth range
    num_slices = len(file_ids)
    start_idx = int(num_slices * config.ROI_DEPTH_MIN)
    end_idx = int(num_slices * config.ROI_DEPTH_MAX)

    # Safety check for very small volumes
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_slices

    candidate_ids = file_ids[start_idx:end_idx]
    if not candidate_ids:
        candidate_ids = file_ids

    # Find max intensity slice
    max_intensity = -1.0
    best_id = candidate_ids[len(candidate_ids) // 2]  # Default to middle

    for fid in candidate_ids:
        path = os.path.join(flair_path, f"Image-{fid}.dcm")
        img = load_dicom(path)
        current_sum = np.sum(img)

        if current_sum > max_intensity:
            max_intensity = current_sum
            best_id = fid

    return best_id, file_ids[0], file_ids[-1]


def build_volume(row, stride):
    """
    Constructs the 12-channel volume for a subject.

    Args:
        row (pd.Series): Metadata row containing paths.
        stride (int): Stride for neighbor selection (Texture vs Context).

    Returns:
        np.ndarray: Shape (12, 224, 224), float32.
    """
    # 1. Get Anchor from FLAIR
    flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])
    res = get_roi_anchor(flair_path)

    if res is None:
        # Return zeros if FLAIR is broken
        return np.zeros(
            (config.NUM_CHANNELS, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32
        )

    anchor_id, min_id, max_id = res

    # 2. Define Target IDs with Geometric Clamping
    # We want [anchor-stride, anchor, anchor+stride]
    # If a neighbor is outside the tumor volume (defined by FLAIR bounds), we clamp it.
    target_ids_raw = [anchor_id - stride, anchor_id, anchor_id + stride]
    clamped_ids = []
    for tid in target_ids_raw:
        if tid < min_id:
            clamped_ids.append(min_id)
        elif tid > max_id:
            clamped_ids.append(max_id)
        else:
            clamped_ids.append(tid)

    # 3. Load Channels across all modalities
    # Order: FLAIR (3), T1w (3), T1wCE (3), T2w (3)
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    channels = []

    for mod in modalities:
        mod_dir = os.path.join(config.INPUT_DIR, row[f"path_{mod}"])

        for cid in clamped_ids:
            # Construct filename for the specific slice ID
            fname = f"Image-{cid}.dcm"
            fpath = os.path.join(mod_dir, fname)

            # Spectral Padding: If file missing in this modality, use zeros
            if os.path.exists(fpath):
                img = load_dicom(fpath)
            else:
                img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

            # Independent Per-Channel Min-Max Normalization
            min_val = np.min(img)
            max_val = np.max(img)

            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            else:
                # If image is constant (e.g. all black), keep as zeros
                img = np.zeros_like(img)

            channels.append(img)

    # Stack channels: (12, 224, 224)
    volume = np.stack(channels, axis=0)
    return volume


def process_dataset_split(
    metadata_path, stride, cache_name, load_cache=True, debug_size=None
):
    """
    Processes a dataset split (train/val/test), handling caching logic.

    Args:
        metadata_path (str): Path to CSV.
        stride (int): Stride for volume construction.
        cache_name (str): Identifier for cache file.
        load_cache (bool): Whether to attempt loading from disk.
        debug_size (int): Limit number of samples for debugging.

    Returns:
        tuple: (data, labels) as numpy arrays.
    """
    cache_file_data = os.path.join(config.CACHE_DIR, f"{cache_name}_data.npy")
    cache_file_labels = os.path.join(config.CACHE_DIR, f"{cache_name}_labels.npy")

    # 1. Try Loading Cache
    if (
        load_cache
        and os.path.exists(cache_file_data)
        and os.path.exists(cache_file_labels)
    ):
        logger.info(f"Loading cached data: {cache_name}")
        data = np.load(cache_file_data)
        labels = np.load(cache_file_labels)

        if debug_size is not None and len(data) > debug_size:
            return data[:debug_size], labels[:debug_size]
        return data, labels

    # 2. Process from Scratch
    logger.info(f"Processing data from {metadata_path} (Stride={stride})...")
    df = pd.read_csv(metadata_path)

    if debug_size is not None:
        df = df.head(debug_size)

    data_list = []
    labels_list = []

    for idx, row in df.iterrows():
        vol = build_volume(row, stride)
        data_list.append(vol)

        # Handle labels (Test set has no MGMT_value)
        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(-1)

    data = np.array(data_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # 3. Save Cache (Only if full run)
    if debug_size is None:
        logger.info(f"Saving cache to {cache_file_data}")
        np.save(cache_file_data, data)
        np.save(cache_file_labels, labels)

    return data, labels


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS data.
    Handles augmentations using Albumentations.
    """

    def __init__(self, data, labels, is_train=False):
        self.data = data
        self.labels = labels
        self.is_train = is_train

        # Define Augmentations
        if self.is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Rotate +/- 15 deg, use Reflection Padding to avoid artifacts
                    A.Rotate(
                        limit=config.AUG_ROTATE_DEG,
                        border_mode=cv2.BORDER_REFLECT,
                        p=config.AUG_PROB,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Input data is (C, H, W) -> (12, 224, 224)
        vol = self.data[idx]
        label = self.labels[idx]

        if self.is_train:
            # Albumentations requires (H, W, C)
            vol_hwc = np.transpose(vol, (1, 2, 0))

            # Apply transform
            augmented = self.transform(image=vol_hwc)["image"]

            # ToTensorV2 converts back to (C, H, W) and returns a Tensor
            vol_tensor = augmented
        else:
            vol_tensor = torch.from_numpy(vol)

        return vol_tensor, torch.tensor(label, dtype=torch.float32)


def get_datasets(load_cache=True, debug_size=config.DEBUG_SAMPLE_SIZE):
    """
    Generates Train and Validation datasets for both Texture (Stride 2) and Context (Stride 5) models.
    """
    # Train - Texture
    train_data_A, train_labels = process_dataset_split(
        config.TRAIN_METADATA,
        config.STRIDE_TEXTURE,
        "train_texture",
        load_cache,
        debug_size,
    )
    # Train - Context
    train_data_B, _ = process_dataset_split(
        config.TRAIN_METADATA,
        config.STRIDE_CONTEXT,
        "train_context",
        load_cache,
        debug_size,
    )

    # Val - Texture
    val_data_A, val_labels = process_dataset_split(
        config.VAL_METADATA,
        config.STRIDE_TEXTURE,
        "val_texture",
        load_cache,
        debug_size,
    )
    # Val - Context
    val_data_B, _ = process_dataset_split(
        config.VAL_METADATA,
        config.STRIDE_CONTEXT,
        "val_context",
        load_cache,
        debug_size,
    )

    train_ds_A = BraTSDataset(train_data_A, train_labels, is_train=True)
    train_ds_B = BraTSDataset(train_data_B, train_labels, is_train=True)

    val_ds_A = BraTSDataset(val_data_A, val_labels, is_train=False)
    val_ds_B = BraTSDataset(val_data_B, val_labels, is_train=False)

    return train_ds_A, train_ds_B, val_ds_A, val_ds_B


def get_test_datasets(load_cache=True):
    """
    Generates Test datasets for both Texture and Context models.
    """
    # Test - Texture
    test_data_A, test_ids = process_dataset_split(
        config.TEST_METADATA, config.STRIDE_TEXTURE, "test_texture", load_cache, None
    )
    # Test - Context
    test_data_B, _ = process_dataset_split(
        config.TEST_METADATA, config.STRIDE_CONTEXT, "test_context", load_cache, None
    )

    test_ds_A = BraTSDataset(test_data_A, test_ids, is_train=False)
    test_ds_B = BraTSDataset(test_data_B, test_ids, is_train=False)

    return test_ds_A, test_ds_B


def get_test_ids():
    """Returns the BraTS21IDs for the test set."""
    df = pd.read_csv(config.TEST_METADATA)
    return df["BraTS21ID"].values
