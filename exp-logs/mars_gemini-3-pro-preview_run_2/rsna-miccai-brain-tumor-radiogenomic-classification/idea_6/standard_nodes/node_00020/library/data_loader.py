import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    IMG_SIZE,
    STRIDE,
    EXCLUDE_TOP_BOTTOM_RATIO,
    ROI_CACHE_PATH,
    INPUT_DIR,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)


def read_dicom_slice(path):
    """
    Reads a DICOM file. Tries OpenCV first, then falls back to raw binary reading.
    Returns a numpy array.
    """
    # 1. Try OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    # 2. Raw Binary Fallback
    # Assumes uncompressed pixel data is located at the end of the file.
    try:
        file_size = os.path.getsize(path)
        # Common MRI dimensions to check against
        dims = [512, 256, 240]

        with open(path, "rb") as f:
            for dim in dims:
                num_pixels = dim * dim
                num_bytes = num_pixels * 2  # uint16 (2 bytes per pixel)

                if file_size >= num_bytes:
                    f.seek(-num_bytes, 2)
                    data = f.read()
                    if len(data) == num_bytes:
                        img = np.frombuffer(data, dtype=np.uint16).reshape(dim, dim)
                        return img
    except Exception:
        pass

    # 3. Last Resort: Return Zeros
    return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)


def compute_anchor_ratio(flair_dir):
    """
    Computes the relative position (0.0-1.0) of the tumor center based on FLAIR intensity.
    Uses the 'Center-Focused Anatomical Anchoring' heuristic.
    """
    if not os.path.exists(flair_dir):
        return 0.5

    # List and sort files numerically
    files = sorted(
        [
            f
            for f in os.listdir(flair_dir)
            if f.endswith(".dcm") or f.startswith("Image-")
        ],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    num_slices = len(files)
    if num_slices == 0:
        return 0.5

    # Exclude top and bottom percentages to avoid artifacts
    start_idx = int(num_slices * EXCLUDE_TOP_BOTTOM_RATIO)
    end_idx = int(num_slices * (1 - EXCLUDE_TOP_BOTTOM_RATIO))

    if start_idx >= end_idx:
        return 0.5

    max_intensity = -1
    best_idx = (start_idx + end_idx) // 2

    # Iterate through valid range
    for i in range(start_idx, end_idx):
        path = os.path.join(flair_dir, files[i])
        img = read_dicom_slice(path)

        # Cite solution_lesson_node_00019: Use full image intensity instead of center crop
        # to avoid missing peripheral tumors.
        mean_val = np.mean(img)

        if mean_val > max_intensity:
            max_intensity = mean_val
            best_idx = i

    return best_idx / num_slices


def get_anchor_ratios(df, load_cached_data=True):
    """
    Manages caching of anchor ratios to avoid re-computing heavy IO operations.
    """
    cache_data = {}

    # 1. Load Cache
    if load_cached_data and os.path.exists(ROI_CACHE_PATH):
        try:
            cache_df = pd.read_parquet(ROI_CACHE_PATH)
            cache_data = dict(zip(cache_df["BraTS21ID"], cache_df["anchor_ratio"]))
        except Exception:
            cache_data = {}

    # 2. Identify missing IDs
    ids_to_process = []
    for _, row in df.iterrows():
        if row["BraTS21ID"] not in cache_data:
            ids_to_process.append(row)

    # 3. Compute missing
    if ids_to_process:
        new_entries = []
        for row in ids_to_process:
            flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
            ratio = compute_anchor_ratio(flair_path)

            cache_data[row["BraTS21ID"]] = ratio
            new_entries.append({"BraTS21ID": row["BraTS21ID"], "anchor_ratio": ratio})

        # 4. Update Cache File
        if new_entries:
            # Reconstruct full dataframe from current cache state
            full_df = pd.DataFrame(
                [{"BraTS21ID": k, "anchor_ratio": v} for k, v in cache_data.items()]
            )
            os.makedirs(os.path.dirname(ROI_CACHE_PATH), exist_ok=True)
            full_df.to_parquet(ROI_CACHE_PATH)

    return cache_data


def get_transforms(phase):
    """
    Returns Albumentations transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), ToTensorV2()])


class MGMTDataset(Dataset):
    def __init__(self, df, anchor_cache, phase="train"):
        self.df = df
        self.anchor_cache = anchor_cache
        self.phase = phase
        self.transforms = get_transforms(phase)
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # Retrieve pre-computed anchor ratio
        anchor_ratio = self.anchor_cache.get(brats_id, 0.5)

        channels = []

        for mod in self.modalities:
            mod_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])

            # List files
            if os.path.exists(mod_path):
                files = sorted(
                    [
                        f
                        for f in os.listdir(mod_path)
                        if f.endswith(".dcm") or f.startswith("Image-")
                    ],
                    key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
                )
            else:
                files = []

            num_slices = len(files)

            if num_slices == 0:
                # Missing modality: create 3 empty slices
                for _ in range(3):
                    channels.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
                continue

            # Calculate indices based on relative anchor
            anchor_idx = int(anchor_ratio * num_slices)

            # Select 3 slices: Peak-STRIDE, Peak, Peak+STRIDE
            indices = [anchor_idx - STRIDE, anchor_idx, anchor_idx + STRIDE]

            for i in indices:
                # Clamp index to valid range
                i = max(0, min(i, num_slices - 1))

                file_path = os.path.join(mod_path, files[i])
                img = read_dicom_slice(file_path)

                # Normalize to 0-1 float
                img = img.astype(np.float32)
                if img.max() > 0:
                    img = (img - img.min()) / (img.max() - img.min())
                else:
                    img = img * 0.0

                channels.append(img)

        # Resize all channels to target size before stacking
        resized_channels = []
        for ch in channels:
            if ch.shape[0] != IMG_SIZE or ch.shape[1] != IMG_SIZE:
                ch = cv2.resize(
                    ch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
                )
            resized_channels.append(ch)

        # Stack channels: (H, W, 12)
        image = np.stack(resized_channels, axis=-1)

        # Apply Transforms (converts to Tensor and CHW)
        augmented = self.transforms(image=image)
        image_tensor = augmented["image"]  # (12, 224, 224)

        # Label
        if "MGMT_value" in row:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)  # Test set

        return image_tensor, label


def get_dataloader(df, batch_size=BATCH_SIZE, phase="train", load_cached_data=True):
    """
    Factory function to create the DataLoader.
    Handles cache generation/loading internally.
    """
    # Prepare Cache
    anchor_cache = get_anchor_ratios(df, load_cached_data=load_cached_data)

    dataset = MGMTDataset(df, anchor_cache, phase=phase)

    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return loader
