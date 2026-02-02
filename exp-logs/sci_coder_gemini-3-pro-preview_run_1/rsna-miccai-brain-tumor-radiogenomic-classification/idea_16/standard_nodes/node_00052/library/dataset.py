import os
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_dicom_image, log_message


def get_transforms(split):
    """
    Returns the Albumentations transform pipeline for the specified split.
    Applies geometric augmentations to the 9-channel slab during training.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            ]
        )
    else:
        return None


def get_median_slice_index(folder_path):
    """
    Scans a directory for DICOM files, extracts instance numbers,
    and returns the median index. Returns 0 if no files found.
    """
    if not os.path.exists(folder_path):
        return 0

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    if not files:
        return 0

    nums = []
    for f in files:
        try:
            # Format is usually Image-{n}.dcm
            n = int(f.split("-")[1].split(".")[0])
            nums.append(n)
        except (IndexError, ValueError):
            pass

    if not nums:
        return 0

    nums.sort()
    return nums[len(nums) // 2]


def load_slab_slices(folder_path, center_idx, depth, image_size):
    """
    Loads 'depth' consecutive slices centered at 'center_idx'.
    Handles missing files by clamping to the nearest valid index.
    Applies resizing and independent min-max normalization.
    """
    # Identify all valid indices in the folder to handle clamping
    valid_indices = []
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
        for f in files:
            try:
                n = int(f.split("-")[1].split(".")[0])
                valid_indices.append(n)
            except:
                pass
    valid_indices.sort()

    if not valid_indices:
        # Return zeros if folder is empty
        return [
            np.zeros((image_size, image_size), dtype=np.float32) for _ in range(depth)
        ]

    min_idx, max_idx = valid_indices[0], valid_indices[-1]

    # Calculate range: e.g., depth 3 centered at C -> C-1, C, C+1
    start_idx = center_idx - (depth // 2)
    indices_to_load = [start_idx + i for i in range(depth)]

    slices = []
    for idx in indices_to_load:
        # Clamp index to valid range to avoid missing files (Thick Slab continuity)
        effective_idx = max(min_idx, min(max_idx, idx))

        fpath = os.path.join(folder_path, f"Image-{effective_idx}.dcm")
        img = load_dicom_image(fpath)

        if img is None:
            img = np.zeros((image_size, image_size), dtype=np.float32)
        else:
            img = cv2.resize(img, (image_size, image_size))

        # Independent Channel Min-Max Scaling to [0, 1]
        mi, ma = img.min(), img.max()
        if ma - mi > 1e-6:
            img = (img - mi) / (ma - mi)
        else:
            # If constant image (e.g. all black), set to 0
            img = np.zeros_like(img)

        slices.append(img)

    return slices


def process_and_cache_data(
    metadata_path, cache_img_path, cache_lbl_path, cache_id_path, load_cached=True
):
    """
    Main data processing function.
    1. Checks if cached .npy files exist and load_cached is True.
    2. If not, reads metadata, performs Deterministic Data Expansion,
       loads images, normalizes, and saves to cache.
    """

    # 1. Attempt to load from cache
    if load_cached and os.path.exists(cache_img_path) and os.path.exists(cache_id_path):
        # Check label cache only if it's expected (not None)
        if cache_lbl_path is None or os.path.exists(cache_lbl_path):
            log_message(
                f"Loading cached data from {os.path.dirname(cache_img_path)}..."
            )
            images = np.load(cache_img_path)
            ids = np.load(cache_id_path)
            labels = np.load(cache_lbl_path) if cache_lbl_path else None
            return images, labels, ids

    # 2. Process from scratch
    log_message(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)
        log_message(f"DEBUG mode: Processing first {len(df)} subjects only.")

    all_images = []
    all_labels = []
    all_ids = []

    # Offsets for expansion: [-5, 0, +5]
    offsets = [-Config.SLAB_STRIDE, 0, Config.SLAB_STRIDE]

    count = 0
    total = len(df)

    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]
        has_label = "MGMT_value" in row
        label = row["MGMT_value"] if has_label else -1.0

        # 1. Independent Heuristic Alignment: Find median for each modality
        medians = {}
        for mod in Config.MODALITIES:  # ["FLAIR", "T1wCE", "T2w"]
            # Construct full path. Metadata contains relative paths e.g. "train/00000/FLAIR"
            col_name = f"{mod.lower()}_path"
            rel_path = row[col_name]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            medians[mod] = get_median_slice_index(full_path)

        # 2. Deterministic Data Expansion: Create 3 slabs
        for offset in offsets:
            slab_channels = []

            # Load slices for each modality
            for mod in Config.MODALITIES:
                rel_path = row[f"{mod.lower()}_path"]
                full_path = os.path.join(Config.INPUT_DIR, rel_path)

                # Center for this slab = Median + Offset
                center_idx = medians[mod] + offset

                # Load 3 slices (z-1, z, z+1)
                slices = load_slab_slices(
                    full_path, center_idx, Config.SLAB_DEPTH, Config.IMAGE_SIZE
                )
                slab_channels.extend(slices)

            # Stack to (9, H, W)
            slab_tensor = np.stack(slab_channels, axis=0)

            all_images.append(slab_tensor)
            all_ids.append(subject_id)
            if has_label:
                all_labels.append(label)

        count += 1
        if count % 50 == 0:
            log_message(f"Processed {count}/{total} subjects...")

    # Convert to numpy arrays
    # Shape: (N_samples, 9, H, W)
    images_np = np.array(all_images, dtype=np.float32)
    ids_np = np.array(all_ids, dtype=np.int64)
    labels_np = np.array(all_labels, dtype=np.float32) if all_labels else None

    # Save to cache
    log_message(f"Saving cache to {Config.WORKING_DIR}...")
    np.save(cache_img_path, images_np)
    np.save(cache_id_path, ids_np)
    if labels_np is not None and cache_lbl_path is not None:
        np.save(cache_lbl_path, labels_np)

    return images_np, labels_np, ids_np


class WIISDataset(Dataset):
    """
    Dataset wrapper for the Weight-Inflated Independent-Slab Network.
    Accepts pre-loaded numpy arrays.
    """

    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image shape: (9, H, W)
        img = self.images[idx]

        # Albumentations expects (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        if self.transform:
            res = self.transform(image=img)
            img = res["image"]

        # Transpose back to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        # Convert to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)
        bra_id = self.ids[idx]

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return img_tensor, label, bra_id
        else:
            return img_tensor, bra_id


def get_dataloader(split, batch_size=Config.BATCH_SIZE, shuffle=True, load_cached=True):
    """
    Factory function to create DataLoaders for train, val, or test splits.
    Handles path resolution and caching logic.
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        c_img = Config.CACHE_TRAIN_IMAGES
        c_lbl = Config.CACHE_TRAIN_LABELS
        c_ids = Config.CACHE_TRAIN_IDS
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        c_img = Config.CACHE_VAL_IMAGES
        c_lbl = Config.CACHE_VAL_LABELS
        c_ids = Config.CACHE_VAL_IDS
        shuffle = False  # Usually don't shuffle val
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        c_img = Config.CACHE_TEST_IMAGES
        c_lbl = None  # No labels for test
        c_ids = Config.CACHE_TEST_IDS
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load data (cached or processed)
    images, labels, ids = process_and_cache_data(
        meta_path, c_img, c_lbl, c_ids, load_cached=load_cached
    )

    # Create Dataset
    dataset = WIISDataset(images, labels, ids, transform=get_transforms(split))

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
    )

    return loader
