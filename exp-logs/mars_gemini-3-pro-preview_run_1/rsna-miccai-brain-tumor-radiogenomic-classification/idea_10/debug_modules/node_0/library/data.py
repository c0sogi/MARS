import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    CACHE_DIR,
    EXPANSION_DEPTHS,
    IMAGE_SIZE,
    INPUT_DIR,
    SEED,
)
from library.utils import (
    get_brain_depth_range,
    read_dicom_slab,
    independent_slab_normalize,
    set_seed,
)

# Ensure reproducibility
set_seed(SEED)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline.
    Ensures geometric consistency across all 3 modality streams.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Elastic and Grid distortions for volumetric variation simulation
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
            ],
            additional_targets={"t1wce": "image", "t2w": "image"},
        )
    return None


class TMSVDataset(Dataset):
    """
    Dataset class for Tri-Stream Modality-Specific Volumetric Network.
    Wraps pre-processed numpy arrays and applies synchronized augmentations.
    """

    def __init__(self, flair_data, t1wce_data, t2w_data, targets, ids, transform=None):
        self.flair_data = flair_data
        self.t1wce_data = t1wce_data
        self.t2w_data = t2w_data
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Retrieve slabs (H, W, 3)
        img_flair = self.flair_data[idx]
        img_t1wce = self.t1wce_data[idx]
        img_t2w = self.t2w_data[idx]

        # Apply synchronized augmentations
        if self.transform:
            augmented = self.transform(image=img_flair, t1wce=img_t1wce, t2w=img_t2w)
            img_flair = augmented["image"]
            img_t1wce = augmented["t1wce"]
            img_t2w = augmented["t2w"]

        # Convert to PyTorch format: (H, W, C) -> (C, H, W)
        # Input to model expects (B, 3, H, W)
        img_flair = np.transpose(img_flair, (2, 0, 1))
        img_t1wce = np.transpose(img_t1wce, (2, 0, 1))
        img_t2w = np.transpose(img_t2w, (2, 0, 1))

        # Convert to tensors
        sample = {
            "flair": torch.tensor(img_flair, dtype=torch.float32),
            "t1wce": torch.tensor(img_t1wce, dtype=torch.float32),
            "t2w": torch.tensor(img_t2w, dtype=torch.float32),
            "BraTS21ID": self.ids[idx],
        }

        if self.targets is not None:
            sample["target"] = torch.tensor(
                self.targets[idx], dtype=torch.float32
            ).unsqueeze(0)

        return sample


def _load_or_create_cache(metadata_path, split_name, load_cached_data=True):
    """
    Handles the deterministic data expansion and caching mechanism.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_files = {
        "flair": os.path.join(CACHE_DIR, f"{split_name}_flair.npy"),
        "t1wce": os.path.join(CACHE_DIR, f"{split_name}_t1wce.npy"),
        "t2w": os.path.join(CACHE_DIR, f"{split_name}_t2w.npy"),
        "targets": os.path.join(CACHE_DIR, f"{split_name}_targets.npy"),
        "ids": os.path.join(CACHE_DIR, f"{split_name}_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print(f"Loading {split_name} data from cache ({CACHE_DIR})...")
        flair_arr = np.load(cache_files["flair"])
        t1wce_arr = np.load(cache_files["t1wce"])
        t2w_arr = np.load(cache_files["t2w"])
        targets_arr = (
            np.load(cache_files["targets"])
            if os.path.exists(cache_files["targets"])
            else None
        )
        ids_arr = np.load(cache_files["ids"])

        # Handle case where targets might be None (test set) but file exists as None object or empty
        if split_name == "test":
            targets_arr = None

        return flair_arr, t1wce_arr, t2w_arr, targets_arr, ids_arr

    # Process data from scratch
    print(f"Processing {split_name} data from metadata...")
    df = pd.read_csv(metadata_path)

    flair_list = []
    t1wce_list = []
    t2w_list = []
    targets_list = []
    ids_list = []

    for _, row in df.iterrows():
        subject_id = row["BraTS21ID"]

        # Construct full paths
        flair_path = os.path.join(INPUT_DIR, row["flair_path"])
        t1wce_path = os.path.join(INPUT_DIR, row["t1wce_path"])
        t2w_path = os.path.join(INPUT_DIR, row["t2w_path"])

        # Determine Brain-Centric ROI using FLAIR (structural reference)
        min_idx, max_idx = get_brain_depth_range(flair_path)

        # Fallback if ROI detection fails (e.g., completely black images)
        if min_idx == 0 and max_idx == 0:
            # Attempt to find range from file count if pixel check failed
            try:
                files = [f for f in os.listdir(flair_path) if f.endswith(".dcm")]
                if files:
                    # Simple heuristic: assume middle 50% is brain
                    file_indices = sorted(
                        [
                            int(f.replace("Image-", "").replace(".dcm", ""))
                            for f in files
                        ]
                    )
                    if file_indices:
                        min_idx = file_indices[int(len(file_indices) * 0.25)]
                        max_idx = file_indices[int(len(file_indices) * 0.75)]
            except Exception:
                pass

        # Deterministic Expansion
        for depth_ratio in EXPANSION_DEPTHS:
            # Calculate center slice index
            if max_idx > min_idx:
                depth_span = max_idx - min_idx
                center_idx = int(min_idx + (depth_span * depth_ratio))
            else:
                # Fallback for single slice or error
                center_idx = min_idx

            # Read and Normalize Slabs
            # Note: utils.read_dicom_slab returns (H, W, D)
            slab_flair = read_dicom_slab(flair_path, center_idx)
            slab_t1wce = read_dicom_slab(t1wce_path, center_idx)
            slab_t2w = read_dicom_slab(t2w_path, center_idx)

            norm_flair = independent_slab_normalize(slab_flair)
            norm_t1wce = independent_slab_normalize(slab_t1wce)
            norm_t2w = independent_slab_normalize(slab_t2w)

            flair_list.append(norm_flair)
            t1wce_list.append(norm_t1wce)
            t2w_list.append(norm_t2w)
            ids_list.append(subject_id)

            if "MGMT_value" in row:
                targets_list.append(row["MGMT_value"])

    # Convert to numpy arrays
    flair_arr = np.array(flair_list, dtype=np.float32)
    t1wce_arr = np.array(t1wce_list, dtype=np.float32)
    t2w_arr = np.array(t2w_list, dtype=np.float32)
    ids_arr = np.array(ids_list, dtype=np.int64)

    if targets_list:
        targets_arr = np.array(targets_list, dtype=np.float32)
    else:
        targets_arr = None

    # Save to cache
    print(f"Saving {split_name} data to cache...")
    np.save(cache_files["flair"], flair_arr)
    np.save(cache_files["t1wce"], t1wce_arr)
    np.save(cache_files["t2w"], t2w_arr)
    np.save(cache_files["ids"], ids_arr)
    if targets_arr is not None:
        np.save(cache_files["targets"], targets_arr)

    return flair_arr, t1wce_arr, t2w_arr, targets_arr, ids_arr


def get_datasets(load_cached_data=True):
    """
    Factory function to create Train, Validation, and Test datasets.
    Handles caching and transformation assignment.
    """
    # 1. Train Data
    print("Preparing Training Dataset...")
    tr_flair, tr_t1wce, tr_t2w, tr_targets, tr_ids = _load_or_create_cache(
        TRAIN_METADATA_PATH, "train", load_cached_data
    )
    train_dataset = TMSVDataset(
        tr_flair,
        tr_t1wce,
        tr_t2w,
        tr_targets,
        tr_ids,
        transform=get_transforms("train"),
    )

    # 2. Validation Data
    print("Preparing Validation Dataset...")
    val_flair, val_t1wce, val_t2w, val_targets, val_ids = _load_or_create_cache(
        VAL_METADATA_PATH, "val", load_cached_data
    )
    val_dataset = TMSVDataset(
        val_flair,
        val_t1wce,
        val_t2w,
        val_targets,
        val_ids,
        transform=None,  # No augmentation for validation
    )

    # 3. Test Data
    print("Preparing Test Dataset...")
    te_flair, te_t1wce, te_t2w, te_targets, te_ids = _load_or_create_cache(
        TEST_METADATA_PATH, "test", load_cached_data
    )
    test_dataset = TMSVDataset(
        te_flair,
        te_t1wce,
        te_t2w,
        te_targets,
        te_ids,
        transform=None,  # No augmentation for testing
    )

    return train_dataset, val_dataset, test_dataset
