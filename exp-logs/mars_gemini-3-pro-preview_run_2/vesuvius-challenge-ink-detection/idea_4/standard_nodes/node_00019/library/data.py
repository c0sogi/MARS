import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specific phase.
    Strictly geometric augmentations for training to preserve radiodensity values.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def load_volume_slice(volume_dir, z_index):
    """
    Loads a single slice from the volume directory.
    """
    filename = f"{z_index:02d}.tif"
    path = os.path.join(volume_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Slice file not found: {path}")

    # Load as grayscale (uint16)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def get_fragment_mips(fragment_id, volume_path, load_cached_data=True):
    """
    Computes or loads the 4-channel Hybrid-Projection MIPs for a fragment.
    Channels: Global (22-42), Slab1 (22-28), Slab2 (29-35), Slab3 (36-42).

    Args:
        fragment_id (str): ID of the fragment.
        volume_path (str): Relative path to the volume directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: Shape (H, W, 4), float32, normalized to [0, 1].
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{fragment_id}_mips.npy")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache for {fragment_id}: {e}. Recomputing.")

    # 2. Compute from Scratch

    # Determine the full range of slices needed (union of all ranges)
    # Ranges are exclusive in python: 22-43 means 22..42
    all_z = sorted(list(set(list(Config.Z_GLOBAL_RANGE))))
    min_z = min(all_z)
    max_z = max(all_z)

    full_volume_path = os.path.join(Config.INPUT_DIR, volume_path)

    # Load all required slices into a stack
    loaded_stack = []
    for z in range(min_z, max_z + 1):
        img = load_volume_slice(full_volume_path, z)
        if img is None:
            raise FileNotFoundError(f"Slice {z} missing in {full_volume_path}")
        loaded_stack.append(img)

    # Stack: (H, W, Depth)
    # Depth corresponds to range(min_z, max_z + 1)
    volume_block = np.stack(loaded_stack, axis=-1)

    # Helper to map global Z index to local block index
    def get_rel_indices(z_range):
        return [z - min_z for z in z_range]

    idx_slab1 = get_rel_indices(Config.Z_SLAB1_RANGE)
    idx_slab2 = get_rel_indices(Config.Z_SLAB2_RANGE)
    idx_slab3 = get_rel_indices(Config.Z_SLAB3_RANGE)

    # Compute Maximum Intensity Projections (MIPs)
    # Cite solution_lesson_node_00018: Prefer strictly stratified projections over hybrid global-local summaries.
    mip_slab1 = np.max(volume_block[..., idx_slab1], axis=-1)
    mip_slab2 = np.max(volume_block[..., idx_slab2], axis=-1)
    mip_slab3 = np.max(volume_block[..., idx_slab3], axis=-1)

    # Stack Channels -> (H, W, 3)
    mips = np.stack([mip_slab1, mip_slab2, mip_slab3], axis=-1)

    # Normalize uint16 to [0, 1] float32
    mips = mips.astype(np.float32) / 65535.0

    # Save to Cache
    np.save(cache_path, mips)

    return mips


class InkDataset(Dataset):
    """
    Dataset for Vesuvius Ink Detection.
    Loads pre-computed 4-channel MIPs and crops patches based on metadata.
    """

    def __init__(self, metadata_df, phase="train", load_cached_data=True):
        self.metadata = metadata_df.reset_index(drop=True)
        self.phase = phase
        self.transform = get_transforms(phase)

        # Pre-load/Cache Fragment Images and Masks
        # This ensures we don't reload heavy files for every patch
        self.fragment_images = {}
        self.fragment_masks = {}
        self.fragment_labels = {}

        unique_fragments = self.metadata["fragment_id"].unique()

        for fid in unique_fragments:
            # Get one row for this fragment to extract paths
            row = self.metadata[self.metadata["fragment_id"] == fid].iloc[0]

            # Load 4-channel MIPs
            self.fragment_images[fid] = get_fragment_mips(
                fid, row["volume_path"], load_cached_data=load_cached_data
            )

            # Load Mask (Valid Area)
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.fragment_masks[fid] = (mask > 0).astype(np.float32)

            # Load Label (Ink) if available (Train/Val)
            if phase in ["train", "val"]:
                label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                self.fragment_labels[fid] = (label > 0).astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Retrieve full fragment data
        full_image = self.fragment_images[fid]  # (H_full, W_full, 4)

        # Crop patch
        image = full_image[y : y + h, x : x + w, :]

        data = {"image": image}

        if self.phase in ["train", "val"]:
            full_label = self.fragment_labels[fid]
            label = full_label[y : y + h, x : x + w]

            # Albumentations expects 'mask' for segmentation targets
            data["mask"] = label

            if self.transform:
                augmented = self.transform(**data)
                image = augmented["image"]
                label = augmented["mask"]

            # Convert label to (1, H, W) tensor
            label = label.unsqueeze(0).float()

            return image, label

        else:
            # Inference mode (if used with patch metadata)
            if self.transform:
                augmented = self.transform(**data)
                image = augmented["image"]
            return image


def get_loaders(train_df, val_df, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    """
    set_seed(Config.SEED)

    train_ds = InkDataset(train_df, phase="train", load_cached_data=load_cached_data)
    val_ds = InkDataset(val_df, phase="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_fragments(test_df, load_cached_data=True):
    """
    Loads and processes all test fragments for inference.

    Returns:
        dict: {fragment_id: {'image': np.ndarray (H,W,4), 'mask': np.ndarray (H,W)}}
    """
    data = {}
    for _, row in test_df.iterrows():
        fid = row["fragment_id"]
        vol_path = row["volume_path"]
        mask_path = row["mask_path"]

        # Load processed 4-channel image
        image = get_fragment_mips(fid, vol_path, load_cached_data)

        # Load binary mask
        full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        data[fid] = {"image": image, "mask": mask}
    return data
