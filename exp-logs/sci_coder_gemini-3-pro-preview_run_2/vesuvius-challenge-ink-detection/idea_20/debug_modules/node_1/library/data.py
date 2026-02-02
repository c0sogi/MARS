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

# --- Helper Functions ---


def load_fragment_volume(
    fragment_id, volume_dir, start_slice, end_slice, load_cached_data=True
):
    """
    Loads a chunk of the 3D volume for a specific fragment.
    Caches the result as a .npy file in Config.WORKING_DIR.

    Args:
        fragment_id (str): ID of the fragment.
        volume_dir (str): Relative path to the volume directory.
        start_slice (int): Start Z-index (inclusive).
        end_slice (int): End Z-index (exclusive).
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: 3D volume of shape (D, H, W).
    """
    cache_filename = f"frag_{fragment_id}_slices_{start_slice}_{end_slice}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            return volume
        except Exception as e:
            print(f"Failed to load cache for {fragment_id}: {e}")

    # 2. Load from Disk (TIFs)
    slices = []
    # Construct full path to volume directory
    full_vol_dir = os.path.join(Config.INPUT_DIR, volume_dir)

    for i in range(start_slice, end_slice):
        filename = f"{i:02d}.tif"
        path = os.path.join(full_vol_dir, filename)

        if not os.path.exists(path):
            # If a slice is missing, we raise an error as data integrity is expected.
            raise FileNotFoundError(f"Slice {path} not found.")

        # Load as uint16
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image {path}")
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (D, H, W)

    # 3. Save to Cache
    try:
        np.save(cache_path, volume)
    except Exception as e:
        print(f"Could not save cache to {cache_path}: {e}")

    return volume


def generate_test_patches(test_metadata_path):
    """
    Generates a dataframe of patches for the test set by tiling the fragments.

    Args:
        test_metadata_path (str): Path to test.csv.

    Returns:
        pd.DataFrame: DataFrame containing patch coordinates.
    """
    df_test_meta = pd.read_csv(test_metadata_path)
    patches = []

    for _, row in df_test_meta.iterrows():
        frag_id = row["fragment_id"]
        mask_path_rel = row["mask_path"]
        volume_path_rel = row["volume_path"]

        full_mask_path = os.path.join(Config.INPUT_DIR, mask_path_rel)

        # Load mask to get dimensions
        mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape

        # Tile the image
        for y in range(0, h, Config.STRIDE):
            for x in range(0, w, Config.STRIDE):
                patches.append(
                    {
                        "fragment_id": frag_id,
                        "x": x,
                        "y": y,
                        "width": Config.TILE_SIZE,
                        "height": Config.TILE_SIZE,
                        "mask_path": mask_path_rel,
                        "volume_path": volume_path_rel,
                    }
                )

    return pd.DataFrame(patches)


# --- Dataset Class ---


class InkDataset(Dataset):
    def __init__(self, dataframe, split="train", transform=None, load_cached_data=True):
        """
        Dataset for Vesuvius Ink Detection.

        Args:
            dataframe (pd.DataFrame): Metadata for patches.
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Augmentations.
            load_cached_data (bool): Whether to use cached volumes.
        """
        self.df = dataframe.reset_index(drop=True)
        self.split = split
        self.transform = transform

        # Define Z-range to load into memory.
        # We need to support Views starting at 16, 20, 24 with depth 12.
        # Max index needed: 24 + 12 = 36.
        # We load range [16, 40) to be safe and cover all views.
        self.z_min = 16
        self.z_max = 40

        self.volumes = {}
        self.masks = {}
        self.labels = {}

        unique_frags = self.df["fragment_id"].unique()

        for fid in unique_frags:
            # Get paths from first occurrence
            row = self.df[self.df["fragment_id"] == fid].iloc[0]

            # Load Volume
            vol_path = row["volume_path"]
            self.volumes[fid] = load_fragment_volume(
                fid, vol_path, self.z_min, self.z_max, load_cached_data
            )

            # Load Mask (Binary Valid Mask)
            mask_p = os.path.join(Config.INPUT_DIR, row["mask_path"])
            self.masks[fid] = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            # Load Label (Ink Label) - Only if available and not test
            if split != "test":
                # Check if label_path exists in row and is not NaN
                if "label_path" in row and pd.notna(row["label_path"]):
                    label_p = os.path.join(Config.INPUT_DIR, row["label_path"])
                    self.labels[fid] = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 1. Determine Z-View (Discrete Multi-View Protocol)
        if self.split == "train":
            # Randomly select view for translation invariance
            view_start = np.random.choice(
                [Config.VIEW_A_START, Config.VIEW_B_START, Config.VIEW_C_START]
            )
        else:
            # Validation/Test: Use Center View (B) for deterministic output
            view_start = Config.VIEW_B_START

        # Calculate local indices in the loaded volume
        # Loaded volume starts at self.z_min
        local_start = view_start - self.z_min
        local_end = local_start + Config.SLAB_DEPTH  # 12 slices

        # 2. Extract Volume Crop
        full_vol = self.volumes[fid]  # (D_loaded, H_frag, W_frag)
        frag_h, frag_w = self.masks[fid].shape

        # Calculate crop boundaries with padding handling
        crop_x_end = min(x + w, frag_w)
        crop_y_end = min(y + h, frag_h)

        pad_right = max(0, x + w - frag_w)
        pad_bottom = max(0, y + h - frag_h)

        # Crop Z, Y, X
        vol_crop = full_vol[local_start:local_end, y:crop_y_end, x:crop_x_end]

        # Pad if necessary (for edge tiles)
        if pad_right > 0 or pad_bottom > 0:
            vol_crop = np.pad(
                vol_crop,
                ((0, 0), (0, pad_bottom), (0, pad_right)),
                mode="constant",
                constant_values=0,
            )

        # 3. Channel Projection (Average Pooling)
        # 12 slices -> 3 channels (4 slices each)
        ch1 = np.mean(vol_crop[0:4], axis=0)
        ch2 = np.mean(vol_crop[4:8], axis=0)
        ch3 = np.mean(vol_crop[8:12], axis=0)

        image = np.stack([ch1, ch2, ch3], axis=-1)  # (H, W, 3)

        # 4. Normalize (uint16 -> [0, 1])
        image = image.astype(np.float32) / 65535.0

        # 5. Get Mask and Label
        # Mask (Valid area)
        mask_full = self.masks[fid]
        mask_crop = mask_full[y:crop_y_end, x:crop_x_end]
        if pad_right > 0 or pad_bottom > 0:
            mask_crop = np.pad(
                mask_crop,
                ((0, pad_bottom), (0, pad_right)),
                mode="constant",
                constant_values=0,
            )

        # Label (Ink)
        label_crop = None
        if self.split != "test" and fid in self.labels:
            label_full = self.labels[fid]
            lbl_c = label_full[y:crop_y_end, x:crop_x_end]
            if pad_right > 0 or pad_bottom > 0:
                lbl_c = np.pad(
                    lbl_c,
                    ((0, pad_bottom), (0, pad_right)),
                    mode="constant",
                    constant_values=0,
                )
            label_crop = (lbl_c > 0).astype(np.float32)

        # 6. Augmentation
        if self.transform:
            # Albumentations
            # Augment both label (if present) and valid_mask to ensure geometric consistency
            masks_to_aug = []
            if label_crop is not None:
                masks_to_aug.append(label_crop)
            masks_to_aug.append(mask_crop)

            data = {
                "image": image,
                "masks": masks_to_aug,
            }
            augmented = self.transform(**data)
            image = augmented["image"]
            aug_masks = augmented["masks"]

            if label_crop is not None:
                label_crop = aug_masks[0].unsqueeze(0)  # (1, H, W)
                mask_crop = aug_masks[1]
            else:
                mask_crop = aug_masks[0]

            # mask_crop is now a tensor from ToTensorV2, ensure (1, H, W)
            mask_tensor = mask_crop.unsqueeze(0).float()
        else:
            # Manual ToTensor
            image = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)
            if label_crop is not None:
                label_crop = torch.from_numpy(label_crop).unsqueeze(0)  # (1, H, W)

            # mask_crop is numpy
            mask_tensor = torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(0)

        # Prepare Output
        output = {"image": image, "fragment_id": fid, "x": x, "y": y}

        if label_crop is not None:
            output["label"] = label_crop

        # Ensure binary valid mask
        mask_tensor = (mask_tensor > 0).float()
        output["valid_mask"] = mask_tensor

        return output


# --- Data Loader Generators ---


def get_dataloaders(load_cached_data=True):
    """
    Creates Training and Validation DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "validation.csv"))

    # Transforms (Geometric Only for Train)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # Datasets
    train_ds = InkDataset(
        train_df,
        split="train",
        transform=train_transform,
        load_cached_data=load_cached_data,
    )

    val_ds = InkDataset(
        val_df,
        split="validation",
        transform=val_transform,
        load_cached_data=load_cached_data,
    )

    # Loaders
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


def get_test_dataloader(load_cached_data=True):
    """
    Creates Test DataLoader.
    """
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
    if not os.path.exists(test_meta_path):
        raise FileNotFoundError("Test metadata not found.")

    test_df = generate_test_patches(test_meta_path)

    transform = A.Compose([ToTensorV2()])

    test_ds = InkDataset(
        test_df, split="test", transform=transform, load_cached_data=load_cached_data
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
