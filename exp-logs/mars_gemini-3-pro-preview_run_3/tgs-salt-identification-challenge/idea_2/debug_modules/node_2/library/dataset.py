import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import pad_image, rle_decode


def preprocess_data(df, split_name, load_cached_data=True):
    """
    Loads, preprocesses, and caches data.

    Logic:
    1. Construct cache paths.
    2. If load_cached_data is True and files exist, load and return.
    3. Else, iterate through dataframe:
       - Load image (Grayscale -> RGB).
       - Create Depth channel (normalized).
       - Stack to 4 channels.
       - Pad image.
       - Load and pad mask (if available).
    4. Save to cache if load_cached_data is True.
    5. Return arrays.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    images_cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_images.npy")
    masks_cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_masks.npy")
    ids_cache_path = os.path.join(Config.CACHE_DIR, f"{split_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(images_cache_path) and os.path.exists(ids_cache_path):
            # If masks are expected (train/val) check for them, else ignore
            has_masks = "rle_mask" in df.columns
            if not has_masks or os.path.exists(masks_cache_path):
                print(f"Loading {split_name} data from cache...")
                images = np.load(images_cache_path)
                ids = np.load(ids_cache_path, allow_pickle=True)

                # Check consistency
                if len(images) == len(df):
                    masks = None
                    if has_masks:
                        masks = np.load(masks_cache_path)
                    return images, masks, ids
                else:
                    print(f"Cache size mismatch for {split_name}. Recomputing...")

    print(f"Preprocessing {split_name} data from scratch...")

    img_list = []
    mask_list = []
    id_list = []

    # Precompute constant for depth normalization
    # Using 1000.0 as a safe upper bound based on analysis (max z ~960)
    DEPTH_MAX = 1000.0

    for _, row in df.iterrows():
        image_id = row["id"]
        id_list.append(image_id)

        # 1. Load Image
        # Image path is relative to INPUT_DIR
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load as grayscale (101, 101)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Normalize Image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to RGB (Repeat channels) -> (101, 101, 3)
        img_rgb = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        # 2. Create Depth Channel
        # Normalized depth value
        z_val = row["z"] / DEPTH_MAX
        # Create constant depth plane (101, 101)
        depth_plane = np.full_like(img, z_val)

        # 3. Stack to create 4-channel input (101, 101, 4)
        img_4c = np.dstack([img_rgb, depth_plane])

        # 4. Pad Image to (128, 128, 4)
        # pad_image handles multi-channel inputs correctly using reflection
        img_padded = pad_image(img_4c)
        img_list.append(img_padded)

        # 5. Process Mask (if exists)
        if "rle_mask" in row:
            rle = row["rle_mask"]
            # Decode RLE to (101, 101)
            mask = rle_decode(rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE))
            # Pad Mask to (128, 128)
            mask_padded = pad_image(mask)
            # Binarize again just in case interpolation caused issues (though pad uses border replicate/reflect)
            # Reflection padding preserves values, so no interpolation noise usually.
            mask_list.append(mask_padded)

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list)

    masks = None
    if mask_list:
        masks = np.array(mask_list, dtype=np.float32)  # Float for BCE/Dice loss
        # Ensure mask has channel dim if needed, but PyTorch usually takes (B, H, W) for targets
        # or (B, 1, H, W). We'll keep it (N, 128, 128) and unsqueeze in Dataset or Transform.

    # Save to cache if requested
    if load_cached_data:
        print(f"Saving {split_name} data to cache...")
        np.save(images_cache_path, images)
        np.save(ids_cache_path, ids)
        if masks is not None:
            np.save(masks_cache_path, masks)

    return images, masks, ids


class SaltDataset(Dataset):
    def __init__(self, images, masks=None, ids=None, transforms=None):
        """
        Args:
            images (np.ndarray): Shape (N, 128, 128, 4)
            masks (np.ndarray): Shape (N, 128, 128) or None
            ids (np.ndarray): Array of image IDs
            transforms (albumentations.Compose): Augmentation pipeline
        """
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (128, 128, 4)

        if self.masks is not None:
            mask = self.masks[idx]  # (128, 128)

            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask is a tensor with channel dimension if not added by transform
            # ToTensorV2 usually converts image to (C, H, W) but mask to (H, W) if it's 2D.
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (1, 128, 128)

            return image, mask
        else:
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented["image"]

            # For inference, we might need the ID to construct the submission
            return image, self.ids[idx]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for Train/Val/Test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # Images are already normalized to [0, 1] and 4-channel.
                # ToTensorV2 converts HWC to CHW and to torch.Tensor.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_dataloaders(debug=False, batch_size=None, num_workers=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        debug (bool): If True, uses a small subset of data and disables caching.
        batch_size (int): Override config batch size.
        num_workers (int): Override config num workers.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Configuration overrides
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 2. Handle Debug / Max Samples
    # If Config.MAX_SAMPLES is set, it overrides debug flag conceptually
    max_samples = Config.MAX_SAMPLES

    use_cache = True

    if debug or max_samples is not None:
        use_cache = False  # Disable cache for partial datasets to avoid corruption
        limit = 50 if debug else max_samples
        if limit:
            print(f"Debug/Limit mode: Using {limit} samples per split.")
            train_df = train_df.iloc[:limit]
            val_df = val_df.iloc[:limit]
            test_df = test_df.iloc[:limit]

    # 3. Preprocess Data
    # Train
    train_imgs, train_masks, train_ids = preprocess_data(
        train_df, "train", load_cached_data=use_cache
    )
    # Val
    val_imgs, val_masks, val_ids = preprocess_data(
        val_df, "val", load_cached_data=use_cache
    )
    # Test
    test_imgs, _, test_ids = preprocess_data(
        test_df, "test", load_cached_data=use_cache
    )

    # 4. Create Datasets
    train_dataset = SaltDataset(
        train_imgs, train_masks, train_ids, transforms=get_transforms("train")
    )
    val_dataset = SaltDataset(
        val_imgs, val_masks, val_ids, transforms=get_transforms("val")
    )
    test_dataset = SaltDataset(
        test_imgs, None, test_ids, transforms=get_transforms("test")
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
