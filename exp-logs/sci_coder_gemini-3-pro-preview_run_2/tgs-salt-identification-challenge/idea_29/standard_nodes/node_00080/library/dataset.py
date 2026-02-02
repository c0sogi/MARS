import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import pad_image

# =========================================================================
# Dataset Class
# =========================================================================


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles images, masks, and depth information.
    """

    def __init__(self, images, depths, ids, masks=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W) or (N, H, W, C).
            depths (np.ndarray): Array of normalized depth values (N, 1).
            ids (list): List of image IDs.
            masks (np.ndarray, optional): Array of binary masks (N, H, W).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as (H, W) or (H, W, C)
        image = self.images[idx]
        depth = self.depths[idx]
        image_id = self.ids[idx]

        # Ensure image is (H, W, C) for albumentations if it's currently (H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Albumentations expects uint8 for images usually, but we might have float if pre-normalized?
        # Standard flow: Load uint8 -> Augment -> Normalize -> Tensor
        # Here we assume images are uint8 [0, 255]

        data = {"image": image}

        if self.masks is not None:
            mask = self.masks[idx]
            # Ensure mask is suitable for albumentations
            data["mask"] = mask

        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if self.masks is not None:
                mask = augmented["mask"]
                # Ensure mask has channel dim for PyTorch (1, H, W) if needed,
                # but usually masks are (H, W) for loss functions like CrossEntropy/Lovasz
                # We'll keep it as (H, W) or add dim depending on loss requirement.
                # Config says masks should be (N, 1, H, W) or (N, H, W).
                # ToTensorV2 doesn't transpose masks automatically if they are 2D.
                mask = mask.long()
                # Add channel dim: (H, W) -> (1, H, W)
                mask = mask.unsqueeze(0)

        # Depth to tensor
        depth = torch.tensor([depth], dtype=torch.float32)

        sample = {
            "image": image,
            "depth": depth,
            "id": image_id,
        }

        if self.masks is not None:
            sample["mask"] = mask

        return sample


# =========================================================================
# Helper Functions
# =========================================================================


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    # ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Since input is Grayscale (1 channel), we can adapt normalization
    # or replicate channels. The Teacher uses ResNet34 pretrained on ImageNet.
    # The common trick is to replicate the grayscale image to 3 channels within the model
    # or modify the first conv layer. The idea description says:
    # "First convolution modified to accept 1-channel input (summing weights)".
    # So we pass 1-channel input. We need 1-channel normalization stats.
    # We calculated global pixel mean/std in analysis: Mean ~148/255=0.58, Std ~65/255=0.25.
    # Let's use the calculated stats for 1-channel normalization.

    # From Data Analysis:
    # Global Pixel Mean: 148.0700 -> ~0.5807
    # Global Pixel Std Dev: 65.2117 -> ~0.2557
    norm_mean = (0.5807,)
    norm_std = (0.2557,)

    if mode == "train":
        return A.Compose(
            [
                # Non-rigid transformations
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_SIGMA,  # Often linked in newer versions
                    p=Config.AUG_ELASTIC_P,
                ),
                # Rigid transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_SHIFT_SCALE_ROTATE_P,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=norm_mean, std=norm_std), ToTensorV2()])


def process_data(df, mode, cache_dir, load_cached_data=True):
    """
    Loads images/masks, pads them, and caches the result as .npy files.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'val', or 'test'.
        cache_dir (str): Directory to store cache.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
               masks will be None for test mode.
    """
    # Define cache paths
    cache_path_imgs = os.path.join(cache_dir, f"{mode}_images.npy")
    cache_path_masks = os.path.join(cache_dir, f"{mode}_masks.npy")
    cache_path_depths = os.path.join(cache_dir, f"{mode}_depths.npy")
    cache_path_ids = os.path.join(cache_dir, f"{mode}_ids.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_path_imgs)
            and os.path.exists(cache_path_ids)
            and os.path.exists(cache_path_depths)
        ):
            # Check masks existence only if not test
            if mode == "test" or os.path.exists(cache_path_masks):
                print(f"Loading {mode} data from cache: {cache_dir}")
                images = np.load(cache_path_imgs)
                depths = np.load(cache_path_depths)
                ids = np.load(cache_path_ids, allow_pickle=True)

                masks = None
                if mode != "test":
                    masks = np.load(cache_path_masks)

                return images, masks, depths, ids

    # 2. Process from Scratch
    print(f"Processing {mode} data from scratch...")

    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    # Handle DEBUG mode
    if Config.DEBUG:
        print(
            f"DEBUG mode: limiting {mode} data to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    for _, row in df.iterrows():
        # Load Image
        # Config.INPUT_ROOT is "./input"
        # row['image_path'] is relative, e.g., "train/images/xxxx.png"
        img_path = os.path.join(Config.INPUT_ROOT, row["image_path"])

        # Read as Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Pad Image (101 -> 128)
        img = pad_image(img)
        images_list.append(img)

        # Load Mask (if exists)
        if mode != "test" and "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = os.path.join(Config.INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback for empty masks if file missing?
                # Metadata check passed, so file should exist.
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            # Normalize mask to 0/1
            mask = (mask > 127).astype(np.uint8)
            # Pad Mask
            mask = pad_image(mask)
            masks_list.append(mask)

        # Depth
        depths_list.append(row["z"])
        ids_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)  # (N, 128, 128)
    depths = np.array(depths_list, dtype=np.float32)
    ids = np.array(ids_list)

    masks = None
    if mode != "test":
        masks = np.array(masks_list, dtype=np.uint8)  # (N, 128, 128)

    # 3. Save to Cache
    print(f"Saving {mode} data to cache: {cache_dir}")
    np.save(cache_path_imgs, images)
    np.save(cache_path_depths, depths)
    np.save(cache_path_ids, ids)
    if masks is not None:
        np.save(cache_path_masks, masks)

    return images, masks, depths, ids


def get_loaders(load_cached_data=True):
    """
    Main function to generate DataLoaders for Train, Val, and Test.
    Handles data loading, caching, depth normalization, and splitting.

    Args:
        load_cached_data (bool): If True, tries to load pre-processed numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Process Data (Load Images/Masks, Pad, Cache)
    # We process each split independently
    train_imgs, train_masks, train_depths, train_ids = process_data(
        train_df, "train", Config.CACHE_DIR, load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = process_data(
        val_df, "val", Config.CACHE_DIR, load_cached_data
    )
    test_imgs, test_masks, test_depths, test_ids = process_data(
        test_df, "test", Config.CACHE_DIR, load_cached_data
    )

    # 3. Depth Normalization
    # Calculate stats from Training set ONLY
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)

    # Avoid division by zero
    if depth_std == 0:
        depth_std = 1.0

    # Apply Standard Scaling
    train_depths = (train_depths - depth_mean) / depth_std
    val_depths = (val_depths - depth_mean) / depth_std
    test_depths = (test_depths - depth_mean) / depth_std

    # Save depth stats for inference reference if needed (optional but good practice)
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")
    if not os.path.exists(stats_path):
        pd.DataFrame({"mean": [depth_mean], "std": [depth_std]}).to_csv(
            stats_path, index=False
        )

    # 4. Create Datasets
    train_dataset = SaltDataset(
        images=train_imgs,
        depths=train_depths,
        ids=train_ids,
        masks=train_masks,
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        images=val_imgs,
        depths=val_depths,
        ids=val_ids,
        masks=val_masks,
        transform=get_transforms("val"),
    )

    test_dataset = SaltDataset(
        images=test_imgs,
        depths=test_depths,
        ids=test_ids,
        masks=None,
        transform=get_transforms("test"),
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
