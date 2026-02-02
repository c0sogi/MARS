import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library import config, utils


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata and images, caching the result as .npy files.
    """
    # Define cache paths
    img_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_masks.npy")
    depth_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_depths.npy")
    id_cache_path = os.path.join(config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(depth_cache_path)
        and os.path.exists(id_cache_path)
    ):
        # Check mask cache existence only if masks are expected (train/val)
        # For simplicity, we check if mask cache exists or if it's test set (which might not have masks)
        # However, we'll rely on the file existence logic below.

        try:
            images = np.load(img_cache_path)
            depths = np.load(depth_cache_path)
            ids = np.load(id_cache_path)
            masks = (
                np.load(mask_cache_path) if os.path.exists(mask_cache_path) else None
            )
            return images, masks, depths, ids
        except Exception as e:
            print(
                f"Failed to load cache for {cache_prefix}: {e}. Reloading from source."
            )

    # Load from source
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    ids = df["id"].values
    # Fill NaNs in depth if any (though dataset desc says 0 nans)
    depths = df["z"].fillna(0).values.astype(np.float32)

    images = []
    masks = []
    has_masks = "mask_path" in df.columns and not df["mask_path"].isnull().all()

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(config.INPUT_DIR, row["image_path"])
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

        # Load Mask if available
        if has_masks:
            mask_path = os.path.join(config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks.append(mask)

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    np.save(img_cache_path, images)

    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
        # Ensure binary masks (0, 1)
        masks = (masks > 127).astype(np.uint8)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        ids,
        transform=None,
        depth_mean=0.0,
        depth_std=1.0,
        mode="train",
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            masks (np.ndarray): Array of masks (N, H, W) or None.
            depths (np.ndarray): Array of depths (N,).
            ids (np.ndarray): Array of IDs.
            transform (albumentations.Compose): Augmentation pipeline.
            depth_mean (float): Mean for depth normalization.
            depth_std (float): Std for depth normalization.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Normalize depth
        d_val = self.depths[idx]
        d_norm = (d_val - self.depth_mean) / self.depth_std
        depth_tensor = torch.tensor([d_norm], dtype=torch.float32)

        if self.mode == "test":
            if self.transform:
                augmented = self.transform(image=img)
                img = augmented["image"]

            # Convert to tensor and normalize to [0, 1]
            # Albumentations PadIfNeeded returns numpy array.
            # We assume transform pipeline does not include ToTensorV2 to allow manual float conversion here.
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

            return img_tensor, depth_tensor, self.ids[idx]

        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return img_tensor, mask_tensor, depth_tensor


def get_transforms(phase):
    """
    Returns the albumentations transform pipeline for the specified phase.
    """
    # Base transform: Pad to 128x128 using reflection
    transforms = [
        A.PadIfNeeded(
            min_height=config.IMG_TARGET_SIZE,
            min_width=config.IMG_TARGET_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            always_apply=True,
        )
    ]

    if phase == "train":
        # Add augmentations for training
        transforms.extend(
            [
                A.ElasticTransform(
                    alpha=config.AUG_ELASTIC_ALPHA,
                    sigma=config.AUG_ELASTIC_SIGMA,
                    border_mode=cv2.BORDER_REFLECT,
                    p=config.AUG_PROB,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

    return A.Compose(transforms)


def get_train_val_loaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation sets.
    Calculates depth statistics from the training set.
    """
    # Load raw data
    train_imgs, train_masks, train_depths, train_ids = load_and_cache_data(
        config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = load_and_cache_data(
        config.VAL_METADATA_PATH, "val", load_cached_data
    )

    # Calculate depth statistics from training set
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)

    # Avoid division by zero
    if depth_std == 0:
        depth_std = 1.0

    # Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transform=get_transforms("train"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        mode="train",
    )

    val_dataset = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transform=get_transforms("val"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        mode="val",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, depth_mean, depth_std


def get_test_loader(depth_mean, depth_std, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    Requires depth statistics from the training set for consistent normalization.
    """
    test_imgs, _, test_depths, test_ids = load_and_cache_data(
        config.TEST_METADATA_PATH, "test", load_cached_data
    )

    test_dataset = SaltDataset(
        test_imgs,
        None,
        test_depths,
        test_ids,
        transform=get_transforms("test"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
