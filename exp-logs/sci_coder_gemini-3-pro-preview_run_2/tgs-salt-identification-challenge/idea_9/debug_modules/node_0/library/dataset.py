import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
INPUT_ROOT = "./input"
CACHE_DIR = "./working/idea_9"
IMG_SIZE = 101
# Standard ImageNet mean/std for 1 channel (using Red channel as proxy)
IMAGENET_MEAN = (0.485,)
IMAGENET_STD = (0.229,)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    transform_list = []

    if phase == "train":
        # Augmentations as per Idea: ElasticTransform, ShiftScaleRotate
        # Probability p=0.2 for intensity/geometric modifications
        transform_list.extend(
            [
                A.ElasticTransform(alpha=120, sigma=6, alpha_affine=None, p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )

    # Normalization and Tensor conversion for all phases
    transform_list.extend(
        [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), A.ToTensorV2()]
    )

    return A.Compose(transform_list)


def load_and_cache_data(df, cache_name, load_cached_data=True):
    """
    Loads images and masks from disk or cache.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_name (str): Unique identifier for the cache file (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, masks_array)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{cache_name}_images.npy")
    mask_cache_path = os.path.join(CACHE_DIR, f"{cache_name}_masks.npy")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(img_cache_path):
        print(f"Loading {cache_name} data from cache...")
        images = np.load(img_cache_path)
        if os.path.exists(mask_cache_path):
            masks = np.load(mask_cache_path)
        else:
            masks = None
        return images, masks

    print(f"Processing {cache_name} data from scratch...")

    images = []
    masks = []
    has_masks = "mask_path" in df.columns and not df["mask_path"].isnull().all()

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

        # Load Mask if available
        if has_masks:
            mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback for missing masks if any (though metadata check passed)
                mask = np.zeros_like(img)
            masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    np.save(img_cache_path, images)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
        # Binarize masks (0 or 255 -> 0 or 1)
        masks = (masks > 127).astype(np.uint8)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    return images, masks


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W).
            masks (np.ndarray): Array of masks (N, H, W) or None.
            depths (np.ndarray): Array of standardized depths (N,).
            ids (list): List of image IDs.
            transform (A.Compose): Albumentations transforms.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]
        image_id = self.ids[idx]

        # Apply transforms
        # Albumentations expects HWC or HW.
        # Since we loaded grayscale (H, W), we pass it directly.
        if self.masks is not None:
            mask = self.masks[idx]
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].float().unsqueeze(0)  # (1, H, W)

            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "depth": torch.tensor([depth], dtype=torch.float32),
                "id": image_id,
            }
        else:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]

            return {
                "image": image_tensor,
                "depth": torch.tensor([depth], dtype=torch.float32),
                "id": image_id,
            }


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Use cached .npy files if available.
        debug (bool): If True, subsamples data for quick debugging.

    Returns:
        dict: {'train': loader, 'val': loader, 'test': loader}
    """
    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Calculate Depth Statistics from Training Set
    # We apply these stats to Val and Test to avoid leakage
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std()

    # Helper to standardize depth
    def standardize_depth(df):
        return (df["z"].values - depth_mean) / (depth_std + 1e-8)

    # Load and Cache Data Arrays
    train_imgs, train_masks = load_and_cache_data(train_df, "train", load_cached_data)
    val_imgs, val_masks = load_and_cache_data(val_df, "val", load_cached_data)
    test_imgs, _ = load_and_cache_data(test_df, "test", load_cached_data)

    # Create Datasets
    train_dataset = SaltDataset(
        images=train_imgs,
        masks=train_masks,
        depths=standardize_depth(train_df),
        ids=train_df["id"].values,
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        images=val_imgs,
        masks=val_masks,
        depths=standardize_depth(val_df),
        ids=val_df["id"].values,
        transform=get_transforms("val"),
    )

    test_dataset = SaltDataset(
        images=test_imgs,
        masks=None,
        depths=standardize_depth(test_df),
        ids=test_df["id"].values,
        transform=get_transforms("test"),
    )

    # Create Loaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders
