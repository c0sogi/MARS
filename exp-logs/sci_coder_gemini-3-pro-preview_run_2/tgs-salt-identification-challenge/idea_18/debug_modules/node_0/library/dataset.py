import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import pad_image, seed_everything

# Constants
CACHE_DIR = "./working/idea_18/"
INPUT_ROOT = "./input"
IMG_ORIG_SIZE = 101
IMG_TARGET_SIZE = 128
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03,
                            p=0.5,
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                    ],
                    p=0.5,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=[IMAGENET_MEAN], std=[IMAGENET_STD], max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=[IMAGENET_MEAN], std=[IMAGENET_STD], max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )


def load_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Loads images, masks, and depths from disk or cache.

    Args:
        df: DataFrame containing metadata.
        cache_prefix: String prefix for cache filenames (e.g., 'train', 'val', 'test').
        load_cached_data: Boolean to enable/disable loading from cache.

    Returns:
        images: (N, 101, 101) uint8
        masks: (N, 101, 101) uint8 (or None for test)
        depths: (N,) float32
        ids: List of IDs
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_masks.npy")
    depth_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_depths.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    has_masks = "mask_path" in df.columns

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(depth_cache_path)
            and os.path.exists(id_cache_path)
        ):
            if not has_masks or os.path.exists(mask_cache_path):
                # print(f"Loading {cache_prefix} data from cache...")
                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                ids = np.load(id_cache_path, allow_pickle=True)
                masks = np.load(mask_cache_path) if has_masks else None
                return images, masks, depths, ids

    # Process from scratch
    # print(f"Processing {cache_prefix} data from scratch...")
    images = []
    masks = []
    depths = df["z"].values.astype(np.float32)
    ids = df["id"].values

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_ROOT, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images.append(img)

        # Load Mask if exists
        if has_masks:
            mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            # Ensure binary 0/1
            mask = (mask > 127).astype(np.uint8)
            masks.append(mask)

    images = np.array(images, dtype=np.uint8)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    return images, masks, depths, ids


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, transform=None, mode="train"):
        """
        Args:
            images: (N, H, W) numpy array
            masks: (N, H, W) numpy array or None
            depths: (N,) numpy array
            ids: List of IDs
            transform: Albumentations transform
            mode: 'train', 'val', or 'test'
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get data
        image = self.images[idx]  # (101, 101)
        depth = self.depths[idx]
        img_id = self.ids[idx]

        # 2. Pad to 128x128 (Reflection)
        # pad_image expects (H, W) or (H, W, C).
        # Our images are (101, 101). Result is (128, 128).
        image = pad_image(image, target_size=IMG_TARGET_SIZE)

        mask = None
        if self.masks is not None:
            mask_raw = self.masks[idx]
            mask = pad_image(mask_raw, target_size=IMG_TARGET_SIZE)

        # 3. Apply Transforms (Augmentation + Normalization + ToTensor)
        # Albumentations expects HWC. We expand dims to HxWx1 for consistency.
        image = np.expand_dims(image, axis=-1)  # (128, 128, 1)

        if mask is not None:
            # Mask should be passed to albumentations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

                # Ensure mask is float tensor (B, H, W) or (B, 1, H, W)
                # ToTensorV2 produces (H, W) for mask if input is (H, W)
                # We want (1, H, W) for BCE/Lovasz usually, or (H, W).
                # Let's standardize to (1, H, W) float
                mask = mask.float().unsqueeze(0)
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Image is now a tensor (C, H, W) due to ToTensorV2

        # 4. Return
        # Depth needs to be a tensor
        depth_tensor = torch.tensor([depth], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            return image, mask, depth_tensor, img_id
        else:
            return image, depth_tensor, img_id


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        batch_size: Batch size.
        num_workers: Number of worker threads.
        load_cached_data: Whether to use cached .npy files.
        debug: If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:50]
        test_df = test_df.iloc[:50]

    # 2. Load Raw Data (with Caching)
    # We use 'train' cache for the training split, 'val' for validation split
    train_imgs, train_masks, train_depths, train_ids = load_and_cache_data(
        train_df, "train", load_cached_data
    )
    val_imgs, val_masks, val_depths, val_ids = load_and_cache_data(
        val_df, "val", load_cached_data
    )
    test_imgs, _, test_depths, test_ids = load_and_cache_data(
        test_df, "test", load_cached_data
    )

    # 3. Depth Normalization (Fit on Train, Transform All)
    # Calculate stats from training depths
    d_mean = train_depths.mean()
    d_std = train_depths.std() + 1e-8  # Avoid div by zero

    train_depths = (train_depths - d_mean) / d_std
    val_depths = (val_depths - d_mean) / d_std
    test_depths = (test_depths - d_mean) / d_std

    # 4. Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transform=get_transforms("val"),
        mode="val",
    )

    test_dataset = SaltDataset(
        test_imgs,
        None,
        test_depths,
        test_ids,
        transform=get_transforms("test"),
        mode="test",
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
