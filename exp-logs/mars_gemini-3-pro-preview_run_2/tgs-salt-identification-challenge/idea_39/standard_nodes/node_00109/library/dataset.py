import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_or_process, rle_decode


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Standard ImageNet mean/std for 1-channel (using R channel as proxy or just standard values)
    # Since we are using a ResNet backbone that expects normalized inputs.
    mean = (0.485,)
    std = (0.229,)

    if phase == "train":
        return A.Compose(
            [
                # Pad 101 -> 128 with reflection
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                # Rigid Augmentations
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.2,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                # Non-Rigid Augmentations (Elastic)
                A.ElasticTransform(
                    alpha=120,
                    sigma=6,
                    alpha_affine=3.6,  # approx 120 * 0.03
                    p=0.2,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(transpose_mask=True),
            ]
        )


def _load_images_from_df(df, input_root, load_masks=True):
    """
    Helper function to load images and masks from a DataFrame into numpy arrays.
    """
    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        img_path = os.path.join(input_root, row["image_path"])

        # Load Image (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Expand dims to (H, W, 1) for consistent Albumentations handling
        img = np.expand_dims(img, axis=2)
        images.append(img)
        ids.append(row["id"])

        # Load Depth if available
        if "z" in row and not pd.isna(row["z"]):
            depths.append(row["z"])
        else:
            depths.append(np.nan)

        # Load Mask if requested and available
        if load_masks and "rle_mask" in row:
            rle = row["rle_mask"]
            mask = rle_decode(rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE))
            # Expand dims to (H, W, 1)
            mask = np.expand_dims(mask, axis=2)
            masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)

    if load_masks:
        masks = np.array(masks, dtype=np.uint8)
        return images, masks, depths, ids
    else:
        return images, depths, ids


def process_dataset(df, mode, load_cached_data=True):
    """
    Loads dataset images/masks, using caching to speed up subsequent runs.
    """
    cache_prefix = f"data_{mode}"

    def _process():
        load_masks = mode in ["train", "val"]
        return _load_images_from_df(df, Config.INPUT_ROOT, load_masks=load_masks)

    data = load_or_process(
        file_name=f"{cache_prefix}.npy",
        process_fn=_process,
        load_cached_data=load_cached_data,
    )

    # Validate cache consistency (Cite debug_lesson_2)
    if len(data[0]) != len(df):
        print(
            f"Cache mismatch for {mode}: Expected {len(df)} samples, got {len(data[0])}. Recomputing..."
        )
        data = load_or_process(
            file_name=f"{cache_prefix}.npy",
            process_fn=_process,
            load_cached_data=False,
        )

    return data


class SaltDataset(Dataset):
    def __init__(
        self, images, ids, masks=None, depths=None, transform=None, depth_stats=None
    ):
        """
        Args:
            images: np.ndarray (N, H, W, 1)
            ids: list of strings
            masks: np.ndarray (N, H, W, 1) or None
            depths: np.ndarray (N,) or None
            transform: Albumentations Compose
            depth_stats: tuple (mean, std) for depth normalization
        """
        self.images = images
        self.ids = ids
        self.masks = masks
        self.depths = depths
        self.transform = transform
        self.depth_stats = depth_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 1)

        # Prepare data for augmentation
        data = {"image": img}
        if self.masks is not None:
            data["mask"] = self.masks[idx]  # (H, W, 1)

        # Apply transforms
        if self.transform:
            augmented = self.transform(**data)
            img_tensor = augmented["image"]  # (1, H, W)

            if self.masks is not None:
                mask_tensor = augmented["mask"]  # (1, H, W)
                # Convert mask to float for loss calculation
                mask_tensor = mask_tensor.float()
        else:
            # Fallback if no transform (should not happen in this pipeline)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            if self.masks is not None:
                mask_tensor = torch.from_numpy(
                    self.masks[idx].transpose(2, 0, 1)
                ).float()

        # Handle Depth
        if self.depths is not None:
            z = self.depths[idx]
            # Normalize depth
            if self.depth_stats:
                mean, std = self.depth_stats
                z = (z - mean) / std

            z_tensor = torch.tensor([z], dtype=torch.float32)

            if self.masks is not None:
                return img_tensor, mask_tensor, z_tensor, self.ids[idx]
            else:
                # For test set, we might have depths or NaNs.
                # If we are in inference mode, we usually don't need z from dataset
                # because we scan over it. But if we did, we'd return it here.
                # The pipeline expects (img, id) for test loader in generate_submission.
                return img_tensor, self.ids[idx]
        else:
            return img_tensor, self.ids[idx]


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # 2. Calculate Depth Statistics (from Training set only)
    # We use the raw 'z' values from the dataframe before processing
    z_train = train_df["z"].values
    z_mean = float(np.mean(z_train))
    z_std = float(np.std(z_train))
    depth_stats = (z_mean, z_std)

    print(f"Depth Statistics: Mean={z_mean:.4f}, Std={z_std:.4f}")

    # 3. Process/Load Data
    # Train
    train_imgs, train_masks, train_depths, train_ids = process_dataset(
        train_df, "train"
    )

    # Val
    val_imgs, val_masks, val_depths, val_ids = process_dataset(val_df, "val")

    # Test (No masks)
    test_imgs, test_depths, test_ids = process_dataset(test_df, "test")

    # 4. Create Datasets
    train_dataset = SaltDataset(
        train_imgs,
        train_ids,
        masks=train_masks,
        depths=train_depths,
        transform=get_transforms("train"),
        depth_stats=depth_stats,
    )

    val_dataset = SaltDataset(
        val_imgs,
        val_ids,
        masks=val_masks,
        depths=val_depths,
        transform=get_transforms("val"),
        depth_stats=depth_stats,
    )

    test_dataset = SaltDataset(
        test_imgs,
        test_ids,
        masks=None,
        depths=test_depths,
        transform=get_transforms("test"),
        depth_stats=depth_stats,
    )

    # 5. Create Loaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
