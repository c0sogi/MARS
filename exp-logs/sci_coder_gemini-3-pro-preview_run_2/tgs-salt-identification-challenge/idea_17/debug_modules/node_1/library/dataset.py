import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import pad_image, set_seed

# Constants
CACHE_DIR = "./working/idea_17/"
METADATA_DIR = "./metadata"
INPUT_DIR = "./input"
IMG_SIZE = 128
ORIG_SIZE = 101


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Approximate 1-channel mean/std derived from ImageNet RGB stats
    # Mean: (0.485+0.456+0.406)/3 = 0.449 -> 0.45
    # Std: (0.229+0.224+0.225)/3 = 0.226 -> 0.225
    mean = (0.45,)
    std = (0.225,)

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # Non-rigid transformations (Elastic, Grid, Optical)
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=6, alpha_affine=None, p=1.0
                        ),
                        A.GridDistortion(p=1.0),
                        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                    ],
                    p=0.2,
                ),
                # Rigid transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        return A.Compose(
            [A.Normalize(mean=mean, std=std), ToTensorV2(transpose_mask=True)]
        )


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        depths,
        masks=None,
        is_labeled=None,
        phase="train",
        transforms=None,
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, 128, 128).
            depths (np.ndarray): Array of normalized depths (N,).
            masks (np.ndarray, optional): Array of masks (N, 128, 128). Can be binary or soft.
            is_labeled (np.ndarray, optional): Array of flags (N,). 1.0 for labeled, 0.0 for unlabeled.
            phase (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Augmentation pipeline.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.is_labeled = is_labeled
        self.phase = phase
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]

        # Expand dims for albumentations (H, W) -> (H, W, 1)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)

        # Apply Augmentations
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=image)
                image = augmented["image"]

        # Bernoulli Depth Masking (Train only)
        # With p=0.5, set depth to 0.0 (which is the mean of normalized depths)
        if self.phase == "train":
            if np.random.rand() < 0.5:
                depth = 0.0

        # Convert depth to tensor
        depth = torch.tensor([depth], dtype=torch.float32)

        # Determine labeled flag
        if self.is_labeled is not None:
            flag = self.is_labeled[idx]
        else:
            flag = 1.0  # Default to supervised

        flag = torch.tensor(flag, dtype=torch.float32)

        # Return tuple
        if mask is not None:
            # Ensure mask is float (handles both binary and soft targets)
            mask = mask.float()
            return image, depth, mask, flag
        else:
            # Inference mode (Test)
            return image, depth


def prepare_data(load_cached_data=True):
    """
    Loads data from disk, processes it (pad, normalize), and caches it to .npy files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    filenames = [
        "train_images.npy",
        "train_masks.npy",
        "train_depths.npy",
        "train_ids.npy",
        "val_images.npy",
        "val_masks.npy",
        "val_depths.npy",
        "val_ids.npy",
        "test_images.npy",
        "test_depths.npy",
        "test_ids.npy",
        "depth_stats.npy",
    ]

    all_exist = all([os.path.exists(os.path.join(CACHE_DIR, f)) for f in filenames])

    if load_cached_data and all_exist:
        print("Loading cached dataset from", CACHE_DIR)
        data = {}
        for f in filenames:
            key = f.replace(".npy", "")
            data[key] = np.load(os.path.join(CACHE_DIR, f), allow_pickle=True)
        return data

    print("Processing dataset from scratch...")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    def process_subset(df, has_mask=True):
        imgs, masks, depths, ids = [], [], [], []
        for _, row in df.iterrows():
            # Load Image
            img_path = os.path.join(INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = pad_image(img, (IMG_SIZE, IMG_SIZE))
            imgs.append(img)

            # Load Mask
            if has_mask:
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                mask = pad_image(mask, (IMG_SIZE, IMG_SIZE))
                mask = (mask > 127).astype(np.float32)  # Binary 0.0 or 1.0
                masks.append(mask)

            depths.append(row["z"])
            ids.append(row["id"])

        return (
            np.array(imgs),
            np.array(masks) if has_mask else None,
            np.array(depths, dtype=np.float32),
            np.array(ids),
        )

    # Process each split
    train_imgs, train_masks, train_depths, train_ids = process_subset(train_df, True)
    val_imgs, val_masks, val_depths, val_ids = process_subset(val_df, True)
    test_imgs, _, test_depths, test_ids = process_subset(test_df, False)

    # Normalize Depths
    d_mean = train_depths.mean()
    d_std = train_depths.std()

    train_depths = (train_depths - d_mean) / d_std
    val_depths = (val_depths - d_mean) / d_std
    test_depths = (test_depths - d_mean) / d_std

    # Save to Cache
    np.save(os.path.join(CACHE_DIR, "train_images.npy"), train_imgs)
    np.save(os.path.join(CACHE_DIR, "train_masks.npy"), train_masks)
    np.save(os.path.join(CACHE_DIR, "train_depths.npy"), train_depths)
    np.save(os.path.join(CACHE_DIR, "train_ids.npy"), train_ids)

    np.save(os.path.join(CACHE_DIR, "val_images.npy"), val_imgs)
    np.save(os.path.join(CACHE_DIR, "val_masks.npy"), val_masks)
    np.save(os.path.join(CACHE_DIR, "val_depths.npy"), val_depths)
    np.save(os.path.join(CACHE_DIR, "val_ids.npy"), val_ids)

    np.save(os.path.join(CACHE_DIR, "test_images.npy"), test_imgs)
    np.save(os.path.join(CACHE_DIR, "test_depths.npy"), test_depths)
    np.save(os.path.join(CACHE_DIR, "test_ids.npy"), test_ids)

    np.save(os.path.join(CACHE_DIR, "depth_stats.npy"), np.array([d_mean, d_std]))

    return {
        "train_images": train_imgs,
        "train_masks": train_masks,
        "train_depths": train_depths,
        "train_ids": train_ids,
        "val_images": val_imgs,
        "val_masks": val_masks,
        "val_depths": val_depths,
        "val_ids": val_ids,
        "test_images": test_imgs,
        "test_depths": test_depths,
        "test_ids": test_ids,
        "depth_mean": d_mean,
        "depth_std": d_std,
    }


def get_loaders(batch_size=32, load_cached_data=True, soft_test_masks=None):
    """
    Constructs DataLoaders for train, val, and test.

    Args:
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached .npy files.
        soft_test_masks (np.ndarray, optional): Soft targets for test images (N_test, 128, 128).
                                                If provided, creates a Semi-Supervised training set.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    data = prepare_data(load_cached_data)

    # --- Training Set Construction ---
    if soft_test_masks is not None:
        # Semi-Supervised Mode (Distillation)
        print("Initializing Semi-Supervised Training Loader...")

        # Combine Labeled Train + Unlabeled Test (with Soft Masks)
        combined_images = np.concatenate(
            [data["train_images"], data["test_images"]], axis=0
        )

        # Combine Hard Masks and Soft Masks
        # Ensure soft_test_masks matches shape
        if soft_test_masks.shape != (len(data["test_images"]), IMG_SIZE, IMG_SIZE):
            raise ValueError(
                f"Soft mask shape mismatch. Expected {(len(data['test_images']), IMG_SIZE, IMG_SIZE)}, got {soft_test_masks.shape}"
            )

        combined_masks = np.concatenate([data["train_masks"], soft_test_masks], axis=0)

        # Combine Depths
        # For test depths in distillation, we can use the normalized depths or force 0.
        # The dataset class handles forcing 0 with p=0.5, but for distillation we might want
        # to force 0 for test images always? The strategy says "Input a constant depth of 0 for these predictions".
        # Let's use the normalized test depths; the Bernoulli mask in Dataset will handle robustness.
        combined_depths = np.concatenate(
            [data["train_depths"], data["test_depths"]], axis=0
        )

        # Create Flags: 1 for Labeled, 0 for Unlabeled
        flags_train = np.ones(len(data["train_images"]), dtype=np.float32)
        flags_test = np.zeros(len(data["test_images"]), dtype=np.float32)
        combined_flags = np.concatenate([flags_train, flags_test], axis=0)

        train_dataset = SaltDataset(
            combined_images,
            combined_depths,
            combined_masks,
            is_labeled=combined_flags,
            phase="train",
            transforms=get_transforms("train"),
        )
    else:
        # Supervised Mode
        print("Initializing Supervised Training Loader...")
        train_dataset = SaltDataset(
            data["train_images"],
            data["train_depths"],
            data["train_masks"],
            is_labeled=np.ones(len(data["train_images"])),
            phase="train",
            transforms=get_transforms("train"),
        )

    # --- Validation Set ---
    val_dataset = SaltDataset(
        data["val_images"],
        data["val_depths"],
        data["val_masks"],
        phase="val",
        transforms=get_transforms("val"),
    )

    # --- Test Set ---
    # For inference, we usually force depth to 0 as per strategy.
    # We can pass an array of zeros (normalized mean) or rely on the model input logic.
    # Here we pass the computed normalized depths, but the inference loop can override if needed.
    # Strategy: "Inference... Input a constant depth value of 0".
    # Since 0 is the mean of normalized data, passing 0.0 is correct.
    test_dataset = SaltDataset(
        data["test_images"],
        np.zeros_like(data["test_depths"]),
        None,
        phase="test",
        transforms=get_transforms("test"),
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }

    return loaders
