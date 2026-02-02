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


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    # ImageNet stats for 1 channel (using Red channel stats as proxy)
    mean = (0.485,)
    std = (0.229,)

    if mode == "train" or mode == "pseudo":
        return A.Compose(
            [
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_ALPHA_AFFINE,
                    p=Config.AUG_ELASTIC_PROB,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_RIGID_PROB,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    def __init__(
        self, images, masks=None, depths=None, ids=None, mode="train", transform=None
    ):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        data = {"image": image}

        # Handle masks
        if self.masks is not None:
            mask = self.masks[idx]
            data["mask"] = mask

        if self.transform:
            augmented = self.transform(**data)
            image = augmented["image"]
            if "mask" in data:
                mask = augmented["mask"]

        # Return based on mode
        if self.mode == "train":
            # Mask: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # Depth: Scalar -> Tensor
            d_val = self.depths[idx] if self.depths is not None else 0.0
            depth = torch.tensor([d_val], dtype=torch.float32)

            return image, mask.float(), depth

        elif self.mode == "pseudo":
            # Mask is soft probability map
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            return image, mask.float()

        elif self.mode == "test":
            return image, self.ids[idx]

        return image


def process_and_cache_data(csv_path, cache_prefix, load_cached_data=True):
    """
    Loads data from CSV, reads images/masks, pads them, and caches to .npy.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{cache_prefix}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{cache_prefix}_depths.npy")
    id_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(id_cache_path)
    ):
        images = np.load(img_cache_path)
        ids = np.load(id_cache_path)
        masks = np.load(mask_cache_path) if os.path.exists(mask_cache_path) else None
        depths = np.load(depth_cache_path) if os.path.exists(depth_cache_path) else None
        return images, masks, depths, ids

    # Process from scratch
    df = pd.read_csv(csv_path)
    if Config.MAX_SAMPLES is not None:
        df = df.head(Config.MAX_SAMPLES)

    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for robustness (should not happen given metadata check)
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

        img_padded = pad_image(img, Config.IMG_SIZE)
        images.append(img_padded)
        ids.append(row["id"])

        # Load Mask
        if "rle_mask" in row:
            rle = row["rle_mask"]
            if pd.isna(rle) or rle == "":
                mask = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
            else:
                mask = rle_decode(rle, (Config.ORIG_SIZE, Config.ORIG_SIZE))
            mask_padded = pad_image(mask, Config.IMG_SIZE)
            masks.append(mask_padded)

        # Load Depth
        if "z" in row:
            depths.append(row["z"])
        else:
            depths.append(np.nan)

    images = np.array(images, dtype=np.uint8)
    ids = np.array(ids)
    np.save(img_cache_path, images)
    np.save(id_cache_path, ids)

    if len(masks) > 0:
        masks = np.array(masks, dtype=np.uint8)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    if len(depths) > 0:
        depths = np.array(depths, dtype=np.float32)
        np.save(depth_cache_path, depths)
    else:
        depths = None

    return images, masks, depths, ids


def get_depth_stats(depths):
    """Calculates mean and std of depths (ignoring NaNs)."""
    valid = depths[~np.isnan(depths)]
    if len(valid) == 0:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(np.mean(valid)), "std": float(np.std(valid))}


def get_train_val_loaders():
    """
    Returns DataLoaders for training and validation sets.
    Also returns depth statistics calculated from the training set.
    """
    # Train Data
    t_imgs, t_masks, t_depths, t_ids = process_and_cache_data(Config.TRAIN_CSV, "train")
    stats = get_depth_stats(t_depths)

    # Normalize depths
    t_depths_norm = (t_depths - stats["mean"]) / (stats["std"] + 1e-6)
    t_depths_norm = np.nan_to_num(t_depths_norm, nan=0.0)

    train_ds = SaltDataset(
        t_imgs,
        t_masks,
        t_depths_norm,
        t_ids,
        mode="train",
        transform=get_transforms("train"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val Data
    v_imgs, v_masks, v_depths, v_ids = process_and_cache_data(Config.VAL_CSV, "val")
    v_depths_norm = (v_depths - stats["mean"]) / (stats["std"] + 1e-6)
    v_depths_norm = np.nan_to_num(v_depths_norm, nan=0.0)

    # Use mode="train" for val dataset to ensure it returns (img, mask, depth) tuple
    # But use "val" transform to disable augmentation
    val_ds = SaltDataset(
        v_imgs,
        v_masks,
        v_depths_norm,
        v_ids,
        mode="train",
        transform=get_transforms("val"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, stats


def get_test_loader():
    """
    Returns DataLoader for the test set.
    """
    imgs, _, _, ids = process_and_cache_data(Config.TEST_CSV, "test")
    ds = SaltDataset(imgs, ids=ids, mode="test", transform=get_transforms("test"))
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )


def get_pseudo_loader(soft_masks_dict):
    """
    Returns DataLoader for the test set with soft pseudo-labels.
    Args:
        soft_masks_dict (dict): Dictionary mapping image ID to soft mask (np.ndarray).
    """
    # Load test images
    imgs, _, _, ids = process_and_cache_data(Config.TEST_CSV, "test")

    # Align soft masks with images
    masks_list = []
    for i in ids:
        if i in soft_masks_dict:
            masks_list.append(soft_masks_dict[i])
        else:
            # Fallback empty mask if ID missing (should not happen in valid pipeline)
            masks_list.append(
                np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            )

    masks = np.array(masks_list, dtype=np.float32)

    ds = SaltDataset(
        imgs, masks=masks, mode="pseudo", transform=get_transforms("pseudo")
    )
    return DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
