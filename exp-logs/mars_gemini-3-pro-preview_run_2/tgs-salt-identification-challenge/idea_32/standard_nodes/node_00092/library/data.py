import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config, get_depth_stats

# =========================================================================
# Transforms
# =========================================================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train/val/test modes.
    Note: Images are already padded to 128x128 in preprocessing.
    """
    if mode == "train":
        return A.Compose(
            [
                # Non-Rigid Augmentation (Elastic)
                A.ElasticTransform(
                    alpha=Config.AUG_ELASTIC_ALPHA,
                    sigma=Config.AUG_ELASTIC_SIGMA,
                    alpha_affine=Config.AUG_ELASTIC_SIGMA * 0.05,
                    p=Config.AUG_PROB,
                ),
                # Rigid Augmentation (Geometric)
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.AUG_PROB,
                ),
                A.HorizontalFlip(p=0.5),
                # Normalization
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )


# =========================================================================
# Preprocessing & Caching
# =========================================================================


def pad_image(img, target_size=128):
    """
    Pads image from 101x101 to 128x128 using reflection padding.
    """
    h, w = img.shape[:2]
    diff_h = target_size - h
    diff_w = target_size - w

    pad_top = diff_h // 2
    pad_bottom = diff_h - pad_top
    pad_left = diff_w // 2
    pad_right = diff_w - pad_left

    return cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def process_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data based on metadata CSV, pads images, and caches them as .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for cached files (e.g., 'train', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: {'images': np.ndarray, 'masks': np.ndarray (or None), 'depths': np.ndarray, 'ids': list}
    """
    # Cache paths
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    p_images = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    p_masks = os.path.join(cache_dir, f"{cache_prefix}_masks.npy")
    p_depths = os.path.join(cache_dir, f"{cache_prefix}_depths.npy")
    p_ids = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(p_images)
            and os.path.exists(p_depths)
            and os.path.exists(p_ids)
        ):
            try:
                images = np.load(p_images)
                depths = np.load(p_depths)
                ids = np.load(p_ids)
                masks = np.load(p_masks) if os.path.exists(p_masks) else None
                return {"images": images, "masks": masks, "depths": depths, "ids": ids}
            except Exception:
                pass  # Fall through to recompute

    # 2. Compute from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    mask_list = []
    depth_list = []
    id_list = []

    has_masks = "mask_path" in df.columns

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Pad Image
        img = pad_image(img, Config.IMG_SIZE)
        img_list.append(img)

        # Load Mask
        if has_masks:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Pad Mask
            mask = pad_image(mask, Config.IMG_SIZE)
            # Binarize
            mask = (mask > 127).astype(np.uint8)
            mask_list.append(mask)

        # Depth
        depth_list.append(row["z"])

        # ID
        id_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)  # (N, 128, 128)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    if has_masks:
        masks = np.array(mask_list, dtype=np.uint8)  # (N, 128, 128)
    else:
        masks = None

    # 3. Save to cache
    np.save(p_images, images)
    np.save(p_depths, depths)
    np.save(p_ids, ids)
    if masks is not None:
        np.save(p_masks, masks)

    return {"images": images, "masks": masks, "depths": depths, "ids": ids}


# =========================================================================
# Dataset
# =========================================================================


class SaltDataset(Dataset):
    def __init__(self, data_dict, mode="train", pseudo_masks=None, transform=None):
        """
        Args:
            data_dict (dict): Dictionary containing 'images', 'masks', 'depths', 'ids'.
            mode (str): 'train', 'val', 'test'.
            pseudo_masks (dict or array): Optional soft masks for unlabeled data.
                                          If array, must match length of data.
                                          If dict, mapped by ID.
            transform (A.Compose): Albumentations transforms.
        """
        self.images = data_dict["images"]
        self.masks = data_dict.get("masks")
        self.depths = data_dict["depths"]
        self.ids = data_dict["ids"]
        self.mode = mode
        self.transform = transform

        # Handle pseudo masks
        self.pseudo_masks = None
        if pseudo_masks is not None:
            if isinstance(pseudo_masks, dict):
                # Convert dict to array aligned with ids
                self.pseudo_masks = np.array(
                    [pseudo_masks[i] for i in self.ids], dtype=np.float32
                )
            else:
                self.pseudo_masks = pseudo_masks

        # Depth Normalization Stats
        stats = get_depth_stats()
        self.depth_mean = stats["mean"]
        self.depth_std = stats["std"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image: (H, W) -> (H, W, 1) for Albumentations
        img = self.images[idx]
        img = np.expand_dims(img, axis=-1)  # (128, 128, 1)

        # Depth: Normalize
        d = self.depths[idx]
        d_norm = (d - self.depth_mean) / self.depth_std
        d_tensor = torch.tensor([d_norm], dtype=torch.float32)

        # Targets
        mask = None

        # Case 1: Use Pseudo Labels (Soft targets)
        if self.pseudo_masks is not None:
            mask = self.pseudo_masks[idx]  # (128, 128) float

        # Case 2: Use Ground Truth (Hard targets)
        elif self.masks is not None:
            mask = self.masks[idx]  # (128, 128) uint8

        # Apply Transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]
        else:
            # Fallback
            t = ToTensorV2()
            if mask is not None:
                res = t(image=img, mask=mask)
                img = res["image"]
                mask = res["mask"]
            else:
                res = t(image=img)
                img = res["image"]

        # Prepare return dict
        sample = {
            "image": img,  # (1, 128, 128)
            "depth": d_tensor,  # (1,)
            "id": self.ids[idx],
        }

        if mask is not None:
            # If mask is uint8 (GT), convert to float
            if mask.dtype == torch.uint8:
                mask = mask.float()

            # Ensure channel dim (1, 128, 128)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            sample["mask"] = mask

            # For Stage 3 Student Loss:
            # Only include explicit 'depth_target' if we have real GT (not pseudo)
            if self.pseudo_masks is None:
                sample["depth_target"] = d_tensor

        return sample


# =========================================================================
# Data Loaders
# =========================================================================


def get_stage1_loaders(fold=0, load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for Stage 1 (Specialist Teacher Training).
    Combines Train and Val metadata, then performs K-Fold split.
    """
    # Load all labeled data
    data_train = process_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    data_val = process_and_cache_data(Config.VAL_METADATA_PATH, "val", load_cached_data)

    # Merge arrays
    images = np.concatenate([data_train["images"], data_val["images"]], axis=0)
    masks = np.concatenate([data_train["masks"], data_val["masks"]], axis=0)
    depths = np.concatenate([data_train["depths"], data_val["depths"]], axis=0)
    ids = np.concatenate([data_train["ids"], data_val["ids"]], axis=0)

    # Debug subsampling
    if debug or (Config.MAX_TRAIN_SAMPLES is not None):
        limit = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 100
        images = images[:limit]
        masks = masks[:limit]
        depths = depths[:limit]
        ids = ids[:limit]

    # K-Fold Split
    kf = KFold(n_splits=Config.STAGE1_FOLDS, shuffle=True, random_state=Config.SEED)
    splits = list(kf.split(images))
    train_idx, val_idx = splits[fold]

    # Create Dicts
    train_data = {
        "images": images[train_idx],
        "masks": masks[train_idx],
        "depths": depths[train_idx],
        "ids": ids[train_idx],
    }
    val_data = {
        "images": images[val_idx],
        "masks": masks[val_idx],
        "depths": depths[val_idx],
        "ids": ids[val_idx],
    }

    # Datasets
    train_ds = SaltDataset(train_data, mode="train", transform=get_transforms("train"))
    val_ds = SaltDataset(val_data, mode="val", transform=get_transforms("val"))

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
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Prepares DataLoader for Inference / Pseudo-labeling (Test set).
    """
    data_test = process_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    test_ds = SaltDataset(data_test, mode="test", transform=get_transforms("test"))

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader


def get_stage3_loaders(pseudo_labels_dict=None, load_cached_data=True, debug=False):
    """
    Prepares DataLoaders for Stage 3 (Student Training).
    Returns:
        labeled_loader: Train + Val (Hard GT)
        unlabeled_loader: Test (Soft Pseudo Labels)
    """
    # 1. Labeled Data (Train + Val)
    data_train = process_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    data_val = process_and_cache_data(Config.VAL_METADATA_PATH, "val", load_cached_data)

    images_lab = np.concatenate([data_train["images"], data_val["images"]], axis=0)
    masks_lab = np.concatenate([data_train["masks"], data_val["masks"]], axis=0)
    depths_lab = np.concatenate([data_train["depths"], data_val["depths"]], axis=0)
    ids_lab = np.concatenate([data_train["ids"], data_val["ids"]], axis=0)

    if debug or (Config.MAX_TRAIN_SAMPLES is not None):
        limit = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 100
        images_lab = images_lab[:limit]
        masks_lab = masks_lab[:limit]
        depths_lab = depths_lab[:limit]
        ids_lab = ids_lab[:limit]

    labeled_data = {
        "images": images_lab,
        "masks": masks_lab,
        "depths": depths_lab,
        "ids": ids_lab,
    }

    labeled_ds = SaltDataset(
        labeled_data, mode="train", transform=get_transforms("train")
    )

    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Unlabeled Data (Test with Pseudo Labels)
    if pseudo_labels_dict is None:
        return labeled_loader, None

    data_test = process_and_cache_data(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    if debug or (Config.MAX_TRAIN_SAMPLES is not None):
        limit = Config.MAX_TRAIN_SAMPLES if Config.MAX_TRAIN_SAMPLES else 100
        data_test["images"] = data_test["images"][:limit]
        data_test["depths"] = data_test["depths"][:limit]
        data_test["ids"] = data_test["ids"][:limit]

    unlabeled_ds = SaltDataset(
        data_test,
        mode="train",  # Use train transforms for student learning
        pseudo_masks=pseudo_labels_dict,
        transform=get_transforms("train"),
    )

    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    return labeled_loader, unlabeled_loader


def get_supervised_loaders():
    """
    Prepares DataLoaders for Supervised Training.
    Train: metadata/train.csv (2400)
    Val: metadata/val.csv (600)
    """
    # Train Data
    data_train = process_and_cache_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data=True
    )

    # Val Data
    data_val = process_and_cache_data(
        Config.VAL_METADATA_PATH, "val", load_cached_data=True
    )

    # Datasets
    train_ds = SaltDataset(data_train, mode="train", transform=get_transforms("train"))
    val_ds = SaltDataset(data_val, mode="val", transform=get_transforms("val"))

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
    )

    return train_loader, val_loader
