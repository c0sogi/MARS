import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Constants
CACHE_DIR = "./working/idea_36/"
INPUT_ROOT = "./input"
IMG_SIZE_ORIG = 101
IMG_SIZE_TARGET = 128
# ImageNet stats averaged for grayscale: (0.485+0.456+0.406)/3, (0.229+0.224+0.225)/3
GRAY_MEAN = (0.449,)
GRAY_STD = (0.226,)


def pad_image(image, target_size=(128, 128)):
    """
    Pads an image to the target size using reflection padding.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h == 0 and pad_w == 0:
        return image

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)


def unpad_image(image, original_size=(101, 101)):
    """
    Crops the center of the image back to the original size.
    Expects image in (H, W) or (H, W, C) format.
    """
    h, w = image.shape[:2]
    orig_h, orig_w = original_size

    start_h = (h - orig_h) // 2
    start_w = (w - orig_w) // 2

    return image[start_h : start_h + orig_h, start_w : start_w + orig_w]


def get_depth_stats(train_df):
    """
    Calculates mean and std of depth from the training dataframe.
    """
    depths = train_df["z"].values
    return {"mean": np.mean(depths), "std": np.std(depths)}


def get_transforms(phase="train"):
    """
    Returns Albumentations transform pipeline.
    """
    transforms_list = []

    # 1. Padding (Deterministic for all phases)
    transforms_list.append(
        A.PadIfNeeded(
            min_height=IMG_SIZE_TARGET,
            min_width=IMG_SIZE_TARGET,
            border_mode=cv2.BORDER_REFLECT_101,
            always_apply=True,
        )
    )

    if phase == "train":
        # 2. Augmentations (Train only)
        # Elastic Transform: alpha=120, sigma=6, alpha_affine=3.6
        transforms_list.append(
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                alpha_affine=3.6,
                p=0.2,
                border_mode=cv2.BORDER_REFLECT_101,
            )
        )
        # Rigid: ShiftScaleRotate
        transforms_list.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
            )
        )
        # Horizontal Flip
        transforms_list.append(A.HorizontalFlip(p=0.5))

    # 3. Normalization (All phases)
    transforms_list.append(A.Normalize(mean=GRAY_MEAN, std=GRAY_STD))

    # 4. ToTensor (All phases)
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def preload_data(metadata_path, phase, load_cached_data=True):
    """
    Loads data from metadata CSV. Caches raw numpy arrays to disk.
    Returns:
        df: pandas DataFrame
        data_dict: Dictionary containing 'images', 'masks' (optional), 'depths', 'ids'
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    cache_prefix = os.path.join(CACHE_DIR, f"{phase}_")
    path_images = cache_prefix + "images.npy"
    path_masks = cache_prefix + "masks.npy"
    path_depths = cache_prefix + "depths.npy"
    path_ids = cache_prefix + "ids.npy"

    df = pd.read_csv(metadata_path)

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_images)
        and os.path.exists(path_depths)
        and os.path.exists(path_ids)
    )
    if phase != "test":
        cache_exists = cache_exists and os.path.exists(path_masks)

    if load_cached_data and cache_exists:
        print(f"Loading cached {phase} data from {CACHE_DIR}...")
        images = np.load(path_images)
        depths = np.load(path_depths)
        ids = np.load(path_ids, allow_pickle=True)
        masks = np.load(path_masks) if phase != "test" else None
        return df, {"images": images, "masks": masks, "depths": depths, "ids": ids}

    print(f"Processing {phase} data from scratch...")
    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(INPUT_ROOT, row["image_path"])
        # Load as Grayscale (1 channel)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images_list.append(img)

        # Load Mask (if not test)
        if phase != "test":
            mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Binarize just in case
            mask = (mask > 127).astype(np.uint8)
            masks_list.append(mask)

        # Load Depth
        # Handle NaN for test set if necessary (though usually filled externally)
        z = row["z"] if not pd.isna(row["z"]) else 0.0
        depths_list.append(z)

        ids_list.append(str(row["id"]))

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    depths = np.array(depths_list, dtype=np.float64)
    ids = np.array(ids_list)

    # Save to cache
    np.save(path_images, images)
    np.save(path_depths, depths)
    np.save(path_ids, ids)

    if phase != "test":
        masks = np.array(masks_list, dtype=np.uint8)
        np.save(path_masks, masks)
    else:
        masks = None

    return df, {"images": images, "masks": masks, "depths": depths, "ids": ids}


class SaltDataset(Dataset):
    def __init__(self, data_dict, transform=None, depth_stats=None):
        """
        Args:
            data_dict: Dict with keys 'images', 'masks', 'depths', 'ids'.
            transform: Albumentations transform.
            depth_stats: Dict {'mean': float, 'std': float} for depth normalization.
        """
        self.images = data_dict["images"]
        self.masks = data_dict.get("masks")
        self.depths = data_dict["depths"]
        self.ids = data_dict["ids"]
        self.transform = transform
        self.depth_stats = depth_stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get Image (H, W)
        image = self.images[idx]

        # Expand to (H, W, 1) for Albumentations consistency
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        # 2. Get Mask (H, W) if available
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            # Expand to (H, W, 1)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

        # 3. Apply Transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # 4. Process Depth
        z = self.depths[idx]
        if self.depth_stats:
            # Standard Scaling: (z - mean) / std
            z = (z - self.depth_stats["mean"]) / (self.depth_stats["std"] + 1e-8)

        # Convert depth to tensor (1,)
        z_tensor = torch.tensor([z], dtype=torch.float32)

        # 5. Return
        # Image is already tensor (C, H, W) from ToTensorV2
        # Mask needs to be tensor (1, H, W) if it exists
        if mask is not None:
            # ToTensorV2 converts mask to (H, W) or (C, H, W) depending on input?
            # Usually ToTensorV2 doesn't transpose mask if it's not passed as 'image'.
            # But Albumentations returns mask as Tensor if ToTensorV2 is used.
            # Let's ensure channel first for mask: (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)

            # Ensure float for BCE loss
            mask = mask.float()

            return image, mask, z_tensor, self.ids[idx]
        else:
            return image, z_tensor, self.ids[idx]
