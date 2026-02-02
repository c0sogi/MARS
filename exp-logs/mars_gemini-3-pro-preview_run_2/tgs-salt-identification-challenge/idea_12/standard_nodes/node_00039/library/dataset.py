import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
IDEA_DIR = "./working/idea_12"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMG_SIZE = 128
ORIG_SIZE = 101


def load_and_cache_data(load_cached=True):
    """
    Loads data from disk or cache.
    Strictly follows the requirement: Check cache -> If missing/forced, compute & save.
    """
    os.makedirs(IDEA_DIR, exist_ok=True)

    cache_files = {
        "train_images": os.path.join(IDEA_DIR, "train_images.npy"),
        "train_masks": os.path.join(IDEA_DIR, "train_masks.npy"),
        "train_depths": os.path.join(IDEA_DIR, "train_depths.npy"),
        "val_images": os.path.join(IDEA_DIR, "val_images.npy"),
        "val_masks": os.path.join(IDEA_DIR, "val_masks.npy"),
        "val_depths": os.path.join(IDEA_DIR, "val_depths.npy"),
        "test_images": os.path.join(IDEA_DIR, "test_images.npy"),
        "test_depths": os.path.join(IDEA_DIR, "test_depths.npy"),
        "test_ids": os.path.join(IDEA_DIR, "test_ids.npy"),
    }

    all_exist = all(os.path.exists(f) for f in cache_files.values())

    if load_cached and all_exist:
        data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
        return data

    def load_set(df, is_test=False):
        images = []
        masks = []
        depths = []
        ids = []

        for idx, row in df.iterrows():
            img_path = os.path.join(INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)

            images.append(img)
            depths.append(row["z"])
            ids.append(row["id"])

            if not is_test:
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
                mask = (mask > 127).astype(np.uint8)
                masks.append(mask)

        return (
            np.array(images),
            np.array(masks) if not is_test else None,
            np.array(depths),
            np.array(ids),
        )

    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    train_imgs, train_msks, train_d, _ = load_set(train_df)
    val_imgs, val_msks, val_d, _ = load_set(val_df)
    test_imgs, _, test_d, test_ids = load_set(test_df, is_test=True)

    np.save(cache_files["train_images"], train_imgs)
    np.save(cache_files["train_masks"], train_msks)
    np.save(cache_files["train_depths"], train_d)
    np.save(cache_files["val_images"], val_imgs)
    np.save(cache_files["val_masks"], val_msks)
    np.save(cache_files["val_depths"], val_d)
    np.save(cache_files["test_images"], test_imgs)
    np.save(cache_files["test_depths"], test_d)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "train_images": train_imgs,
        "train_masks": train_msks,
        "train_depths": train_d,
        "val_images": val_imgs,
        "val_masks": val_msks,
        "val_depths": val_d,
        "test_images": test_imgs,
        "test_depths": test_d,
        "test_ids": test_ids,
    }


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        transform=None,
        depth_mean=0,
        depth_std=1,
        training=True,
    ):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]

        if self.depth_std > 0:
            z = (depth - self.depth_mean) / self.depth_std
        else:
            z = depth - self.depth_mean

        # Bernoulli Depth Masking
        # If training: 50% chance to mask depth to 0 (mean)
        if self.training:
            if np.random.rand() < 0.5:
                z = 0.0
        # If testing (no masks provided): Force depth to 0 as per strategy
        elif self.masks is None:
            z = 0.0

        if self.masks is not None:
            mask = self.masks[idx]
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Ensure mask has channel dimension (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return (
                image,
                mask.float(),
                torch.tensor([z], dtype=torch.float32),
            )
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image, torch.tensor([z], dtype=torch.float32)


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ]
        )
