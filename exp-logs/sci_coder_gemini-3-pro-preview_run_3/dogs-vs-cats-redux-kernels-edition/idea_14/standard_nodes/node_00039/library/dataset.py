import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    IMG_SIZE_TEACHER,
    IMG_SIZE_STUDENT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AUG_CROP_SCALE,
    AUG_COLOR_JITTER_INTENSITY,
    SEED,
)


def load_metadata(split, debug=False, load_cached_data=True):
    """
    Loads metadata for a given split (train, val, test).
    Implements caching using parquet to ensure deterministic loading and faster access.
    """
    cache_filename = f"metadata_{split}.parquet"
    cache_path = os.path.join(WORKING_DIR, cache_filename)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if debug:
                # Deterministic sampling for debug
                df = df.iloc[:200]
            return df
        except Exception:
            # If load fails, proceed to load from source
            pass

    # 2. Load from source
    if split == "train":
        path = TRAIN_METADATA_PATH
    elif split == "val":
        path = VAL_METADATA_PATH
    elif split == "test":
        path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to cache metadata to {cache_path}: {e}")

    if debug:
        df = df.iloc[:200]

    return df


def get_transforms(mode="train", img_size=256):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'.
        img_size (int): Target image size (height and width).
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size,
                    width=img_size,
                    scale=AUG_CROP_SCALE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=AUG_COLOR_JITTER_INTENSITY,
                    contrast=AUG_COLOR_JITTER_INTENSITY,
                    saturation=AUG_COLOR_JITTER_INTENSITY,
                    hue=0.1,
                    p=0.8,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test pipeline
        return A.Compose(
            [
                # Resize to target size.
                A.Resize(
                    height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )


class CatDogDataset(Dataset):
    """
    Standard Dataset for training single models (ResNet, ConvNeXt).
    """

    def __init__(self, split="train", transform=None, debug=False, img_size=256):
        self.split = split
        self.df = load_metadata(split, debug=debug)
        self.transform = transform
        self.img_size = img_size

        # If no transform provided, create default based on split and img_size
        if self.transform is None:
            mode = "train" if split == "train" else "val"
            self.transform = get_transforms(mode=mode, img_size=img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        img_path = os.path.join(INPUT_DIR, row["filepath"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen based on metadata check)
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get label/id
        if self.split == "test":
            # For test set, return ID as the target
            target = row["id"]
        else:
            # For train/val, return label (float for BCE)
            target = torch.tensor(row["label"], dtype=torch.float32)

        return image, target


class DualResolutionDataset(Dataset):
    """
    Dataset for Distillation.
    Returns:
        - img_teacher: Tensor (C, 256, 256)
        - img_student: Tensor (C, 224, 224)
        - target: Tensor (label)

    Ensures geometric consistency: The student image is a downscaled version of the teacher image
    (same crop, flip, and color jitter).
    """

    def __init__(self, split="train", debug=False):
        self.split = split
        self.df = load_metadata(split, debug=debug)

        # Base augmentation pipeline (geometric + color) -> Outputs Teacher Size
        # No Normalization or ToTensor here, as we need to resize the numpy array first.
        self.aug_transform = A.Compose(
            [
                A.RandomResizedCrop(
                    height=IMG_SIZE_TEACHER,
                    width=IMG_SIZE_TEACHER,
                    scale=AUG_CROP_SCALE,
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=AUG_COLOR_JITTER_INTENSITY,
                    contrast=AUG_COLOR_JITTER_INTENSITY,
                    saturation=AUG_COLOR_JITTER_INTENSITY,
                    hue=0.1,
                    p=0.8,
                ),
            ]
        )

        # Normalization and Tensor conversion pipeline
        self.norm_transform = A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(INPUT_DIR, row["filepath"])

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((IMG_SIZE_TEACHER, IMG_SIZE_TEACHER, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.split == "train":
            # 1. Apply base augmentation to get Teacher image (256x256)
            augmented = self.aug_transform(image=image)
            img_teacher_np = augmented["image"]

            # 2. Resize Teacher image to Student size (224x224) using Bicubic
            img_student_np = cv2.resize(
                img_teacher_np,
                (IMG_SIZE_STUDENT, IMG_SIZE_STUDENT),
                interpolation=cv2.INTER_CUBIC,
            )

            # 3. Normalize and convert to Tensor
            img_teacher = self.norm_transform(image=img_teacher_np)["image"]
            img_student = self.norm_transform(image=img_student_np)["image"]

        else:
            # Validation logic: Simple resize to respective sizes
            # Teacher
            t_np = cv2.resize(
                image,
                (IMG_SIZE_TEACHER, IMG_SIZE_TEACHER),
                interpolation=cv2.INTER_CUBIC,
            )
            img_teacher = self.norm_transform(image=t_np)["image"]

            # Student
            s_np = cv2.resize(
                image,
                (IMG_SIZE_STUDENT, IMG_SIZE_STUDENT),
                interpolation=cv2.INTER_CUBIC,
            )
            img_student = self.norm_transform(image=s_np)["image"]

        target = torch.tensor(row["label"], dtype=torch.float32)

        return img_teacher, img_student, target
