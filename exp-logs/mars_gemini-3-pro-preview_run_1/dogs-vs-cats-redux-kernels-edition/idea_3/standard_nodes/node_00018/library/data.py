import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import CFG
from library.utils import seed_everything


class CatDogDataset(Dataset):
    """
    Dataset class for loading Dog and Cat images.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        # Prepend input directory to filepaths
        self.file_paths = [os.path.join("./input", fp) for fp in df["filepath"].values]

        if self.mode in ["train", "val"]:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        if image is None:
            # Handle missing images gracefully or raise error
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            image = cv2.resize(image, (CFG.image_size, CFG.image_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode in ["train", "val"]:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return image, label
        else:
            # For test set, return dummy label
            return image, torch.tensor(0.0)


def get_transforms(data="train"):
    """
    Returns Albumentations transforms based on the data mode.
    """
    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(CFG.image_size, CFG.image_size),
                    scale=(CFG.train_aug_params["resize_crop_min_scale"], 1.0),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=15,
                    p=CFG.train_aug_params["shift_scale_rotate_prob"],
                ),
                A.ColorJitter(
                    brightness=CFG.train_aug_params["color_jitter_brightness"],
                    contrast=CFG.train_aug_params["color_jitter_contrast"],
                    saturation=CFG.train_aug_params["color_jitter_saturation"],
                    hue=CFG.train_aug_params["color_jitter_hue"],
                    p=CFG.train_aug_params["color_jitter_prob"],
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(height=CFG.image_size, width=CFG.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def prepare_data(load_cached_data=True):
    """
    Loads training data and creates folds.
    Uses caching to store the dataframe with fold assignments.
    """
    os.makedirs(CFG.output_dir, exist_ok=True)
    cache_path = os.path.join(CFG.output_dir, "train_folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        print("Creating folds...")
        if not os.path.exists(CFG.train_csv):
            raise FileNotFoundError(f"Train metadata not found at {CFG.train_csv}")

        df = pd.read_csv(CFG.train_csv)

        # Create Folds
        skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        df["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            df.loc[val_idx, "fold"] = fold

        # Save cache
        df.to_parquet(cache_path, index=False)
        print(f"Saved cached data to {cache_path}")

    return df


def get_loaders(fold, df):
    """
    Creates train and validation loaders for a specific fold.
    """
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    if CFG.debug:
        train_df = train_df.sample(
            n=min(len(train_df), 100), random_state=CFG.seed
        ).reset_index(drop=True)
        valid_df = valid_df.sample(
            n=min(len(valid_df), 50), random_state=CFG.seed
        ).reset_index(drop=True)

    train_dataset = CatDogDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    valid_dataset = CatDogDataset(
        valid_df, transforms=get_transforms("valid"), mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_loader():
    """
    Creates the test loader.
    """
    if not os.path.exists(CFG.test_csv):
        raise FileNotFoundError(f"Test metadata not found at {CFG.test_csv}")

    df = pd.read_csv(CFG.test_csv)

    dataset = CatDogDataset(df, transforms=get_transforms("test"), mode="test")

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
