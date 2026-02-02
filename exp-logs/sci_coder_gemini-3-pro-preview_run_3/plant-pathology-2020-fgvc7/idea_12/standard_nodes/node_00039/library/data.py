import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def load_dataset_dataframe(split="train", load_cached_data=True):
    """
    Loads the dataframe for the specified split (train, val, test).
    Implements strict caching logic using Parquet files.
    """
    # Ensure cache directory exists
    os.makedirs(Config.cache_dir, exist_ok=True)

    cache_path = os.path.join(Config.cache_dir, f"{split}_df.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # print(f"Loaded {split} data from cache: {cache_path}")
            return df
        except Exception:
            # print(f"Failed to load {split} cache. Recomputing...")
            pass

    # 2. Load from metadata CSVs
    if split == "train":
        csv_path = Config.train_csv_path
    elif split == "val":
        csv_path = Config.val_csv_path
    elif split == "test":
        csv_path = Config.test_csv_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        # print(f"Saved {split} data to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}. Error: {e}")

    return df


def get_transforms(data="train", img_size=512):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.
        img_size (int): Target image resolution.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                # Strong Geometric Augmentation
                A.HorizontalFlip(p=0.5),
                # Exclude VerticalFlip/Transpose per domain knowledge (gravity priors)
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=Config.aug_prob,
                ),
                # No Cutout/CoarseDropout (preserves small lesions)
                # No Brightness/Contrast (preserves color signals)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    elif data in ["valid", "test"]:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transform=None, data_root=None, is_test=False):
        self.df = df
        self.transform = transform
        self.data_root = data_root if data_root else Config.input_root
        self.is_test = is_test
        self.image_ids = self.df["image_id"].values

        # Pre-calculate file paths to avoid overhead in __getitem__
        # Metadata 'file_path' is relative e.g., 'images/Train_0.jpg'
        self.file_paths = [
            os.path.join(self.data_root, fp) for fp in self.df["file_path"].values
        ]

        if not self.is_test:
            self.labels = self.df[Config.class_labels].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image_id = self.image_ids[idx]

        # Load image
        img = cv2.imread(path)
        if img is None:
            # Handle missing image gracefully (though metadata check should prevent this)
            # Create a black image of default size to prevent crash
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.is_test:
            return img, image_id
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label


def get_loaders(df_train, df_val, df_test, model_cfg):
    """
    Creates DataLoaders for train, val, and test sets based on model config.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data (optional).
        model_cfg (dict): Configuration dictionary containing 'img_size' and 'batch_size'.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    img_size = model_cfg["img_size"]
    batch_size = model_cfg["batch_size"]

    # Transforms
    train_transform = get_transforms(data="train", img_size=img_size)
    val_transform = get_transforms(data="valid", img_size=img_size)

    # Datasets
    train_dataset = AppleDataset(
        df_train, transform=train_transform, data_root=Config.input_root, is_test=False
    )

    val_dataset = AppleDataset(
        df_val, transform=val_transform, data_root=Config.input_root, is_test=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = None
    if df_test is not None:
        test_dataset = AppleDataset(
            df_test,
            transform=val_transform,  # Use validation transform for test
            data_root=Config.input_root,
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader, test_loader
