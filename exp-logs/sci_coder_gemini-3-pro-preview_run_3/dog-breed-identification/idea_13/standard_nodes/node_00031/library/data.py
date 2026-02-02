import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from library.config import Config

# ImageNet normalization constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DogDataset(Dataset):
    """
    Custom Dataset for Dog Breed Classification.
    Handles loading images via OpenCV, converting to RGB, and applying transforms.
    """

    def __init__(self, df, transform=None, mode="train", label_encoder=None):
        self.df = df
        self.transform = transform
        self.mode = mode
        self.label_encoder = label_encoder

        # Pre-compute full file paths
        # Metadata contains relative paths (e.g., 'train/id.jpg')
        self.file_paths = [
            os.path.join(Config.input_dir, fp) for fp in df["file_path"].values
        ]

        if self.mode != "test":
            self.labels = df["breed"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # Load image
        img = cv2.imread(path)
        if img is None:
            # Fallback for corrupt/missing images (should not happen given metadata validation)
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms compatibility
        img = transforms.ToPILImage()(img)

        # Apply augmentations
        if self.transform:
            img = self.transform(img)

        if self.mode == "test":
            return img
        else:
            label_name = self.labels[idx]
            label_idx = self.label_encoder[label_name]
            return img, torch.tensor(label_idx, dtype=torch.long)


def get_transforms(mode="train", image_size=224):
    """
    Constructs the data augmentation pipeline.
    Strictly follows the strategy: RandomResizedCrop -> RandomHorizontalFlip -> RandAugment.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        # Validation/Test: Resize to slightly larger then CenterCrop
        # Standard ImageNet evaluation protocol
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )


def get_data_splits(load_cached_data=True):
    """
    Merges provided train/val metadata and generates 5-fold stratified splits.
    Caches the result to ensure fold consistency across runs.
    """
    cache_path = os.path.join(Config.working_dir, "train_folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load original metadata
    train_meta = pd.read_csv(Config.train_metadata_path)
    val_meta = pd.read_csv(Config.val_metadata_path)

    # Concatenate to form the full training dataset
    full_df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Generate Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    full_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["breed"])):
        full_df.loc[val_idx, "fold"] = fold

    # Cache the dataframe
    os.makedirs(Config.working_dir, exist_ok=True)
    full_df.to_parquet(cache_path)

    return full_df


def get_classes(df):
    """
    Extracts and sorts unique breed names to create a consistent label encoding.
    """
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates training and validation DataLoaders for a specific fold.
    """
    # Load dataset with fold assignments
    df = get_data_splits(load_cached_data=load_cached_data)

    # Split data based on fold index
    train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
    val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Debug mode: subset data
    if Config.debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)

    # Generate class mapping
    classes, class_to_idx = get_classes(df)

    # Initialize Datasets
    train_dataset = DogDataset(
        train_df,
        transform=get_transforms(mode="train", image_size=Config.image_size),
        mode="train",
        label_encoder=class_to_idx,
    )
    val_dataset = DogDataset(
        val_df,
        transform=get_transforms(mode="val", image_size=Config.image_size),
        mode="val",
        label_encoder=class_to_idx,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, classes


def get_test_dataloader():
    """
    Creates the DataLoader for the test set.
    """
    test_df = pd.read_csv(Config.test_metadata_path)

    test_dataset = DogDataset(
        test_df,
        transform=get_transforms(mode="test", image_size=Config.image_size),
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader, test_df["id"].values
