import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Uses PIL for image loading to align with torchvision transforms.
    """

    def __init__(self, df, data_root, transform=None, output_label=True):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transform = transform
        self.output_label = output_label

        # Ensure file_path column exists or construct it
        if "file_path" not in self.df.columns:
            # Fallback if file_path is missing, though metadata usually has it
            # Assuming images are in train_images or test_images based on context
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Construct full file path
        # The metadata 'file_path' is relative to input_dir (e.g., "train_images/xyz.jpg")
        img_path = os.path.join(self.data_root, row["file_path"])

        try:
            # Use PIL as requested
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in case of error to prevent crash
            image = Image.new("RGB", (Config.image_size, Config.image_size))

        if self.transform:
            image = self.transform(image)

        if self.output_label:
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            return image


def get_transforms(data_type, cfg):
    """
    Returns torchvision transforms for training or validation/test.
    """
    if data_type == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=2, magnitude=9
                ),  # Standard RandAugment settings
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif data_type == "valid" or data_type == "test":
        return transforms.Compose(
            [
                # Resize to slightly larger than target size, then center crop
                transforms.Resize(int(cfg.image_size * 256 / 224)),
                transforms.CenterCrop(cfg.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def prepare_folds(load_cached_data=True):
    """
    Merges train and validation metadata and creates stratified K-Folds.
    Caches the result to a parquet file.
    """
    cache_path = os.path.join(Config.working_dir, "folds.parquet")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        df = pd.read_parquet(cache_path)
        return df

    print("Creating new folds...")
    # Load original metadata
    train_meta = pd.read_csv(Config.train_metadata_path)
    val_meta = pd.read_csv(Config.val_metadata_path)

    # Merge them
    df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # Create Stratified K-Folds
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold

    # Cache the result
    df.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return df


def get_dataloaders(fold_id, cfg):
    """
    Creates DataLoaders for a specific fold.
    """
    # Prepare data with folds
    df = prepare_folds(load_cached_data=True)

    # Split into train and validation for this fold
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    valid_df = df[df["fold"] == fold_id].reset_index(drop=True)

    # Debug mode: subset data
    if cfg.debug:
        train_df = train_df.head(cfg.debug_sample_size)
        valid_df = valid_df.head(cfg.debug_sample_size)
        print(
            f"DEBUG MODE: Reduced train size to {len(train_df)} and valid size to {len(valid_df)}"
        )

    # Create Datasets
    train_dataset = CassavaDataset(
        train_df,
        Config.input_dir,
        transform=get_transforms("train", cfg),
        output_label=True,
    )

    valid_dataset = CassavaDataset(
        valid_df,
        Config.input_dir,
        transform=get_transforms("valid", cfg),
        output_label=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def get_test_dataloader(cfg):
    """
    Creates DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.test_metadata_path)

    test_dataset = CassavaDataset(
        df_test,
        Config.input_dir,
        transform=get_transforms("test", cfg),
        output_label=False,  # Test set labels in metadata are placeholders
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
