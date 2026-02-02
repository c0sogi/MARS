import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from timm.data import Mixup

from library.config import Config
from library.utils import seed_everything


class CassavaDataset(Dataset):
    """
    Dataset class for Cassava Leaf Disease Classification.
    Uses PIL for image loading to ensure compatibility with torchvision transforms.
    """

    def __init__(self, dataframe, transform=None, output_label=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image_id, label, and file_path.
            transform (callable, optional): Optional transform to be applied on a sample.
            output_label (bool): Whether to return the label (True for train/val, False for test).
        """
        self.df = dataframe
        self.transform = transform
        self.output_label = output_label
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train_images/1000015157.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        try:
            # Use PIL as requested for native alignment with augmentations
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback to a black image in case of corruption to prevent crash
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.output_label:
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            return image


def get_transforms(img_size, is_train=True):
    """
    Generates the augmentation pipeline based on the training phase and image size.

    Args:
        img_size (int): Target image resolution (e.g., 224 or 384).
        is_train (bool): Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    # ImageNet statistics
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose(
            [
                # Geometric Augmentations
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.08, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # Photometric Augmentations (RandAugment)
                transforms.RandAugment(num_ops=2, magnitude=9),
                # Conversion and Normalization
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        # Validation/Test Transform
        # Resize to slightly larger than target (maintain crop_pct ~0.875) then CenterCrop
        scale_size = int(img_size / 0.875)
        return transforms.Compose(
            [
                transforms.Resize(
                    scale_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def get_metadata(load_cached_data=True):
    """
    Loads metadata, merges train/val splits, and generates stratified folds.
    Implements caching to ensure deterministic splits across runs.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with a 'fold' column.
    """
    cache_path = os.path.join(Config.OUTPUT_DIR, "folds.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Verify it has the fold column
            if "fold" in df.columns:
                return df
        except Exception:
            pass  # Fallback to regeneration if load fails

    # 2. Generate from scratch
    # Load original metadata provided in ./metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge them to create a full dataset for 5-fold CV
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold_idx

    # 3. Save to cache
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_mixup_fn(mixup_prob, label_smoothing=0.0):
    """
    Returns a Mixup function/object based on the provided probability and smoothing.
    Uses timm.data.Mixup.

    Args:
        mixup_prob (float): Probability of applying Mixup/CutMix.
        label_smoothing (float): Label smoothing value (0.0 to 1.0).

    Returns:
        Mixup object or None.
    """
    # If mixup is disabled but label smoothing is requested, we still use Mixup class
    # with mixup_alpha=0 to handle the label smoothing and target conversion.
    if mixup_prob > 0 or label_smoothing > 0:
        return Mixup(
            mixup_alpha=0.8 if mixup_prob > 0 else 0.0,
            cutmix_alpha=1.0 if mixup_prob > 0 else 0.0,
            prob=(
                mixup_prob if mixup_prob > 0 else 1.0
            ),  # If only smoothing, apply always
            switch_prob=0.5,
            mode="batch",
            label_smoothing=label_smoothing,
            num_classes=Config.NUM_CLASSES,
        )
    return None


def get_dataloaders(fold_idx, img_size, batch_size, load_cached_data=True):
    """
    Creates Train and Validation DataLoaders for a specific fold.

    Args:
        fold_idx (int): The fold index to use for validation (0 to N_FOLDS-1).
        img_size (int): Image resolution.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached fold splits.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load and split data
    df = get_metadata(load_cached_data=load_cached_data)

    df_train = df[df["fold"] != fold_idx].reset_index(drop=True)
    df_val = df[df["fold"] == fold_idx].reset_index(drop=True)

    # Debug mode: reduce data size
    if Config.DEBUG:
        df_train = df_train.iloc[:100]
        df_val = df_val.iloc[:50]

    # Create Datasets
    train_dataset = CassavaDataset(
        df_train, transform=get_transforms(img_size, is_train=True), output_label=True
    )

    val_dataset = CassavaDataset(
        df_val, transform=get_transforms(img_size, is_train=False), output_label=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(img_size, batch_size):
    """
    Creates the Test DataLoader using the sample submission metadata.

    Args:
        img_size (int): Image resolution.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Test data loader.
    """
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = CassavaDataset(
        df_test,
        transform=get_transforms(img_size, is_train=False),
        output_label=False,  # Test data usually has dummy labels
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
