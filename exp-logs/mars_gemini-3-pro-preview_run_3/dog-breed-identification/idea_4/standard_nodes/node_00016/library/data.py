import os
import cv2
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


def get_class_mapping(df):
    """
    Generates a deterministic mapping between class names and indices.
    Assumes df contains a 'breed' column.
    """
    classes = sorted(df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_class = {idx: cls_name for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class


class DogDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mode="train", class_to_idx=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            root_dir (str): Root directory for images (usually Config.INPUT_DIR).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict, optional): Mapping from breed name to index. Required for train/val.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train/id.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using PIL (compatible with torchvision transforms)
        # Convert to RGB to handle grayscale or RGBA images consistently
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Fallback for missing/corrupt images (though metadata validation should catch this)
            # Create a black image of standard size
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            breed = row["breed"]
            label = self.class_to_idx[breed]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image and id
            img_id = row["id"]
            return image, img_id


def get_transforms(phase, resolution):
    """
    Returns the data augmentation pipeline based on the phase and resolution.

    Args:
        phase (str): 'train' or 'val'/'test'.
        resolution (int): Target input resolution (e.g., 224, 384).
    """
    # ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if phase == "train":
        return transforms.Compose(
            [
                # 1. RandomResizedCrop: Essential for scale invariance and preventing overfitting
                transforms.RandomResizedCrop(resolution, scale=(0.08, 1.0)),
                # 2. RandomHorizontalFlip: Basic spatial invariance
                transforms.RandomHorizontalFlip(),
                # 3. RandAugment: Strong regularization
                transforms.RandAugment(
                    num_ops=Config.RAND_AUGMENT_N, magnitude=Config.RAND_AUGMENT_M
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # Validation / Test / Inference
        # Resize to slightly larger than target, then center crop
        # Standard ratio is often 256/224 = 1.14
        resize_dim = int(resolution * 256 / 224)

        return transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )


def get_dataloaders(resolution, batch_size, debug=False):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        resolution (int): Image resolution for transforms.
        batch_size (int): Batch size for DataLoaders.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader, classes
    """
    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    df_val = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Generate Class Mapping from Training Data
    # This ensures consistency across runs as long as train data is static
    classes, class_to_idx, _ = get_class_mapping(df_train)

    # Verify we have the expected number of classes
    if len(classes) != Config.NUM_CLASSES:
        print(f"Warning: Found {len(classes)} classes, expected {Config.NUM_CLASSES}")

    # Debug Mode: Subset data
    if debug:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Define Transforms
    train_transform = get_transforms("train", resolution)
    val_transform = get_transforms("val", resolution)
    # Test transform is same as val (deterministic resize + crop)
    test_transform = get_transforms("test", resolution)

    # Create Datasets
    train_dataset = DogDataset(
        df_train,
        root_dir=Config.INPUT_DIR,
        transform=train_transform,
        mode="train",
        class_to_idx=class_to_idx,
    )

    val_dataset = DogDataset(
        df_val,
        root_dir=Config.INPUT_DIR,
        transform=val_transform,
        mode="val",
        class_to_idx=class_to_idx,
    )

    test_dataset = DogDataset(
        df_test,
        root_dir=Config.INPUT_DIR,
        transform=test_transform,
        mode="test",
        class_to_idx=None,  # Not needed for test
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches during training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, classes
