import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library import config
from library import utils


class DogCatDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filepath, label/id).
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'. Determines what is returned.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata filepaths are relative to INPUT_DIR (e.g., "train/cat.0.jpg")
        img_path = os.path.join(config.INPUT_DIR, row["filepath"])

        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images or errors (though dataset is assumed clean)
            # Create a black image
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.mode in ["train", "val"]:
            # Return image and label (float for potential BCE loss)
            label = row["label"]
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            # Return image and id for submission
            img_id = row["id"]
            return image, torch.tensor(img_id, dtype=torch.long)


def get_transforms(stage="train"):
    """
    Returns the transformation pipeline for the specified stage.

    Args:
        stage (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if stage == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # For validation and test, we simply resize to the target size.
        # Note: The prompt specifies resizing to 224x224.
        return transforms.Compose(
            [
                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def get_dataloaders(
    train_meta_path=config.TRAIN_META,
    val_meta_path=config.VAL_META,
    test_meta_path=config.TEST_META,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    debug_subset_size=config.DEBUG_SUBSET_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_meta_path (str): Path to training metadata CSV.
        val_meta_path (str): Path to validation metadata CSV.
        test_meta_path (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug_subset_size (int or None): If set, limits dataset size for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(train_meta_path)
    val_df = pd.read_csv(val_meta_path)
    test_df = pd.read_csv(test_meta_path)

    # Debugging: Subset data if requested
    if debug_subset_size is not None:
        train_df = train_df.iloc[:debug_subset_size]
        val_df = val_df.iloc[:debug_subset_size]
        test_df = test_df.iloc[:debug_subset_size]

    # Initialize Datasets
    train_dataset = DogCatDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_dataset = DogCatDataset(val_df, transform=get_transforms("val"), mode="val")
    test_dataset = DogCatDataset(test_df, transform=get_transforms("test"), mode="test")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
