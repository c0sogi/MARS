import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from library.config import Config


def get_transforms(image_size, is_train=True):
    """
    Returns the data augmentation and normalization pipeline.

    Args:
        image_size (int): Target spatial resolution (e.g., 224 or 256).
        is_train (bool): Whether to apply training augmentations.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if is_train:
        return transforms.Compose(
            [
                # Context-Preserving Augmentation: RandomResizedCrop with scale (0.8, 1.0)
                # Ensures the subject is not cropped out while providing scale invariance.
                transforms.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.8, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                # ColorJitter with intensity >= 0.2 to handle lighting variance
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                # Strictly use Bicubic Interpolation for validation/test to maintain high fidelity
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class PetDataset(Dataset):
    """
    Dataset class for loading Dog vs Cat images.
    Handles reading from metadata CSVs, loading images via OpenCV,
    converting to PIL for transforms, and returning tensors.
    """

    def __init__(self, csv_path, mode="train", transform=None, debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Transform pipeline.
            debug (bool): If True, use a small subset of data for debugging.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SUBSET_SIZE]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct path
        rel_path = row["filepath"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        img = cv2.imread(full_path)

        if img is None:
            # Handle missing/corrupt images gracefully by returning a blank image
            # This prevents the dataloader from crashing during long runs
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms (ensures Bicubic support)
        img = Image.fromarray(img)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Return based on mode
        if self.mode in ["train", "val"]:
            label = row["label"]
            # Return float tensor for BCEWithLogitsLoss
            return img, torch.tensor(label, dtype=torch.float32)
        elif self.mode == "test":
            img_id = row["id"]
            return img, img_id
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def get_dataloaders(resolution, batch_size, debug=False):
    """
    Creates DataLoaders for training and validation.

    Args:
        resolution (int): Image resolution (e.g., 224, 256).
        batch_size (int): Batch size.
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Transforms
    train_transform = get_transforms(resolution, is_train=True)
    val_transform = get_transforms(resolution, is_train=False)

    # Datasets
    train_dataset = PetDataset(
        Config.TRAIN_METADATA, mode="train", transform=train_transform, debug=debug
    )
    val_dataset = PetDataset(
        Config.VAL_METADATA, mode="val", transform=val_transform, debug=debug
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
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


def get_test_dataloader(resolution, batch_size, debug=False):
    """
    Creates DataLoader for testing.

    Args:
        resolution (int): Image resolution.
        batch_size (int): Batch size.
        debug (bool): Debug mode.

    Returns:
        DataLoader: test_loader
    """
    test_transform = get_transforms(resolution, is_train=False)

    test_dataset = PetDataset(
        Config.TEST_METADATA, mode="test", transform=test_transform, debug=debug
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
