import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config, seed_everything


class CactusDataset(Dataset):
    """
    Custom Dataset for loading Cactus images from disk.
    """

    def __init__(self, metadata_path, root_dir, transform=None, debug=False):
        """
        Args:
            metadata_path (str): Path to the CSV file containing image IDs and labels.
            root_dir (str): Root directory containing the images (e.g., ./input).
            transform (callable, optional): Optional transform to be applied on a sample.
            debug (bool): If True, use a small subset of the data for debugging.
        """
        self.df = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform

        if debug:
            self.df = self.df.sample(n=100, random_state=Config.SEED).reset_index(
                drop=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve metadata for the current index
        row = self.df.iloc[idx]

        # Construct the full image path
        # The 'file_path' in metadata is relative to the input directory (e.g., "train/id.jpg")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = transforms.ToTensor()(image)

        # Retrieve label
        # Ensure label is a float for compatibility with BCELoss/BCEWithLogitsLoss
        label = row["has_cactus"]

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase):
    """
    Generates the transformation pipeline based on the phase (train/val/test).

    Args:
        phase (str): The phase of execution ('train', 'val', 'test').

    Returns:
        torchvision.transforms.Compose: The composed transformations.
    """
    # Common normalization: Maps [0, 1] to [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),  # Convert numpy array to PIL Image for torchvision transforms
                transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.AUG_VFLIP_PROB),
                transforms.RandomRotation(degrees=Config.AUG_ROTATION_DEGREES),
                transforms.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER_BRIGHTNESS,
                    contrast=Config.AUG_COLOR_JITTER_CONTRAST,
                    saturation=Config.AUG_COLOR_JITTER_SATURATION,
                    hue=Config.AUG_COLOR_JITTER_HUE,
                ),
                transforms.ToTensor(),  # Converts PIL Image to Tensor (C, H, W) in [0, 1]
                normalize,
            ]
        )
    else:
        # Validation and Test phases (No augmentation, just formatting)
        return transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), normalize]
        )


def get_dataloaders(debug=Config.DEBUG):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        debug (bool): Whether to run in debug mode (subset of data).

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    seed_everything(Config.SEED)

    # Define transforms for each phase
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    # Instantiate Datasets
    # Config.INPUT_DIR is "./input"
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=train_transform,
        debug=debug,
    )

    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=val_transform,
        debug=debug,
    )

    test_dataset = CactusDataset(
        metadata_path=Config.TEST_METADATA_PATH,
        root_dir=Config.INPUT_DIR,
        transform=test_transform,
        debug=debug,
    )

    # Instantiate DataLoaders
    # Pin memory speeds up host-to-device transfer for CUDA
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop last incomplete batch to maintain batch size consistency
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
