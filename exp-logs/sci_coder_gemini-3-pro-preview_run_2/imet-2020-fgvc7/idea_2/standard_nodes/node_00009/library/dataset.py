import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    Handles image loading, preprocessing, and multi-hot label encoding.
    """

    def __init__(self, df, root_dir, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, attribute_ids, file_path).
            root_dir (str): Root directory containing the images.
            transform (albumentations.Compose): Transformations to apply to the image.
            mode (str): 'train', 'val', or 'test'. Determines label handling.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., "train/abc.png")
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Handle missing or corrupt images gracefully
        if image is None:
            # Fallback: create a black image to prevent crashing
            # In a real scenario, we might log this, but here we just return a placeholder
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Targets
        if self.mode in ["train", "val"]:
            target = torch.zeros(self.num_classes, dtype=torch.float32)

            attr_ids = row["attribute_ids"]
            if isinstance(attr_ids, str) and len(attr_ids) > 0:
                # Parse space-separated IDs
                indices = [int(x) for x in attr_ids.split()]
                target[indices] = 1.0

            return image, target

        else:
            # Test mode: return image and ID (for submission mapping)
            return image, row["id"]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if mode == "train":
        return A.Compose(
            [
                # Resize first to ensure base size
                # Using RandomResizedCrop for slight scale variation (0.9 - 1.0)
                # This acts as both resizing and augmentation
                A.RandomResizedCrop(
                    size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic Resize
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    debug=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates and returns DataLoaders for training and validation.

    Args:
        debug (bool): If True, subsamples the dataset for quick debugging.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load Metadata
    # The metadata files are already generated in ./metadata
    if not os.path.exists(Config.TRAIN_META_PATH) or not os.path.exists(
        Config.VAL_META_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Debug Subsampling
    if debug:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"Debug mode: Train size {len(df_train)}, Val size {len(df_val)}")

    # Create Datasets
    train_dataset = ArtworkDataset(
        df=df_train,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train"),
        mode="train",
    )

    val_dataset = ArtworkDataset(
        df=df_val,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="val"),
        mode="val",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates and returns DataLoader for the test set.

    Returns:
        DataLoader: Test loader.
    """
    if not os.path.exists(Config.TEST_META_PATH):
        raise FileNotFoundError("Test metadata file not found.")

    df_test = pd.read_csv(Config.TEST_META_PATH)

    test_dataset = ArtworkDataset(
        df=df_test,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="test"),
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
