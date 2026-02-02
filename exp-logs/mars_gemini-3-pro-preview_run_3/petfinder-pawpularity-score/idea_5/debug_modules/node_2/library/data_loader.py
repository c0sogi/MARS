import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config

# Metadata feature columns as defined in the dataset description
METADATA_COLS = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


class PetDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity dataset.
    Loads images, applies transformations, and retrieves metadata and targets.
    """

    def __init__(self, df, transforms=None, mode="train", input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            transforms (albumentations.Compose): Transformations to apply to images.
            mode (str): 'train', 'val', or 'test'.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = input_dir

        # Pre-construct full file paths
        # df['file_path'] is relative (e.g., "train/id.jpg")
        self.file_paths = (
            self.df["file_path"].apply(lambda x: os.path.join(self.input_dir, x)).values
        )
        self.ids = self.df["Id"].values

        # Extract binary metadata features
        self.meta_features = self.df[METADATA_COLS].values.astype(np.float32)

        # Extract targets for training/validation
        if self.mode in ["train", "val"]:
            self.targets = self.df["Pawpularity"].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load image using OpenCV
        image = cv2.imread(file_path)

        # Handle potential missing or corrupt images
        if image is None:
            # Return a black image of correct size to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Prepare metadata tensor
        meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        item = {"image": image, "metadata": meta, "id": self.ids[idx]}

        # Add target if available
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            item["target"] = target

        return item


def get_transforms(image_size):
    """
    Returns the Albumentations composition for image preprocessing.
    Includes Resizing, Normalization (ImageNet stats), and conversion to Tensor.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )


def get_dataloaders(debug=Config.DEBUG):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, loads a small subset of the data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Subset for debugging if requested
    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Get transformations
    transforms = get_transforms(Config.IMAGE_SIZE)

    # Instantiate Datasets
    train_dataset = PetDataset(train_df, transforms=transforms, mode="train")
    val_dataset = PetDataset(val_df, transforms=transforms, mode="val")
    test_dataset = PetDataset(test_df, transforms=transforms, mode="test")

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
