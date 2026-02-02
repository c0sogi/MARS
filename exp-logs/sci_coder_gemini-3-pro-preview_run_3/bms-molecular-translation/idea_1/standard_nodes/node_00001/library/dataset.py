import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import InchiTokenizer


def get_transforms(phase: str):
    """
    Returns the image transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # val or test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for Chemical Image to InChI translation.
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer: InchiTokenizer, transform=None, mode="train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, file_path, InChI).
            tokenizer (InchiTokenizer): Tokenizer instance for text processing.
            transform (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode

        # Pre-check columns
        if "file_path" not in df.columns:
            raise ValueError("DataFrame must contain 'file_path' column.")
        if mode in ["train", "val"] and "InChI" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'InChI' column for train/val mode."
            )
        if mode == "test" and "image_id" not in df.columns:
            raise ValueError("DataFrame must contain 'image_id' column for test mode.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing images, though metadata validation should prevent this
            # Creating a black image to avoid crashing training
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided: resize and to tensor
            T = A.Compose(
                [
                    A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            image = T(image=image)["image"]

        if self.mode in ["train", "val"]:
            inchi_text = row["InChI"]
            # Tokenize with padding to MAX_TEXT_LEN
            seq = self.tokenizer.text_to_sequence(
                inchi_text, max_len=Config.MAX_TEXT_LEN, padding=True
            )
            return image, seq

        elif self.mode == "test":
            image_id = row["image_id"]
            return image, image_id

        else:
            raise ValueError(f"Invalid mode: {self.mode}")


def get_dataloaders(debug=False, debug_size=1000):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        debug (bool): If True, use a small subset of data.
        debug_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Initialize Tokenizer
    tokenizer = InchiTokenizer()

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug Mode
    if debug:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)
        print(f"[DEBUG] Dataframes truncated to {debug_size} samples.")

    # Initialize Datasets
    train_dataset = ChemicalDataset(
        train_df, tokenizer, transform=get_transforms("train"), mode="train"
    )

    val_dataset = ChemicalDataset(
        val_df, tokenizer, transform=get_transforms("val"), mode="val"
    )

    test_dataset = ChemicalDataset(
        test_df, tokenizer, transform=get_transforms("test"), mode="test"
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, tokenizer
