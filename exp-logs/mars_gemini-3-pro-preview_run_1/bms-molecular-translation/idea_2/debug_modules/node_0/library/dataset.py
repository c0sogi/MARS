import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.tokenizer import Tokenizer


class BmsDataset(Dataset):
    """
    Custom Dataset for loading BMS Molecular Translation data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        config: Config,
        transform=None,
        mode: str = "train",
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.transform = transform
        self.mode = mode
        self.input_dir = config.input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Construct full image path
        full_path = os.path.join(self.input_dir, file_path)

        # Read image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though verification script passed)
            # Create a black image of correct size
            image = np.zeros(
                (self.config.image_size[0], self.config.image_size[1], 3),
                dtype=np.uint8,
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic transform if none provided: Resize and ToTensor
            resize = A.Compose(
                [
                    A.Resize(self.config.image_size[0], self.config.image_size[1]),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            augmented = resize(image=image)
            image = augmented["image"]

        # Handle labels
        if self.mode in ["train", "valid"]:
            inchi_text = row["InChI"]
            sequence = self.tokenizer.text_to_sequence(inchi_text)
            seq_len = torch.sum(sequence != self.tokenizer.PAD_IDX)
            return image, sequence, seq_len
        else:
            # Test mode: return image and original text (placeholder) or just ID
            # Returning placeholder text/ID allows submission creation
            inchi_text = row["InChI"] if "InChI" in row else ""
            return image, inchi_text


def get_transforms(config: Config, mode: str = "train"):
    """
    Returns albumentations transforms for train/valid/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config.image_size[0], config.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.image_size[0], config.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_loaders(config: Config, tokenizer: Tokenizer):
    """
    Initializes and returns DataLoaders for train, validation, and test sets.
    """
    print("Initializing DataLoaders...")

    # Load Metadata
    if not os.path.exists(config.train_metadata_path):
        raise FileNotFoundError(
            f"Train metadata not found at {config.train_metadata_path}"
        )
    if not os.path.exists(config.val_metadata_path):
        raise FileNotFoundError(f"Val metadata not found at {config.val_metadata_path}")
    if not os.path.exists(config.test_metadata_path):
        raise FileNotFoundError(
            f"Test metadata not found at {config.test_metadata_path}"
        )

    train_df = pd.read_csv(config.train_metadata_path)
    val_df = pd.read_csv(config.val_metadata_path)
    test_df = pd.read_csv(config.test_metadata_path)

    # Debug Mode: Subset data
    if config.debug and config.subset_size:
        print(f"Debug mode enabled. Subsetting data to {config.subset_size} samples.")
        train_df = train_df.iloc[: config.subset_size]
        val_df = val_df.iloc[: config.subset_size]
        test_df = test_df.iloc[: config.subset_size]

    # Transforms
    train_transform = get_transforms(config, mode="train")
    val_transform = get_transforms(config, mode="valid")
    test_transform = get_transforms(config, mode="test")

    # Datasets
    train_dataset = BmsDataset(
        train_df, tokenizer, config, transform=train_transform, mode="train"
    )
    val_dataset = BmsDataset(
        val_df, tokenizer, config, transform=val_transform, mode="valid"
    )
    test_dataset = BmsDataset(
        test_df, tokenizer, config, transform=test_transform, mode="test"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
