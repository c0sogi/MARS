import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.HorizontalFlip(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms
        return A.Compose(
            [
                A.Resize(Config.img_size, Config.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class PawpularityDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None, mode="train"):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing the images (input folder).
            transform (callable, optional): Albumentations transform pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Handle Debug Mode: Sample a subset of data
        if Config.debug:
            sample_n = min(len(self.df), Config.debug_sample_size)
            self.df = self.df.sample(n=sample_n, random_state=Config.seed).reset_index(
                drop=True
            )

        # Feature Column Handling
        # Ensure we can map Config.feature_cols to the dataframe columns
        self.feature_cols = Config.feature_cols.copy()

        # Resolve potential naming discrepancy: "Subject Focus" (Config) vs "Focus" (Dataset)
        if (
            "Subject Focus" in self.feature_cols
            and "Subject Focus" not in self.df.columns
        ):
            if "Focus" in self.df.columns:
                self.df = self.df.rename(columns={"Focus": "Subject Focus"})

        # Verify features exist
        missing_features = [
            col for col in self.feature_cols if col not in self.df.columns
        ]
        if missing_features:
            print(
                f"Warning: The following feature columns are missing in {csv_path}: {missing_features}"
            )
            # In a strict pipeline, we might raise an error, but here we'll let it slide
            # assuming 0s or handling downstream, but for this task, we assume data integrity.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative, e.g., "train/id.jpg"
        file_path = row["file_path"]
        full_path = os.path.join(self.root_dir, file_path)

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images to prevent crash, though data check passed
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default transform if none provided
            T = get_transforms(mode="val")
            image = T(image=image)["image"]

        # 3. Extract Metadata Features
        # Get the 12 binary features
        features = row[self.feature_cols].values.astype(np.float32)
        features = torch.tensor(features)

        # 4. Extract Target
        target = 0.0
        if self.mode in ["train", "val"]:
            if Config.target_col in row:
                raw_target = row[Config.target_col]
                # Scale target from [1, 100] to [0, 1] for BCEWithLogitsLoss
                target = raw_target / 100.0

        return {
            "image": image,
            "features": features,
            "target": torch.tensor(target, dtype=torch.float32),
            "id": row["Id"],
        }


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    """
    # Define Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Initialize Datasets
    # Use val_transform for training to ensure deterministic feature extraction for Linear Probing
    train_dataset = PawpularityDataset(
        csv_path=Config.train_metadata_path,
        root_dir=Config.input_root,
        transform=val_transform,
        mode="train",
    )

    val_dataset = PawpularityDataset(
        csv_path=Config.val_metadata_path,
        root_dir=Config.input_root,
        transform=val_transform,
        mode="val",
    )

    test_dataset = PawpularityDataset(
        csv_path=Config.test_metadata_path,
        root_dir=Config.input_root,
        transform=test_transform,
        mode="test",
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
