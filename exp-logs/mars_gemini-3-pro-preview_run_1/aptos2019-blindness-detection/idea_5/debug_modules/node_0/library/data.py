import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


def get_transforms(phase: str, image_size: int = 512):
    """
    Returns the albumentations transformation pipeline.

    Args:
        phase (str): 'train' for augmentation, 'val' or 'test' for deterministic resizing.
        image_size (int): Target image size (height and width).

    Returns:
        A.Compose: The transformation pipeline.
    """
    # Normalization statistics (ImageNet)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Simple Resizing ("Squashing") as per strategy
                A.Resize(height=image_size, width=image_size),
                # Geometric Invariance Suite
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization and Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Deterministic resizing for validation/inference
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class RetinopathyDataset(Dataset):
    """
    Custom Dataset for Diabetic Retinopathy Severity Prediction.
    Handles image loading, preprocessing, and ordinal target generation.
    """

    def __init__(
        self, df: pd.DataFrame, input_dir: str, transforms=None, mode: str = "train"
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id_code, file_path, [diagnosis]).
            input_dir (str): Root directory for input images.
            transforms (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.input_dir = input_dir
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative, e.g., "train_images/xxxx.png"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt images (should not happen based on metadata check)
            # Return a black image of correct size to avoid crashing
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            image = self.transforms(image=image)["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = row["diagnosis"]

            # Generate Ordinal Targets
            # For K=5 classes (0-4), we have K-1=4 binary outputs.
            # Target is a vector of size 4.
            # y_i = 1 if label > i, else 0
            # Example: Label 2 -> [1, 1, 0, 0]
            target = torch.zeros(Config.num_outputs, dtype=torch.float32)
            if label > 0:
                # Set the first 'label' elements to 1
                # e.g. label=2: indices 0 and 1 become 1.
                target[:label] = 1.0

            return image, target
        else:
            # Test mode: return image and id_code for submission generation
            return image, row["id_code"]


def get_dataloaders(
    train_csv_path: str = Config.train_meta_path,
    val_csv_path: str = Config.val_meta_path,
    test_csv_path: str = Config.test_meta_path,
    input_dir: str = Config.input_dir,
    batch_size: int = Config.batch_size,
    image_size: int = Config.image_size,
    num_workers: int = Config.num_workers,
    debug: bool = Config.debug,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_csv_path (str): Path to training metadata CSV.
        val_csv_path (str): Path to validation metadata CSV.
        test_csv_path (str): Path to test metadata CSV.
        input_dir (str): Root directory of input data.
        batch_size (int): Batch size.
        image_size (int): Image resolution.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata DataFrames
    df_train = pd.read_csv(train_csv_path)
    df_val = pd.read_csv(val_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Debug Mode: Subsample data
    if debug:
        print("DEBUG MODE: Using small subset of data.")
        df_train = df_train.sample(
            n=min(len(df_train), 32), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), 32), random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(len(df_test), 32), random_state=Config.seed
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = RetinopathyDataset(
        df_train,
        input_dir,
        transforms=get_transforms("train", image_size),
        mode="train",
    )

    val_dataset = RetinopathyDataset(
        df_val, input_dir, transforms=get_transforms("val", image_size), mode="val"
    )

    test_dataset = RetinopathyDataset(
        df_test, input_dir, transforms=get_transforms("test", image_size), mode="test"
    )

    # Instantiate DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
