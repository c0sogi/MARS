import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import library.config as config


class RetinopathyDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy detection.
    Handles image loading, pad-to-square resizing, augmentations, and ordinal label encoding.
    """

    def __init__(self, csv_path, phase="train", transform=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Albumentations transforms.
        """
        self.phase = phase
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Ensure input directory is correct
        self.input_dir = config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative to input dir (e.g., "train_images/id.png")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for potential missing/corrupt images to prevent crash
            # Create a black image of target size
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Pad-to-Square Strategy
        # Pad shorter dimension to match longer dimension to preserve aspect ratio
        h, w, c = image.shape
        if h != w:
            longest = max(h, w)
            # Calculate padding (center the image)
            top = (longest - h) // 2
            bottom = longest - h - top
            left = (longest - w) // 2
            right = longest - w - left

            # Pad with black (0)
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        # 2. Resize to target size
        # Explicit resize ensures dimensions match model input
        if image.shape[0] != config.IMAGE_SIZE or image.shape[1] != config.IMAGE_SIZE:
            image = cv2.resize(
                image,
                (config.IMAGE_SIZE, config.IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        # 3. Apply Augmentations and Normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            default_tf = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            image = default_tf(image=image)["image"]

        # 4. Handle Targets
        if self.phase == "test":
            # No targets for test set
            return {"image": image, "id_code": row["id_code"]}
        else:
            label = row["diagnosis"]

            # Create Ordinal Label Vector for Rank-Consistent Regression
            # Size: NUM_ORDINAL_OUTPUTS (e.g., 4 outputs for 5 classes)
            # Label k -> first k elements are 1, rest 0
            # Example: Label 2 -> [1, 1, 0, 0]
            ordinal_label = np.zeros(config.NUM_ORDINAL_OUTPUTS, dtype=np.float32)
            if label > 0:
                # Set the first 'label' indices to 1
                # e.g. if label=2, indices 0 and 1 become 1.
                ordinal_label[:label] = 1.0

            return {
                "image": image,
                "label": torch.tensor(ordinal_label, dtype=torch.float32),
                "target": torch.tensor(
                    label, dtype=torch.long
                ),  # Original int label for metrics
                "id_code": row["id_code"],
            }


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Normalization constants (ImageNet standards)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalization & Tensor conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Normalize
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])


def create_dataloaders(train_csv, val_csv, test_csv, batch_size=config.BATCH_SIZE):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        train_csv (str): Path to train metadata.
        val_csv (str): Path to val metadata.
        test_csv (str): Path to test metadata.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define Transforms
    train_tf = get_transforms("train")
    val_tf = get_transforms("val")
    test_tf = get_transforms("test")

    # Instantiate Datasets
    train_ds = RetinopathyDataset(train_csv, phase="train", transform=train_tf)
    val_ds = RetinopathyDataset(val_csv, phase="val", transform=val_tf)
    test_ds = RetinopathyDataset(test_csv, phase="test", transform=test_tf)

    # Instantiate Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches during training
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
