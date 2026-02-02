import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed


class CatDogDataset(Dataset):
    """
    Custom Dataset for loading Dog vs Cat images.
    """

    def __init__(self, df, input_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            input_dir (str): Path to the input directory containing images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Flag to indicate if this is the test set (returns ID instead of label).
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.input_dir, row["filepath"])

        # Read image using OpenCV (loads as BGR)
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Image not found at {filepath}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and ID for test set
            img_id = row["id"]
            return image, torch.tensor(img_id, dtype=torch.long)
        else:
            # Return image and label for train/val set
            label = row["label"]
            # Float32 is required for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the data transformation pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Standard ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        # Stronger augmentation for training (Cite solution_lesson_node_00002)
        transform_list = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        # Standard evaluation preprocessing
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
        ]

    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return transforms.Compose(transform_list)


def create_dataloaders(
    batch_size=32,
    num_workers=8,
    input_dir="./input",
    metadata_dir="./metadata",
    max_samples=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of subprocesses for data loading.
        input_dir (str): Base directory for images.
        metadata_dir (str): Directory containing metadata CSVs.
        max_samples (int, optional): Limit the number of samples for debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Ensure reproducibility
    set_seed(42)

    # Load metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Optional debugging: subset data
    if max_samples is not None:
        train_df = train_df.iloc[:max_samples]
        val_df = val_df.iloc[:max_samples]
        test_df = test_df.iloc[:max_samples]

    # Create datasets
    train_dataset = CatDogDataset(
        train_df, input_dir, transform=get_transforms("train"), is_test=False
    )

    val_dataset = CatDogDataset(
        val_df, input_dir, transform=get_transforms("val"), is_test=False
    )

    test_dataset = CatDogDataset(
        test_df, input_dir, transform=get_transforms("test"), is_test=True
    )

    # Create dataloaders
    # Pin memory improves transfer speed to GPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
