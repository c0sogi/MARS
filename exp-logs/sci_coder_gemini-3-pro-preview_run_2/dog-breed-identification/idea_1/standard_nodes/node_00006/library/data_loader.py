import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ConvNeXt_Large_Weights
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IMG_SIZE,
    MEAN,
    STD,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import set_seed


class DogDataset(Dataset):
    """
    Custom Dataset for loading Dog images.
    """

    def __init__(
        self,
        metadata_df,
        transform=None,
        class_to_idx=None,
        is_test=False,
        input_dir=INPUT_DIR,
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing image paths and labels/IDs.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from class name to integer index. Required if is_test is False.
            is_test (bool): Flag to indicate if this is the test set (returns ID instead of label).
            input_dir (str): Root directory for images.
        """
        self.df = metadata_df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.is_test = is_test
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like 'train/id.jpg'
        rel_path = row["file_path"]
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            # For test set, return image and ID
            return img, row["id"]
        else:
            # For train/val sets, return image and label index
            label_name = row["breed"]
            label_idx = self.class_to_idx[label_name]
            return img, label_idx


def get_transforms(img_size=IMG_SIZE, mean=MEAN, std=STD):
    """
    Returns the composition of transforms for the dataset.
    Uses the specific transforms defined by the pre-trained model weights.
    """
    # ConvNeXt-Large weights provide their own optimal transform pipeline
    # We prepend ToPILImage because the dataset loads images as numpy arrays (OpenCV)
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            ConvNeXt_Large_Weights.DEFAULT.transforms(),
        ]
    )


def create_dataloaders(
    train_path=TRAIN_METADATA_PATH,
    val_path=VAL_METADATA_PATH,
    test_path=TEST_METADATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    debug_limit=None,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_path (str): Path to training metadata CSV.
        val_path (str): Path to validation metadata CSV.
        test_path (str): Path to test metadata CSV.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes.
        debug_limit (int, optional): Limit the number of samples for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
    """
    # Set seed for reproducibility
    set_seed(SEED)

    # Load Metadata
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Determine Classes from the full training set (before any debug limiting)
    # This ensures class mapping remains consistent even if we subsample
    classes = sorted(train_df["breed"].unique().tolist())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Apply Debug Limit if specified
    if debug_limit is not None:
        train_df = train_df.iloc[:debug_limit]
        val_df = val_df.iloc[:debug_limit]
        test_df = test_df.iloc[:debug_limit]

    # Get Transforms
    transform = get_transforms(img_size=IMG_SIZE, mean=MEAN, std=STD)

    # Create Datasets
    train_ds = DogDataset(
        train_df, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    val_ds = DogDataset(
        val_df, transform=transform, class_to_idx=class_to_idx, is_test=False
    )

    test_ds = DogDataset(test_df, transform=transform, class_to_idx=None, is_test=True)

    # Create DataLoaders
    # Pin memory speeds up host to device transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes
