import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(split="train"):
    """
    Returns the transformation pipeline for a specific data split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        # For val and test, just convert to tensor (scales to 0-1)
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Classification task.
    """

    def __init__(self, metadata_df, transform=None, data_dir=Config.INPUT_DIR):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'id', 'has_cactus', 'file_path'.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_dir (str): Base directory for image files.
        """
        self.metadata = metadata_df
        self.transform = transform
        self.data_dir = data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.metadata.iloc[idx]

        # Construct full file path
        # Metadata contains relative paths like "train/id.jpg"
        img_path = os.path.join(self.data_dir, row["file_path"])

        # Load image
        # Open as RGB to ensure 3 channels (even if original is grayscale, though EDA showed all RGB)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        # Ensure it's a float tensor for BCEWithLogitsLoss
        # Shape will be (1,)
        label = torch.tensor([row["has_cactus"]], dtype=torch.float32)

        return image, label


def get_datasets(debug=Config.DEBUG):
    """
    Loads metadata and creates Dataset objects for train, val, and test splits.

    Args:
        debug (bool): If True, truncates datasets for debugging purposes.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Debug mode: sample a subset
    if debug:
        train_size = min(len(train_df), Config.DEBUG_SAMPLE_SIZE)
        val_size = min(len(val_df), Config.DEBUG_SAMPLE_SIZE)
        test_size = min(len(test_df), Config.DEBUG_SAMPLE_SIZE)

        train_df = train_df.iloc[:train_size]
        val_df = val_df.iloc[:val_size]
        test_df = test_df.iloc[:test_size]

    # Create Datasets
    train_ds = CactusDataset(train_df, transform=get_transforms("train"))

    val_ds = CactusDataset(val_df, transform=get_transforms("val"))

    test_ds = CactusDataset(test_df, transform=get_transforms("test"))

    return train_ds, val_ds, test_ds
