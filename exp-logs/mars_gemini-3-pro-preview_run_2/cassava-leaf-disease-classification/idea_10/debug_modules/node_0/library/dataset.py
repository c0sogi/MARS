import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config


class CassavaDataset(Dataset):
    """
    Custom Dataset for Cassava Leaf Disease Classification.
    Handles loading of images from disk and applying transformations.
    """

    def __init__(self, df: pd.DataFrame, transform=None, return_id: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transform (callable, optional): Optional transform to be applied on a sample.
            return_id (bool): If True, returns (image, label, image_id). Otherwise (image, label).
        """
        self.df = df
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train_images/123.jpg")
        # Config.INPUT_DIR is "./input"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using PIL as required
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load image at {img_path}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        # Ensure label is a long tensor for CrossEntropyLoss
        label = torch.tensor(row["label"], dtype=torch.long)

        if self.return_id:
            return image, label, row["image_id"]

        return image, label


def get_dataset(
    phase: str, transform=None, debug: bool = False, return_id: bool = False
) -> CassavaDataset:
    """
    Factory function to create a CassavaDataset instance for a specific phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        transform (callable, optional): Transform pipeline.
        debug (bool): If True, limits the dataset size to a small subset for debugging.
        return_id (bool): Whether to return image IDs along with tensors.

    Returns:
        CassavaDataset: The configured dataset instance.
    """
    phase = phase.lower()

    # Select metadata file based on phase
    if phase == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif phase in ["val", "valid", "validation"]:
        csv_path = Config.VAL_METADATA_PATH
    elif phase in ["test", "inference"]:
        csv_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown phase: {phase}. Expected 'train', 'val', or 'test'.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    # Load metadata
    df = pd.read_csv(csv_path)

    # Debugging: Limit dataset size
    if debug:
        # Use a small subset (e.g., 100 samples) to verify pipeline
        df = df.head(100).copy()

    dataset = CassavaDataset(df, transform=transform, return_id=return_id)
    return dataset
