import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from library.config import Config


class CassavaDataset(Dataset):
    """
    PyTorch Dataset for Cassava Leaf Disease Classification.
    Handles loading images via PIL and applying transformations.
    """

    def __init__(self, df: pd.DataFrame, transform=None, output_label=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image_id, label, file_path).
            transform (callable, optional): Transformation pipeline to apply to the image.
            output_label (bool): Whether to return the label along with the image.
                                 Set to False for inference if labels are not needed.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, label) if output_label is True, else (image)
        """
        row = self.df.iloc[index]

        # Construct the full path to the image
        # The metadata 'file_path' is relative to the input directory (e.g., "train_images/xyz.jpg")
        image_path = os.path.join(Config.input_dir, row["file_path"])

        # Load image using PIL (strictly enforcing native loading)
        # Convert to RGB to ensure consistency (3 channels)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # In case of a read error, print and raise, or handle gracefully.
            # Given the verified metadata, this should be rare.
            raise IOError(f"Error loading image at {image_path}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return image and label if requested
        if self.output_label:
            # Ensure label is a long tensor for CrossEntropyLoss
            label = torch.tensor(row["label"], dtype=torch.long)
            return image, label
        else:
            return image


def load_dataset_dataframe(
    csv_path: str, debug: bool = False, debug_size: int = 100
) -> pd.DataFrame:
    """
    Helper function to load the metadata DataFrame from a CSV file.
    Supports a debug mode to load a small subset of the data.

    Args:
        csv_path (str): Path to the CSV file.
        debug (bool): If True, returns a subsample of the dataframe.
        debug_size (int): The number of samples to return in debug mode.

    Returns:
        pd.DataFrame: The loaded (and potentially subsampled) DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if debug:
        # Use fixed seed for reproducibility during debugging
        sample_n = min(len(df), debug_size)
        df = df.sample(n=sample_n, random_state=Config.seed).reset_index(drop=True)

    return df
