import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class SETIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading SETI spectrograms.
    Handles loading from .npy files, type casting, and instance normalization.
    """

    def __init__(self, metadata_path, input_dir=Config.INPUT_DIR, transform=None):
        """
        Args:
            metadata_path (str): Path to the CSV file containing metadata (id, target, file_path).
            input_dir (str): Root directory where input files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Steps:
        1. Load .npy file
        2. Cast to float32
        3. Apply Instance Normalization (Zero Mean, Unit Variance per sample)
        4. Convert to Tensor
        """
        # Get row from metadata
        row = self.metadata.iloc[idx]

        # Construct full file path
        # Metadata file_path is relative (e.g., "train/0/xxxx.npy")
        full_path = os.path.join(self.input_dir, row["file_path"])

        # Load numpy array
        # Shape: (6, 273, 256)
        # Default format is float16, we need to handle potential loading errors gracefully if needed,
        # but we assume data integrity based on metadata validation.
        try:
            image = np.load(full_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Spectrogram file not found at: {full_path}")

        # Cast to float32 for training stability
        image = image.astype(np.float32)

        # Instance Normalization
        # We normalize each spectrogram independently to have mean=0 and std=1.
        # This is critical as absolute power levels vary between observations.
        mean = np.mean(image)
        std = np.std(image)

        # Epsilon to prevent division by zero
        eps = 1e-6
        image = (image - mean) / (std + eps)

        # Vertically stack the 6 panels to preserve Doppler drift geometry (Cite solution_lesson_node_00004)
        # Shape becomes (1638, 256)
        image = np.vstack(image)

        # Add channel dimension: (1, 1638, 256)
        image = image[np.newaxis, :, :]

        # Convert to PyTorch Tensor
        image_tensor = torch.from_numpy(image)

        # Retrieve target
        # For test set, this will be a placeholder (e.g., 0.5)
        target = row["target"]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Apply external transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, target_tensor
