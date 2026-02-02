import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import pad_image


class SETIDataset(Dataset):
    """
    Custom Dataset for SETI Signal Detection.
    Loads .npy spectrograms, pads them, splits into On/Off target streams,
    and applies synchronized augmentations.
    """

    def __init__(self, metadata_path: str, mode: str = "train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            mode (str): Operation mode - 'train', 'val', or 'test'.
                        'train' enables data augmentation.
        """
        self.metadata_path = metadata_path
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        self.df = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # row['file_path'] is relative, e.g., "train/0/00042890562ff68.npy"
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load the spectrogram
        # Shape: (6, 273, 256) -> (Cadence_Pos, Frequency, Time)
        try:
            image = np.load(file_path).astype(np.float32)
        except Exception as e:
            # Fallback for corrupt/missing files (though unlikely given verification)
            # Return zeros to prevent crashing
            image = np.zeros(
                (6, Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.float32
            )

        # Pad image to (6, 288, 256) using the provided utility
        # This pads the frequency dimension (axis 1)
        image = pad_image(image)

        # Split into On-Target (Signal) and Off-Target (Reference) streams
        # On-Target: A, C, E -> Indices 0, 2, 4
        # Off-Target: B, D, F -> Indices 1, 3, 5
        on_target = image[[0, 2, 4], :, :]
        off_target = image[[1, 3, 5], :, :]

        # Apply Synchronized Augmentations if in training mode
        if self.mode == "train":
            # Horizontal Flip (Time Reversal) - Axis 2 (Width)
            if np.random.rand() < 0.5:
                # Must use .copy() to avoid negative stride issues in PyTorch
                on_target = np.flip(on_target, axis=2).copy()
                off_target = np.flip(off_target, axis=2).copy()

            # Vertical Flip (Frequency Inversion) - Axis 1 (Height)
            if np.random.rand() < 0.5:
                on_target = np.flip(on_target, axis=1).copy()
                off_target = np.flip(off_target, axis=1).copy()

        # Convert to PyTorch Tensors
        on_target_tensor = torch.from_numpy(on_target)
        off_target_tensor = torch.from_numpy(off_target)

        # Get Target
        target = row["target"]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        # Return format: (inputs, target)
        # Inputs is a tuple of (Stream A, Stream B) for the Siamese Network
        return (on_target_tensor, off_target_tensor), target_tensor
