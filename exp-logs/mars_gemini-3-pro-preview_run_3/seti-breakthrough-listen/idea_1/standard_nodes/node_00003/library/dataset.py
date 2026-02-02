import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for loading SETI spectrograms from .npy files.
    """

    def __init__(self, df, input_dir=Config.INPUT_DIR, sample_size=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'id', 'target', and 'file_path'.
            input_dir (str): Root directory where the files are located.
            sample_size (int, optional): If provided, limits the dataset to this many samples
                                         for debugging/testing purposes.
        """
        self.input_dir = input_dir

        # Apply debugging limit if requested
        if sample_size is not None:
            df = df.iloc[:sample_size]

        # Pre-extract data from DataFrame to arrays for faster access in __getitem__
        # This avoids the overhead of pandas .iloc indexing during iteration
        self.file_paths = df["file_path"].values
        self.targets = df["target"].values
        self.ids = df["id"].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Construct full file path
        relative_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, relative_path)

        try:
            # Load the spectrogram
            # Native format is float16, shape (6, 273, 256)
            image = np.load(full_path)  # (6, 273, 256)

            # Vertically stack the 6 panels to form a single large image (1638, 256)
            # This preserves raw info (Cite solution_lesson_node_00002) and allows
            # the CNN to learn spatial relationships (Cite solution_lesson_node_00001).
            image = np.vstack(image)  # (1638, 256)

            # Add channel dimension: (1, 1638, 256)
            image = image[np.newaxis, ...]

            # Cast to float32 for PyTorch compatibility
            image = image.astype(np.float32)

        except Exception as e:
            # Fallback in case of file read error to prevent training crash
            # Returns a zero tensor of the expected shape
            image = np.zeros(Config.INPUT_SHAPE, dtype=np.float32)
            print(f"Warning: Error loading {full_path}: {e}")

        # Convert to PyTorch Tensor
        image_tensor = torch.from_numpy(image)

        # Get target and convert to tensor
        # Shape (1,) is standard for BCEWithLogitsLoss with binary targets
        target = self.targets[idx]
        target_tensor = torch.tensor([target], dtype=torch.float32)

        return image_tensor, target_tensor
