import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from library.config import Config


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pawpularity Contest.

    This dataset handles the loading of pet images and their associated metadata/targets
    from the structured CSV files located in the metadata directory. It provides
    an interface compatible with PyTorch DataLoaders and Hugging Face image processors.
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str = Config.INPUT_DIR,
        transform=None,
        debug: bool = False,
        debug_size: int = 100,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (e.g., ./metadata/train.csv).
            root_dir (str): Root directory containing the raw images (usually ./input).
            transform (callable, optional): Optional transform/processor to be applied on the image.
                                            This can be a torchvision transform or a Hugging Face
                                            AutoImageProcessor.
            debug (bool): If True, limits the dataset size for debugging purposes.
            debug_size (int): The number of samples to load when debug is True.
        """
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.transform = transform
        self.debug = debug

        # Load metadata dataframe
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        # Handle debug mode
        if self.debug:
            df = df.head(debug_size).copy()

        # Store IDs
        self.ids = df[Config.ID_COL].values

        # Store relative file paths
        self.file_paths = df[Config.FILE_PATH_COL].values

        # Store binary metadata features
        # Ensure they are float32 for compatibility with neural networks/regressors
        self.meta_features = df[Config.META_FEATURES].values.astype(np.float32)

        # Store targets if available (Train/Val sets)
        if Config.TARGET_COL in df.columns:
            self.targets = df[Config.TARGET_COL].values.astype(np.float32)
        else:
            # For test set, targets might not exist. Fill with dummy values or None.
            # Using -1.0 as a placeholder for missing targets.
            self.targets = np.full(len(df), -1.0, dtype=np.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'id': The Pet Profile ID.
                - 'image': The processed image (PIL or Tensor depending on transform).
                - 'meta': Tensor of binary metadata features.
                - 'target': Tensor containing the Pawpularity score.
        """
        # Construct full image path
        # The CSV contains relative paths like "train/0007de...jpg"
        # We join this with the root input directory.
        img_rel_path = self.file_paths[idx]
        img_full_path = os.path.join(self.root_dir, img_rel_path)

        # Load Image
        try:
            # Open image and convert to RGB to handle grayscale or RGBA images consistently
            image = Image.open(img_full_path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            # In a production pipeline, we might log this or skip.
            # Here we raise an error as data integrity is assumed verified.
            raise FileNotFoundError(f"Failed to load image: {img_full_path}") from e

        # Retrieve Metadata and Target
        meta = self.meta_features[idx]
        target = self.targets[idx]

        # Apply Transform (e.g., AutoImageProcessor)
        if self.transform:
            # The transform is expected to handle the PIL Image
            image = self.transform(image)

        # Return dictionary
        return {
            "id": self.ids[idx],
            "image": image,
            "meta": torch.tensor(meta, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }
