import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from library.config import INPUT_DIR


class DogCatDataset(Dataset):
    """
    A custom Dataset class for loading Dog and Cat images from disk.

    Attributes:
        df (pd.DataFrame): DataFrame containing image metadata.
        transform (callable, optional): Transformations to apply to the images.
        mode (str): The mode of the dataset ('train', 'val', or 'test').
    """

    def __init__(self, df: pd.DataFrame, transform=None, mode: str = "train"):
        """
        Initializes the DogCatDataset.

        Args:
            df (pd.DataFrame): DataFrame containing metadata.
                               For train/val: must contain 'filepath' and 'label'.
                               For test: must contain 'filepath' and 'id'.
            transform (callable, optional): A function/transform that takes in a PIL image
                                            and returns a transformed version.
            mode (str): One of 'train', 'val', 'test'. Defaults to 'train'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple:
                - If mode is 'train' or 'val': (image_tensor, label_tensor)
                - If mode is 'test': (image_tensor, id_val)
        """
        row = self.df.iloc[idx]

        # Construct the absolute file path
        # The metadata 'filepath' is relative to the INPUT_DIR (e.g., "train/cat.0.jpg")
        rel_path = row["filepath"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Load image using OpenCV
        # cv2.imread returns a NumPy array in BGR format
        image = cv2.imread(full_path)

        # Check for invalid images (though metadata validation should catch this)
        if image is None:
            # Create a blank black image as a fallback to prevent crashing
            # Assuming a default size of 256x256 which will likely be resized by transforms
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL Image
        # This is required because the torchvision transforms (like RandomResizedCrop)
        # in library/transforms.py are designed to work with PIL Images or Tensors.
        # Passing a numpy array directly to RandomResizedCrop can cause issues.
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return data based on mode
        if self.mode == "test":
            # For test set, return the image and the ID (for submission mapping)
            img_id = row["id"]
            # Return ID (DataLoader default_collate handles integers fine)
            return image, img_id
        else:
            # For train/val sets, return the image and the label
            label = row["label"]
            # Convert label to float32 tensor for BCEWithLogitsLoss
            # Shape is scalar, will be batched to (Batch_Size,)
            return image, torch.tensor(label, dtype=torch.float32)
