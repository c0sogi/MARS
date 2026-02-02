import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from library.config import Config


class DogBreedDataset(Dataset):
    """
    A PyTorch Dataset for the Dog Breed Classification task.

    It implements the Multi-View strategy by generating three distinct geometric views
    (Standard, Global, Local) for each image on the fly. This allows the heterogeneous
    ensemble to leverage different visual signals (balance, shape context, texture detail).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: dict,
        input_dir: str = Config.INPUT_DIR,
        class_to_idx: dict = None,
        is_test: bool = False,
        debug_subset_size: int = None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels (for train/val).
            transforms (dict): Dictionary containing 'standard', 'global', and 'local' transformation pipelines.
            input_dir (str): Root directory where images are stored.
            class_to_idx (dict, optional): Mapping from breed name to integer index. Required if is_test is False.
            is_test (bool): Flag indicating if this is the test set (no labels).
            debug_subset_size (int, optional): If provided, limits the dataset to this many samples for debugging.
        """
        self.df = df.copy()
        self.transforms = transforms
        self.input_dir = input_dir
        self.class_to_idx = class_to_idx
        self.is_test = is_test

        # Debugging: Subset the data if requested to speed up development cycles
        if debug_subset_size is not None and debug_subset_size > 0:
            if debug_subset_size < len(self.df):
                self.df = self.df.iloc[:debug_subset_size].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract metadata
        image_id = row["id"]
        rel_path = row["file_path"]

        # Construct full image path
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image (RGB)
        # We use PIL as it is the standard backend for torchvision transforms
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except (OSError, IOError) as e:
            raise IOError(f"Failed to load image at {img_path}: {e}")

        # Apply Multi-View Transforms
        # The transforms dictionary is expected to have keys: 'standard', 'global', 'local'
        # These correspond to the geometric strategies defined in library.transforms
        try:
            view_standard = self.transforms["standard"](image)
            view_global = self.transforms["global"](image)
            view_local = self.transforms["local"](image)
        except KeyError as e:
            raise KeyError(f"Transforms dictionary missing required key: {e}")

        # Construct the sample dictionary
        sample = {
            "id": image_id,
            "standard": view_standard,
            "global": view_global,
            "local": view_local,
        }

        # Process Label if not test set
        if not self.is_test:
            breed = row["breed"]
            if self.class_to_idx is None:
                raise ValueError(
                    "class_to_idx must be provided for training/validation sets."
                )

            label_idx = self.class_to_idx.get(breed)
            if label_idx is None:
                raise ValueError(f"Breed '{breed}' not found in class_to_idx mapping.")

            sample["label"] = torch.tensor(label_idx, dtype=torch.long)

        return sample
