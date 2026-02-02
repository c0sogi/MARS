import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.taxonomy import get_taxonomy_mappings


def get_transforms(data_split, image_size):
    """
    Returns the Albumentations transformation pipeline based on the data split
    and the target image size (supporting progressive resizing).

    Args:
        data_split (str): 'train', 'val', or 'test'.
        image_size (int): Target spatial dimension (e.g., 224 or 300).

    Returns:
        A.Compose: The composed augmentation pipeline.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data_split == "train":
        return A.Compose(
            [
                # RandomResizedCrop is standard for training CNNs to handle scale invariance
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                # Add slight color jitter to improve robustness
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we want deterministic resizing
        # We resize the short edge to image_size and then center crop,
        # or just resize directly. Direct resizing is often sufficient and faster.
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class HerbariumDataset(Dataset):
    def __init__(self, split, image_size, transform=None, debug=None):
        """
        PyTorch Dataset for the Herbarium 2020 FGVC7 task.

        Args:
            split (str): One of 'train', 'val', 'test'.
            image_size (int): The input resolution for the model.
            transform (A.Compose, optional): Albumentations transform pipeline.
                                             If None, generated via get_transforms.
            debug (bool): If True, limits dataset to Config.DEBUG_SUBSET_SIZE.
        """
        self.split = split
        self.image_size = image_size
        self.debug = debug if debug is not None else Config.DEBUG
        self.input_dir = Config.INPUT_DIR

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Handle Debug Mode
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # Merge Taxonomy info for Train/Val
        self.has_labels = split != "test"
        if self.has_labels:
            # Load taxonomy mapping (cached)
            taxonomy_df = get_taxonomy_mappings(load_cached_data=True)

            # Merge on category_id to get family_id and genus_id
            # The metadata CSVs have 'category_id', taxonomy_df has 'category_id', 'family_id', 'genus_id'
            self.df = self.df.merge(
                taxonomy_df[["category_id", "family_id", "genus_id", "species_id"]],
                on="category_id",
                how="left",
            )

            # Extract label arrays for fast access
            self.species_ids = self.df["species_id"].values
            self.genus_ids = self.df["genus_id"].values
            self.family_ids = self.df["family_id"].values

        # File paths and Image IDs
        self.file_paths = self.df["file_path"].values
        self.image_ids = self.df["image_id"].values

        # Setup Transforms
        if transform is None:
            self.transform = get_transforms(split, image_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # file_paths are relative, e.g., "nybg2020/train/..."
        file_path = os.path.join(self.input_dir, self.file_paths[idx])

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images (though verification script showed none)
            # Create a black image to prevent crashing
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare Output
        sample = {"image": image, "image_id": self.image_ids[idx]}

        if self.has_labels:
            sample["species_id"] = torch.tensor(self.species_ids[idx], dtype=torch.long)
            sample["genus_id"] = torch.tensor(self.genus_ids[idx], dtype=torch.long)
            sample["family_id"] = torch.tensor(self.family_ids[idx], dtype=torch.long)

        return sample
