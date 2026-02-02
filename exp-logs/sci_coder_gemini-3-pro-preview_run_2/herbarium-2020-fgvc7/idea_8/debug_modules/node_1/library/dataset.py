import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided library
from library.utils import process_taxonomy


class HerbariumDataset(Dataset):
    """
    Dataset class for the Herbarium 2020 FGVC7 competition.
    Handles loading of images and hierarchical labels (Species, Genus, Family).
    """

    def __init__(
        self,
        csv_path,
        taxonomy_map=None,
        transform=None,
        input_dir="./input",
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
            taxonomy_map (pd.DataFrame, optional): DataFrame mapping category_id to hierarchical labels.
                                                   Required for training/validation sets with labels.
            transform (albumentations.Compose, optional): Augmentation pipeline.
            input_dir (str): Root directory of the dataset.
        """
        self.input_dir = input_dir
        self.transform = transform

        # Load metadata
        self.df = pd.read_csv(csv_path)

        # Determine if we are in training/val mode (with labels) or test mode
        # We need both the column in the CSV and the taxonomy map
        self.has_labels = "category_id" in self.df.columns and taxonomy_map is not None

        if self.has_labels:
            # Merge taxonomy information
            # taxonomy_map columns: category_id, family, genus, species_label, genus_label, family_label
            # We perform a left join to attach labels to the images
            self.df = self.df.merge(taxonomy_map, on="category_id", how="left")

            # Verify that merge didn't fail (no NaNs in labels)
            if self.df["species_label"].isnull().any():
                raise ValueError(
                    "Some category_ids in the dataset were not found in the taxonomy map."
                )

            # Pre-extract labels for faster access
            self.species_labels = self.df["species_label"].values.astype(np.int64)
            self.genus_labels = self.df["genus_label"].values.astype(np.int64)
            self.family_labels = self.df["family_label"].values.astype(np.int64)

        # Pre-extract file paths and IDs
        self.file_paths = self.df["file_path"].values
        self.image_ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resolve image path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        image = cv2.imread(full_path)

        # Handle missing or corrupt images gracefully (though dataset is verified)
        if image is None:
            # Return a black image of standard size to prevent crash
            # Assuming 300x300 as a safe default based on model specs
            image = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = ToTensorV2()(image=image)["image"]

        if self.has_labels:
            # Return image and tuple of hierarchical labels
            # Targets: (Species, Genus, Family)
            species = torch.tensor(self.species_labels[idx], dtype=torch.long)
            genus = torch.tensor(self.genus_labels[idx], dtype=torch.long)
            family = torch.tensor(self.family_labels[idx], dtype=torch.long)
            return image, (species, genus, family)
        else:
            # Return image and image_id for inference/submission
            image_id = torch.tensor(self.image_ids[idx], dtype=torch.long)
            return image, image_id


def get_transforms(img_size, mode="train"):
    """
    Generates the augmentation pipeline.

    Args:
        img_size (int): The spatial resolution (height/width) for the model.
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic resizing.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                # RandomResizedCrop: Randomly crop a portion of the image and resize it to img_size.
                # Scale (0.6, 1.0) ensures we don't crop too aggressively, preserving context.
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                # Random Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # Normalize and Convert to Tensor
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize to target size deterministically
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
