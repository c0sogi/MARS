import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import load_hierarchy_mappings


class PlantDataset(Dataset):
    def __init__(
        self, df, transform=None, root_dir="./input", hierarchy=None, is_test=False
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame with metadata.
            transform (callable, optional): Albumentations transforms.
            root_dir (str): Root directory for images.
            hierarchy (dict, optional): Mapping dictionaries for genus and family.
            is_test (bool): Whether this is the test set (returns image_id instead of labels).
        """
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.hierarchy = hierarchy
        self.is_test = is_test

        if not self.is_test and self.hierarchy:
            self.species_to_genus = self.hierarchy["species_to_genus"]
            self.species_to_family = self.hierarchy["species_to_family"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Handle potential missing images by returning a blank image
            # (Should not happen given metadata validation, but good for robustness)
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission mapping
            return image, str(row["image_id"])
        else:
            # Return image and hierarchical labels
            species_id = int(row["category_id"])

            # Map species to genus and family
            # Default to -1 or 0 if mapping fails (though mapping should be complete)
            genus_id = self.species_to_genus.get(species_id, 0)
            family_id = self.species_to_family.get(species_id, 0)

            return (
                image,
                torch.tensor(species_id, dtype=torch.long),
                torch.tensor(genus_id, dtype=torch.long),
                torch.tensor(family_id, dtype=torch.long),
            )


def get_transforms(data_type, image_size=256):
    """
    Returns Albumentations transforms based on data type.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.8, 1.0)
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Valid and Test
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_batch_size=32,
    val_batch_size=64,
    image_size=256,
    num_workers=2,
    sample_size=None,
    cache_dir="./working/idea_8",
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation/inference.
        image_size (int): Input image resolution.
        num_workers (int): Number of subprocesses for data loading.
        sample_size (int, optional): If set, limits dataset size for debugging.
        cache_dir (str): Directory to cache hierarchy mappings.

    Returns:
        tuple: (train_loader, val_loader, test_loader, hierarchy_dict)
    """
    # 1. Load Hierarchy Mappings
    hierarchy = load_hierarchy_mappings(cache_dir=cache_dir)

    # 2. Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # 3. Apply Sampling (for debugging)
    if sample_size is not None:
        train_df = train_df.head(sample_size)
        val_df = val_df.head(sample_size)
        test_df = test_df.head(sample_size)

    # 4. Define Transforms
    train_transforms = get_transforms("train", image_size)
    val_transforms = get_transforms("valid", image_size)

    # 5. Create Datasets
    train_dataset = PlantDataset(
        train_df, transform=train_transforms, hierarchy=hierarchy, is_test=False
    )

    val_dataset = PlantDataset(
        val_df, transform=val_transforms, hierarchy=hierarchy, is_test=False
    )

    test_dataset = PlantDataset(
        test_df, transform=val_transforms, hierarchy=None, is_test=True
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, hierarchy
