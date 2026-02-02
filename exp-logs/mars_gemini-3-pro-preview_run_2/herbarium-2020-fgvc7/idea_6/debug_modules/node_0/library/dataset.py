import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the provided taxonomy utility
from library.taxonomy_utils import build_taxonomy_mapping

# Dataset-specific statistics calculated during Data Analysis
# These values (approx 0.75 mean) indicate the white background nature of herbarium sheets.
MEAN = [0.771, 0.755, 0.720]
STD = [0.300, 0.305, 0.309]


def get_transforms(split, image_size):
    """
    Returns the image transformations for a specific data split and resolution.

    Args:
        split (str): One of 'train', 'val', 'test'.
        image_size (int): The target height and width of the image.

    Returns:
        A.Compose: The albumentations composition of transforms.
    """
    if split == "train":
        return A.Compose(
            [
                # RandomResizedCrop is excellent for learning scale-invariant features
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.6, 1.0)
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and test, we simply resize to the target dimension.
        # CenterCrop is skipped to ensure the entire specimen is visible,
        # as herbarium sheets often fill the frame.
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )


class HerbariumDataset(Dataset):
    """
    PyTorch Dataset for the Herbarium 2020 FGVC7 Competition.
    Handles loading images and hierarchical labels (Species, Genus, Family).
    """

    def __init__(
        self,
        csv_path,
        taxonomy_map=None,
        transform=None,
        is_test=False,
        input_root="./input",
        debug_size=None,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            taxonomy_map (pd.DataFrame, optional): DataFrame mapping category_id to genus/family IDs.
                                                   Required for train/val splits.
            transform (A.Compose, optional): Albumentations transforms to apply.
            is_test (bool): Set to True for the test set (suppresses label loading).
            input_root (str): Root directory where the 'nybg2020' folder is located.
            debug_size (int, optional): If provided, limits the dataset to this many samples for debugging.
        """
        self.transform = transform
        self.is_test = is_test
        self.input_root = input_root

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Optional debugging subset
        if debug_size is not None:
            df = df.head(debug_size)

        self.image_ids = df["image_id"].values
        self.file_paths = df["file_path"].values

        # Load labels if not in test mode
        if not self.is_test:
            if taxonomy_map is None:
                raise ValueError(
                    "taxonomy_map must be provided for training and validation sets."
                )

            if "category_id" not in df.columns:
                raise ValueError(f"Column 'category_id' missing in {csv_path}")

            self.category_ids = df["category_id"].values

            # Create fast lookup dictionaries from the taxonomy dataframe
            # taxonomy_map index is 'category_id'
            genus_lookup = taxonomy_map["genus_id"].to_dict()
            family_lookup = taxonomy_map["family_id"].to_dict()

            # Map species (category_id) to genus and family
            # Using list comprehension for speed over pandas apply
            self.genus_ids = np.array(
                [genus_lookup.get(cat_id, -1) for cat_id in self.category_ids],
                dtype=np.int64,
            )
            self.family_ids = np.array(
                [family_lookup.get(cat_id, -1) for cat_id in self.category_ids],
                dtype=np.int64,
            )

            # Validation check
            if np.any(self.genus_ids == -1) or np.any(self.family_ids == -1):
                print(
                    f"Warning: Found {np.sum(self.genus_ids == -1)} samples with missing taxonomy mapping in {csv_path}."
                )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        # file_path is relative, e.g., "nybg2020/train/..."
        full_path = os.path.join(self.input_root, self.file_paths[idx])

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for corrupt/missing images to prevent crashing
            # Create a blank image with noise to avoid NaN gradients
            image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 3. Return Data
        if self.is_test:
            # For test set, return image and image_id (needed for submission)
            return image, self.image_ids[idx]
        else:
            # For train/val, return image and dictionary of hierarchical labels
            targets = {
                "species": torch.tensor(self.category_ids[idx], dtype=torch.long),
                "genus": torch.tensor(self.genus_ids[idx], dtype=torch.long),
                "family": torch.tensor(self.family_ids[idx], dtype=torch.long),
            }
            return image, targets
