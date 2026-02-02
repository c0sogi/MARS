import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import get_class_mappings
from library.taxonomy_utils import TaxonomyProcessor


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformations for the specified data type.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    img_size = Config.IMG_SIZE

    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if data_type == "train":
        return A.Compose(
            [
                # Conservative Augmentation: Scale 0.8-1.0 prevents aggressive cropping
                # which can remove discriminative features for fine-grained classification.
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize to target size deterministically
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class PlantDataset(Dataset):
    """
    PyTorch Dataset for Plant Classification.
    Handles loading images and providing hierarchical labels (Species, Genus, Family).
    """

    def __init__(self, csv_file, root_dir, transform=None, mode="train", debug=False):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            root_dir (str): Root directory containing image folders (e.g., ./input).
            transform (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Determines returned data.
            debug (bool): If True, samples a small subset for debugging.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        # Load metadata
        self.df = pd.read_csv(csv_file)

        # Handle debugging (sample subset)
        if debug:
            sample_size = min(len(self.df), Config().SAMPLE_SIZE or 5000)
            self.df = self.df.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)

        # Prepare labels for training/validation
        if self.mode in ["train", "val"]:
            self._prepare_labels()

    def _prepare_labels(self):
        """
        Prepares species, genus, and family indices for the dataset.
        """
        # 1. Get Species Mapping (Category ID -> Model Index)
        self.class_to_idx, _ = get_class_mappings(load_cached_data=True)

        # 2. Get Taxonomy Mappings (Species Index -> Genus/Family Index)
        # We initialize the processor which loads/computes the maps
        tax_processor = TaxonomyProcessor(load_cached_data=True)
        self.species_to_genus_map, self.species_to_family_map = tax_processor.get_maps()

        # 3. Pre-compute indices for the dataframe to speed up __getitem__
        # We map the 'category_id' column to 'species_idx'
        # Rows with unknown categories (shouldn't happen in clean data) will raise error
        self.species_indices = self.df["category_id"].map(self.class_to_idx).values

        # Ensure mapping was successful
        if np.isnan(self.species_indices).any():
            raise ValueError(
                "Some category_ids in the CSV do not exist in the class mappings."
            )

        self.species_indices = self.species_indices.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            If mode == 'test':
                image (Tensor), image_id (str)
            If mode == 'train' or 'val':
                image (Tensor), species_label (int), genus_label (int), family_label (int)
        """
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in CSV is relative to input directory (e.g., train_images/...)
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for robustness, though data should be clean
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            return image, str(row["image_id"])
        else:
            # Retrieve hierarchical labels
            species_idx = self.species_indices[idx]

            # Look up parent taxa using the mapped arrays
            # These arrays are indexed by species_idx
            genus_idx = self.species_to_genus_map[species_idx]
            family_idx = self.species_to_family_map[species_idx]

            return (
                image,
                torch.tensor(species_idx, dtype=torch.long),
                torch.tensor(genus_idx, dtype=torch.long),
                torch.tensor(family_idx, dtype=torch.long),
            )
