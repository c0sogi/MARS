import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HerbariumDataset(Dataset):
    def __init__(self, df, root_dir, taxonomy_maps=None, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            root_dir (str): Root directory for images (usually './input').
            taxonomy_maps (tuple): (species_to_idx, species_to_family, species_to_order).
                                   Required for training/validation.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

        if not self.is_test and taxonomy_maps:
            self.species_to_idx, self.species_to_family, self.species_to_order = (
                taxonomy_maps
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the row
        row = self.df.iloc[idx]

        # Construct full image path
        # The file_path in metadata is relative to input dir (e.g., 'train/images/...')
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)

        # Handle potential missing images or read errors
        if image is None:
            # Return a black image of standard size (224x224) to avoid crashing
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # For test set, return image and image_id (for submission)
            return image, row["image_id"]
        else:
            # For train/val, retrieve labels
            cat_id = row["category_id"]

            # Map original category_id to internal indices for the 3 heads
            species_label = self.species_to_idx[cat_id]
            family_label = self.species_to_family[cat_id]
            order_label = self.species_to_order[cat_id]

            return image, species_label, family_label, order_label
