import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_transforms, get_hierarchy_mappings


class PlantDataset(Dataset):
    """
    PyTorch Dataset for Hierarchical Plant Classification.

    Handles loading images, applying transformations, and generating
    hierarchical targets (Species, Genus, Family) for the multi-task model.
    """

    def __init__(self, df, mode, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (image_id, file_path, [category_id]).
            mode (str): 'train', 'valid', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Handle Debugging: Slice dataset if DEBUG is enabled in Config
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Load hierarchy mappings for training/validation
        if self.mode in ["train", "valid"]:
            # Retrieve mappings from utils (cached)
            self.species_to_genus, self.species_to_family, _, _ = (
                get_hierarchy_mappings(load_cached_data=True)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input directory (e.g., "train_images/...")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing/corrupt images (though metadata generation checks this)
            # Create a black image to prevent crashing
            image = np.zeros(
                (Config.STAGE_1_RES, Config.STAGE_1_RES, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return logic based on mode
        if self.mode == "test":
            # For test, return image and image_id for submission
            return image, row["image_id"]
        else:
            # For train/valid, return image and dictionary of hierarchical labels
            species_label = row["category_id"]

            # Map species to genus and family
            # Ensure labels are longs for CrossEntropyLoss
            targets = {
                "species": torch.tensor(species_label, dtype=torch.long),
                "genus": torch.tensor(
                    self.species_to_genus[species_label], dtype=torch.long
                ),
                "family": torch.tensor(
                    self.species_to_family[species_label], dtype=torch.long
                ),
            }

            return image, targets


def get_dataloader(
    df, mode, batch_size, image_size, shuffle=False, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create a DataLoader.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        mode (str): 'train', 'valid', or 'test'.
        batch_size (int): Batch size.
        image_size (int): Resolution for transforms.
        shuffle (bool): Whether to shuffle data.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    transforms = get_transforms(data=mode, image_size=image_size)
    dataset = PlantDataset(df=df, mode=mode, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train"),  # Drop last incomplete batch during training
    )

    return loader
