import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from library import config
from library import transforms


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Apply seed immediately upon module import
set_seed(config.SEED)


class DogDataset(Dataset):
    """
    PyTorch Dataset for the Dog Breed Classification task.
    Implements the Multi-Scale Deep Feature Pyramid strategy data loading.
    Generates Global, Standard, and Robust Local views with Test Time Augmentation (Flip).
    """

    def __init__(self, csv_path, mode="train"):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            mode (str): One of 'train', 'val', 'test'.
        """
        self.mode = mode
        self.csv_path = csv_path

        # Validate and load the metadata dataframe
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Initialize the view transformation pipelines
        # Returns a dict with keys: 'global', 'standard', 'local'
        self.view_transforms = transforms.get_view_transforms()

        # Establish Class Mapping
        # To ensure consistent label encoding across Train, Val, and Test sets,
        # we always derive the class vocabulary from the Training set.
        if os.path.exists(config.TRAIN_CSV):
            train_df = pd.read_csv(config.TRAIN_CSV)
            self.classes = sorted(train_df["breed"].unique().tolist())
        else:
            # Fallback for edge cases where train.csv might be missing
            if "breed" in self.df.columns:
                self.classes = sorted(self.df["breed"].unique().tolist())
            else:
                self.classes = []

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]

        # Construct full image path
        # Metadata contains relative path (e.g., 'train/id.jpg')
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        try:
            # Open image and ensure RGB (handles Grayscale/RGBA inputs)
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            # In a strict pipeline, we raise the error.
            # In production, one might return a dummy image, but here we fail fast.
            raise IOError(f"Failed to load image at {full_path}: {e}")

        # ---------------------------------------------------------------------
        # Test Time Augmentation (TTA): Horizontal Flip
        # ---------------------------------------------------------------------
        # We generate views for both the Original and the Flipped image.
        # Features from these will be averaged later in the pipeline.
        image_flipped = F.hflip(image)

        # ---------------------------------------------------------------------
        # View 1: Global (Shape)
        # ---------------------------------------------------------------------
        # Transform returns (3, 224, 224)
        global_t = self.view_transforms["global"]
        global_orig = global_t(image)
        global_flip = global_t(image_flipped)

        # Stack into (2, 3, 224, 224)
        global_view = torch.stack([global_orig, global_flip])

        # ---------------------------------------------------------------------
        # View 2: Standard (Context)
        # ---------------------------------------------------------------------
        # Transform returns (3, 224, 224)
        standard_t = self.view_transforms["standard"]
        standard_orig = standard_t(image)
        standard_flip = standard_t(image_flipped)

        # Stack into (2, 3, 224, 224)
        standard_view = torch.stack([standard_orig, standard_flip])

        # ---------------------------------------------------------------------
        # View 3: Robust Local (Texture)
        # ---------------------------------------------------------------------
        # Transform returns a stack of 5 crops: (5, 3, 224, 224)
        local_t = self.view_transforms["local"]
        local_orig = local_t(image)
        local_flip = local_t(image_flipped)

        # Concatenate into (10, 3, 224, 224)
        # 5 crops from original + 5 crops from flipped
        local_view = torch.cat([local_orig, local_flip], dim=0)

        # ---------------------------------------------------------------------
        # Label Processing
        # ---------------------------------------------------------------------
        label_idx = -1
        if self.mode != "test":
            if "breed" in row:
                breed = row["breed"]
                label_idx = self.class_to_idx.get(breed, -1)

        return {
            "id": img_id,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "global_view": global_view,
            "standard_view": standard_view,
            "local_view": local_view,
        }


def get_dataloader(
    csv_path, mode="train", batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
):
    """
    Factory function to create a configured DataLoader for the DogDataset.

    Args:
        csv_path (str): Path to metadata CSV.
        mode (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        torch.utils.data.DataLoader: Configured loader.
    """
    dataset = DogDataset(csv_path, mode=mode)

    # Shuffle is only necessary for the training set
    shuffle = mode == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
