import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from library.config import Config
from library.utils import load_and_preprocess_image


def get_transforms(mode="train"):
    """
    Returns a composition of transforms for data augmentation and normalization.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic transforms.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    transforms_list = []

    if mode == "train":
        # Augmentations applied on the tensor (C, H, W)
        transforms_list.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    # Normalization using standard ImageNet mean and std
    # Input tensors from utils are [0, 1], so this is appropriate.
    transforms_list.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return T.Compose(transforms_list)


class WhalePairsDataset(Dataset):
    """
    Dataset for training a Siamese Network.
    Generates pairs of images:
    - Positive pairs: (Anchor, Same Id)
    - Negative pairs: (Anchor, Different Id)
    """

    def __init__(self, csv_file, subset_size=None, transform=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            subset_size (int, optional): Limit dataset size for debugging.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        # 1. Identify Anchors: Must be known whales (not new_whale)
        self.known_whales = self.df[self.df["Id"] != "new_whale"].reset_index(drop=True)

        if subset_size is not None:
            self.known_whales = self.known_whales.head(subset_size)

        # 2. Group all data by Id for efficient sampling
        # We include 'new_whale' here because they can serve as negative pairs
        self.all_groups = self.df.groupby("Id")["file_path"].apply(list).to_dict()
        self.all_ids = list(self.all_groups.keys())

    def __len__(self):
        return len(self.known_whales)

    def __getitem__(self, idx):
        # 1. Select Anchor
        row = self.known_whales.iloc[idx]
        anchor_id = row["Id"]
        anchor_path_rel = row["file_path"]
        anchor_full_path = os.path.join(Config.INPUT_DIR, anchor_path_rel)

        # 2. Determine Pair Type (50% Positive, 50% Negative)
        # We use a float check for randomness
        is_positive = np.random.random() < 0.5

        if is_positive:
            target = torch.tensor(1.0, dtype=torch.float32)
            # Pick a candidate from the same ID
            candidates = self.all_groups[anchor_id]
            # If singleton, this will pick the same image (augmentation handles variance)
            pair_path_rel = np.random.choice(candidates)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)
            # Pick a candidate from a different ID
            # This loop ensures we don't accidentally pick the same ID
            while True:
                neg_id = np.random.choice(self.all_ids)
                if neg_id != anchor_id:
                    break
            candidates = self.all_groups[neg_id]
            pair_path_rel = np.random.choice(candidates)

        pair_full_path = os.path.join(Config.INPUT_DIR, pair_path_rel)

        # 3. Load Images
        img1 = load_and_preprocess_image(
            anchor_full_path, height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH
        )
        img2 = load_and_preprocess_image(
            pair_full_path, height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH
        )

        # 4. Apply Transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, target


class WhaleInferenceDataset(Dataset):
    """
    Dataset for Inference and Validation (Gallery Generation).
    Returns single images with their metadata.
    """

    def __init__(self, csv_file, subset_size=None, transform=None):
        """
        Args:
            csv_file (str): Path to the metadata CSV file.
            subset_size (int, optional): Limit dataset size for debugging.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        if subset_size is not None:
            self.df = self.df.head(subset_size)
        self.transform = transform

        # Check if 'Id' column exists (it won't for test.csv)
        self.has_labels = "Id" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = load_and_preprocess_image(
            file_path, height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH
        )

        # Apply Transforms
        if self.transform:
            img = self.transform(img)

        # Metadata
        image_name = row["Image"]
        whale_id = row["Id"] if self.has_labels else ""

        return img, image_name, whale_id
