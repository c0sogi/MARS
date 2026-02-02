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


class WhaleTripletDataset(Dataset):
    """
    Dataset for training with Triplet Loss.
    Cite solution_lesson_node_00004: Triplet data structure (Anchor, Positive, Negative).
    """

    def __init__(self, csv_file, subset_size=None, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        # Anchors must be known whales
        self.known_whales = self.df[self.df["Id"] != "new_whale"].reset_index(drop=True)

        if subset_size is not None:
            self.known_whales = self.known_whales.head(subset_size)

        # Group all data by Id
        self.all_groups = self.df.groupby("Id")["file_path"].apply(list).to_dict()
        self.all_ids = list(self.all_groups.keys())

    def __len__(self):
        return len(self.known_whales)

    def __getitem__(self, idx):
        # 1. Anchor
        row = self.known_whales.iloc[idx]
        anchor_id = row["Id"]
        anchor_path_rel = row["file_path"]

        # 2. Positive (Same ID)
        candidates = self.all_groups[anchor_id]
        pos_path_rel = np.random.choice(candidates)

        # 3. Negative (Different ID)
        while True:
            neg_id = np.random.choice(self.all_ids)
            if neg_id != anchor_id:
                break
        neg_path_rel = np.random.choice(self.all_groups[neg_id])

        # Load Images
        anchor_img = load_and_preprocess_image(
            os.path.join(Config.INPUT_DIR, anchor_path_rel),
            height=Config.IMG_HEIGHT,
            width=Config.IMG_WIDTH,
        )
        pos_img = load_and_preprocess_image(
            os.path.join(Config.INPUT_DIR, pos_path_rel),
            height=Config.IMG_HEIGHT,
            width=Config.IMG_WIDTH,
        )
        neg_img = load_and_preprocess_image(
            os.path.join(Config.INPUT_DIR, neg_path_rel),
            height=Config.IMG_HEIGHT,
            width=Config.IMG_WIDTH,
        )

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img


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
