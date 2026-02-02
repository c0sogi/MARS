import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    A PyTorch Dataset for Bird Spectrograms.
    Uses rectangular resizing and Pseudo-RGB conversion.
    """

    def __init__(self, df, root_dir, image_size=(224, 448), train=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory of the input data.
            image_size (tuple): Target (height, width).
            train (bool): Whether to apply training augmentations.
        """
        self.df = df
        self.root_dir = root_dir
        self.image_size = image_size  # (H, W)
        self.train = train
        # Identify label columns (species_0 to species_18)
        self.labels = [c for c in df.columns if c.startswith("species_")]

        # Define Albumentations pipeline
        # Cite solution_lesson_node_00034: Use rectangular input
        # Cite solution_lesson_node_00022: Keep photometric augmentations
        if self.train:
            self.transform = A.Compose(
                [
                    A.Resize(height=image_size[0], width=image_size[1]),
                    A.RandomBrightnessContrast(p=0.5),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=image_size[0], width=image_size[1]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct path to filtered spectrograms
        rel_path = row["file_path_spec"]
        rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
        img_path = os.path.join(self.root_dir, rel_path)

        # Load Image (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((256, 1000), dtype=np.uint8)

        # Convert to Pseudo-RGB (Cite solution_lesson_node_00030)
        # Replicate the single channel 3 times
        image = cv2.merge([image, image, image])

        # Apply Albumentations
        augmented = self.transform(image=image)["image"]

        # Normalize to [0, 1]
        img_norm = augmented.astype(np.float32) / 255.0

        # Convert to Tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)

        # Prepare Labels
        if self.train:
            label_vec = row[self.labels].values.astype(np.float32)
            return img_tensor, torch.tensor(label_vec)
        else:
            dummy_labels = np.zeros(len(self.labels), dtype=np.float32)
            return img_tensor, torch.tensor(dummy_labels)


def get_loaders(input_dir, metadata_dir, batch_size=32, num_workers=2, image_size=224):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        input_dir (str): Path to input directory.
        metadata_dir (str): Path to metadata directory containing csv files.
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        image_size (int): Image size for resizing.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Initialize Datasets
    train_ds = BirdDataset(train_df, input_dir, image_size=image_size, train=True)
    val_ds = BirdDataset(val_df, input_dir, image_size=image_size, train=False)
    test_ds = BirdDataset(test_df, input_dir, image_size=image_size, train=False)

    # Initialize Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
