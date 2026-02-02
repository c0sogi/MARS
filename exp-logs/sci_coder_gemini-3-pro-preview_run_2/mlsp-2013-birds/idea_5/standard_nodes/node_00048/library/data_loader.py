import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from library.utils import set_seed, seed_worker


class BirdDataset(Dataset):
    """
    Standard Bird Spectrogram Dataset.
    Uses rectangular resizing and Pseudo-RGB conversion.
    Cite solution_lesson_node_00034, solution_lesson_node_00030
    """

    def __init__(self, df, root_dir, image_size=224, train=True):
        self.df = df
        self.root_dir = root_dir
        self.image_size = image_size  # This will be height
        self.train = train
        self.labels = [c for c in df.columns if c.startswith("species_")]

        # Rectangular resize: Height=224, Width=448 (Cite solution_lesson_node_00034)
        target_height = image_size
        target_width = int(image_size * 2)

        if self.train:
            self.transform = A.Compose(
                [
                    A.Resize(height=target_height, width=target_width),
                    A.RandomBrightnessContrast(p=0.5),
                    A.CoarseDropout(
                        max_holes=8, max_height=20, max_width=20, p=0.5
                    ),  # SpecAugment-like
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=target_height, width=target_width),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["file_path_spec"]
        rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
        img_path = os.path.join(self.root_dir, rel_path)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros((256, 1000), dtype=np.uint8)

        # Albumentations
        augmented = self.transform(image=image)["image"]

        # Pseudo-RGB: Replicate channels (Cite solution_lesson_node_00030)
        image_rgb = cv2.merge([augmented, augmented, augmented])

        # Normalize [0, 1] and CHW layout
        img_norm = image_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)  # (C, H, W)

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

    # Initialize Loaders with RNG seeding (Cite solution_lesson_node_00045)
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader
