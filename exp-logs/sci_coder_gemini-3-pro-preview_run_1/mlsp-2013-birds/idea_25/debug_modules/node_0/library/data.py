import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed


def get_transforms(phase="train", img_size=(256, 640)):
    """
    Returns the Albumentations transform pipeline.

    Args:
        phase (str): 'train' or 'val'/'test'.
        img_size (tuple): Target size (height, width).
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout (CoarseDropout)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def mixup_data(x, y, alpha=0.2, device="cuda"):
    """
    Performs input-level Mixup.

    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target batch.
        alpha (float): Mixup beta distribution parameter.
        device (str): Device to perform operations on.

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class BirdDataset(Dataset):
    def __init__(self, df, input_dir, transform=None, pseudo_labels=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            input_dir (str): Root directory of the input dataset (e.g., './input').
            transform (A.Compose): Albumentations transforms.
            pseudo_labels (dict, optional): Dictionary mapping rec_id (int) to
                                           soft label numpy array. Used for Student training.
        """
        self.df = df
        self.input_dir = input_dir
        self.transform = transform
        self.pseudo_labels = pseudo_labels

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

        # Pre-construct spectrogram directory path
        self.spectrogram_dir = os.path.join(
            self.input_dir, "supplemental_data", "spectrograms"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Resolve file path
        # Metadata 'file_path' points to wav (e.g., essential_data/src_wavs/PC10...wav)
        # We need the corresponding BMP in supplemental_data/spectrograms
        wav_rel_path = row["file_path"]
        basename = os.path.basename(wav_rel_path)
        bmp_filename = os.path.splitext(basename)[0] + ".bmp"
        img_path = os.path.join(self.spectrogram_dir, bmp_filename)

        # Load Image (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for safety (though EDA showed all exist)
            # Create a blank black image of approx correct size
            img = np.zeros((256, 640), dtype=np.uint8)

        # Channel Replication: 1 -> 3 channels
        img = cv2.merge([img, img, img])

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Get Target
        if self.pseudo_labels is not None and rec_id in self.pseudo_labels:
            # Use soft pseudo-label
            target = self.pseudo_labels[rec_id]
            target = torch.tensor(target, dtype=torch.float32)
        else:
            # Use hard ground truth from dataframe
            target = row[self.label_cols].values.astype(np.float32)
            target = torch.tensor(target, dtype=torch.float32)

        return img, target, rec_id


def get_dataloaders(
    metadata_dir="./metadata",
    input_dir="./input",
    batch_size=32,
    img_size=(256, 640),
    pseudo_labels_dict=None,
    num_workers=4,
):
    """
    Creates DataLoaders for Train, Val, and Test sets.

    Args:
        metadata_dir (str): Directory containing train.csv, val.csv, test.csv.
        input_dir (str): Root input directory.
        batch_size (int): Batch size.
        img_size (tuple): Image size (H, W).
        pseudo_labels_dict (dict, optional): Dictionary of pseudo-labels for student training.
                                             If provided, it is passed to the Train dataset.
        num_workers (int): Number of DataLoader workers.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # If using pseudo-labels, we might want to combine train and test for the student
    # But typically, we just augment the training set or replace labels.
    # For this implementation, we follow the standard split but allow pseudo-labels to override
    # targets in the training set if keys match (or if we append test data to train_df externally).

    # Create Transforms
    train_transform = get_transforms(phase="train", img_size=img_size)
    eval_transform = get_transforms(phase="val", img_size=img_size)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df, input_dir, transform=train_transform, pseudo_labels=pseudo_labels_dict
    )

    val_dataset = BirdDataset(
        val_df,
        input_dir,
        transform=eval_transform,
        pseudo_labels=None,  # Validation always uses GT
    )

    test_dataset = BirdDataset(
        test_df, input_dir, transform=eval_transform, pseudo_labels=None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
