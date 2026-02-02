import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data: str = "train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train', 'val', or 'test'. Defines the augmentation strength.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.RandomBrightnessContrast(p=0.5),
                # CoarseDropout simulates SpecAugment (time/freq masking) on the spectrogram image
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE[0] * 0.1),
                    max_width=int(Config.IMG_SIZE[1] * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    elif data in ["val", "test"]:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def load_dataframe(csv_path: str, debug: bool = Config.DEBUG) -> pd.DataFrame:
    """
    Loads the metadata DataFrame and applies debug subsetting if necessary.

    Args:
        csv_path (str): Path to the CSV file.
        debug (bool): Whether to use a small subset for debugging.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    return df


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification using Filtered Spectrograms.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: A.Compose = None,
        img_dir: str = Config.FILTERED_SPECTROGRAM_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, paths, labels).
            transforms (albumentations.Compose): Transforms to apply to the images.
            img_dir (str): Directory containing the filtered spectrogram images.
        """
        self.df = df
        self.transforms = transforms
        self.img_dir = img_dir

        # Identify label columns dynamically (species_0, species_1, ...)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract filename from the metadata path.
        # Metadata path example: "supplemental_data/spectrograms/PC10_....bmp"
        # We only want "PC10_....bmp" to join with our specific img_dir (filtered_spectrograms)
        original_rel_path = row["file_path_spec"]
        filename = os.path.basename(original_rel_path)

        img_path = os.path.join(self.img_dir, filename)

        # Load Image
        image = cv2.imread(img_path)

        # Handle missing images (robustness)
        if image is None:
            # Create a blank image if file is missing/corrupt to prevent crashing
            # In a real scenario, we might want to log this.
            image = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Augmentations/Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Labels
        # We return labels as FloatTensor for BCEWithLogitsLoss
        labels = row[self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels)

        # Also return rec_id for tracking/submission
        rec_id = row["rec_id"]

        return image, labels, rec_id
