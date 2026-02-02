import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config


def get_transforms(view_mode, mean, std, img_size):
    """
    Generates the Albumentations transformation pipeline based on the view mode and backbone requirements.

    Args:
        view_mode (str): 'warped' (direct resize) or 'preserved' (resize + pad).
        mean (tuple): Normalization mean (RGB).
        std (tuple): Normalization std (RGB).
        img_size (int): Target image height/width.

    Returns:
        A.Compose: The composed transformation pipeline.
    """
    transforms_list = []

    if view_mode == "warped":
        # Direct resize: distorts aspect ratio but fills the square
        transforms_list.append(A.Resize(height=img_size, width=img_size))
    elif view_mode == "preserved":
        # Preserved ratio: Resize longest edge to img_size, then pad
        transforms_list.append(A.LongestMaxSize(max_size=img_size))
        transforms_list.append(
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_REFLECT_101,
                value=0,
            )
        )
    else:
        raise ValueError(
            f"Unknown view_mode: {view_mode}. Expected 'warped' or 'preserved'."
        )

    # Common normalization and tensor conversion
    transforms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])

    return A.Compose(transforms_list)


class PetDataset(Dataset):
    """
    PyTorch Dataset for Pet Pawpularity.
    Loads images and binary metadata features.
    """

    def __init__(self, csv_path, transform=None, debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (A.Compose, optional): Albumentations transforms.
            debug (bool): If True, limits the dataset to a small subset for debugging.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.input_dir = Config.INPUT_DIR
        self.binary_features = Config.BINARY_FEATURES

        if debug:
            # Use a small subset for debugging
            self.df = self.df.head(64).reset_index(drop=True)

        # Check if target column exists (it won't for the test set usually,
        # but our metadata generation script might preserve structure)
        self.has_target = "Pawpularity" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path is relative, e.g., "train/0007de...jpg"
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read with OpenCV (BGR)
        image = cv2.imread(img_path)

        if image is None:
            # Handle missing image gracefully (return black image)
            # This shouldn't happen given the verification checks
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        # 3. Extract Metadata Features
        # Get the binary features as a float tensor
        meta_values = row[self.binary_features].values.astype(np.float32)
        meta = torch.tensor(meta_values, dtype=torch.float32)

        # 4. Extract Target
        target = 0.0
        if self.has_target:
            target = row["Pawpularity"]

        # Return dictionary
        return {
            "image": image,
            "meta": meta,
            "target": torch.tensor(target, dtype=torch.float32),
            "id": str(row["Id"]),
        }


def get_dataloader(
    csv_path,
    batch_size=Config.BATCH_SIZE,
    view_mode="warped",
    backbone_type="dinov2",
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    debug=False,
):
    """
    Factory function to create a DataLoader with specific configurations.

    Args:
        csv_path (str): Path to the metadata CSV.
        batch_size (int): Batch size.
        view_mode (str): 'warped' or 'preserved'. Controls image resizing strategy.
        backbone_type (str): 'clip', 'dinov2', or 'convnext'. Determines normalization stats.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, uses a subset of data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1. Resolve Normalization Statistics based on Backbone
    if backbone_type not in Config.BACKBONES:
        raise ValueError(
            f"Unknown backbone_type: {backbone_type}. Available: {list(Config.BACKBONES.keys())}"
        )

    backbone_cfg = Config.BACKBONES[backbone_type]
    mean = backbone_cfg["mean"]
    std = backbone_cfg["std"]

    # 2. Build Transformation Pipeline
    transform = get_transforms(
        view_mode=view_mode, mean=mean, std=std, img_size=Config.IMG_SIZE
    )

    # 3. Instantiate Dataset
    dataset = PetDataset(csv_path=csv_path, transform=transform, debug=debug)

    # 4. Instantiate DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return loader
