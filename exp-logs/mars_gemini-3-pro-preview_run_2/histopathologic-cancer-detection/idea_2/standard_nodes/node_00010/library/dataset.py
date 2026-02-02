import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(phase: str):
    """
    Constructs the data transformation pipeline based on the execution phase.

    Args:
        phase (str): One of 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    # Retrieve normalization stats from Config
    mean = Config.DATA_MEAN
    std = Config.DATA_STD

    # Target input size for the model
    crop_size = Config.INPUT_SIZE[0]

    if phase == "train":
        return transforms.Compose(
            [
                # Convert numpy array (from cv2) to PIL Image for torchvision transforms
                transforms.ToPILImage(),
                # --- Geometric Augmentations (on full 96x96 context) ---
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # Random rotation. Since 64x64 fits inside the inscribed circle of 96x96,
                # we can rotate freely without introducing black artifacts in the crop.
                transforms.RandomRotation(degrees=180),
                # --- Intensity Augmentations ---
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                # --- Cropping ---
                # Crop the center 64x64 region (containing the 32x32 ROI + context)
                transforms.CenterCrop(crop_size),
                # --- Normalization ---
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        # Validation and Test phases
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class PathologyDataset(Dataset):
    """
    PyTorch Dataset for loading pathology patches.
    """

    def __init__(self, mode: str, transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Transform pipeline to apply.
        """
        self.mode = mode
        self.transform = transform

        # Determine which metadata file to load
        if mode == "train":
            csv_path = Config.TRAIN_CSV
        elif mode == "val":
            csv_path = Config.VAL_CSV
        elif mode == "test":
            csv_path = Config.TEST_CSV
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
            )

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Handle Debugging: Sample a subset if enabled
        if Config.DEBUG:
            # Deterministic sampling for reproducibility
            sample_n = min(len(self.df), Config.DEBUG_SAMPLES)
            self.df = self.df.sample(n=sample_n, random_state=Config.SEED).reset_index(
                drop=True
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.df.iloc[idx]

        # Construct absolute file path
        # row['file_path'] is relative (e.g., "train/xxxx.tif")
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        else:
            # Fallback to basic tensor conversion if no transform provided
            img = transforms.ToTensor()(img)

        # Retrieve label
        # For test set, we return a dummy label (0.0) or keep the placeholder
        label = torch.tensor(row["label"], dtype=torch.float32)

        return img, label
