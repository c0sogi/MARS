import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode="train", img_size=Config.IMG_SIZE):
    """
    Constructs the data augmentation pipeline using Albumentations.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): Target image size (height and width).

    Returns:
        A.Compose: The composed transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Moderate augmentation: Random crop and flip
                # Avoids heavy distortions that might obscure artistic attributes
                A.RandomResizedCrop(
                    height=img_size, width=img_size, scale=(0.75, 1.0), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                # Normalize using ImageNet mean/std
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic transforms for Validation and Test
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.

    Supports:
    1. Loading images from disk via OpenCV.
    2. Parsing multi-label ground truth (Hard Labels).
    3. Loading pre-computed teacher logits (Soft Labels) for distillation.
    """

    def __init__(
        self,
        df,
        transform=None,
        soft_labels=None,
        mode="train",
        input_dir=Config.INPUT_DIR,
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'id', 'file_path', and 'attribute_ids'.
            transform (A.Compose): Albumentations transforms.
            soft_labels (np.ndarray, optional): Array of teacher logits aligned with df.
                                                Shape: (len(df), num_classes).
            mode (str): 'train', 'val', or 'test'.
            input_dir (str): Root directory containing the image files.
        """
        self.df = df
        self.transform = transform
        self.soft_labels = soft_labels
        self.mode = mode
        self.input_dir = input_dir
        self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative (e.g., "train/0001.png")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image (BGR -> RGB)
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for data integrity issues (return black image)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # --- Test Mode ---
        # Return image and ID for submission generation
        if self.mode == "test":
            return image, row["id"]

        # --- Train/Val Mode ---
        # 1. Process Hard Labels (Multi-hot)
        hard_label = torch.zeros(self.num_classes, dtype=torch.float32)
        attr_ids = row.get("attribute_ids", "")

        if pd.notna(attr_ids) and isinstance(attr_ids, str) and attr_ids.strip() != "":
            try:
                # Convert "0 1 2" -> [0, 1, 2]
                indices = [int(x) for x in attr_ids.split()]
                hard_label[indices] = 1.0
            except ValueError:
                # Handle malformed strings gracefully
                pass

        # 2. Process Soft Labels (Distillation)
        if self.soft_labels is not None:
            # Retrieve cached teacher logits
            # Assumes soft_labels array is perfectly aligned with the dataframe
            soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)

            # Return triplet for DistillationLoss
            return image, soft_label, hard_label

        # Standard return for training/validation without distillation
        return image, hard_label


def load_dataset(mode, transform=None, debug=False, use_soft_labels=False):
    """
    Factory function to initialize the ArtworkDataset with correct metadata and settings.

    Args:
        mode (str): 'train', 'val', or 'test'.
        transform (A.Compose, optional): Custom transforms. If None, uses default.
        debug (bool): If True, restricts dataset to Config.DEBUG_SUBSET_SIZE.
        use_soft_labels (bool): If True, loads soft labels from Config.TEACHER_PREDS_PATH.
                                (Only applicable in 'train' mode).

    Returns:
        ArtworkDataset: Initialized dataset instance.
    """
    # 1. Determine Metadata Path
    if mode == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # 2. Load Metadata
    df = pd.read_csv(meta_path)

    # 3. Load Soft Labels (Optional, Train only)
    soft_labels = None
    if mode == "train" and use_soft_labels:
        if os.path.exists(Config.TEACHER_PREDS_PATH):
            # Load logits from .npy file
            soft_labels = np.load(Config.TEACHER_PREDS_PATH)

            # Basic integrity check
            if len(soft_labels) != len(df):
                print(
                    f"Warning: Soft labels length ({len(soft_labels)}) does not match "
                    f"metadata length ({len(df)}). Alignment may be incorrect."
                )
        else:
            print(
                f"Warning: use_soft_labels=True but {Config.TEACHER_PREDS_PATH} not found. "
                "Proceeding with hard labels only."
            )

    # 4. Handle Debugging (Slicing)
    if debug:
        subset_size = min(len(df), Config.DEBUG_SUBSET_SIZE)
        df = df.iloc[:subset_size].reset_index(drop=True)

        # Must slice soft labels to maintain alignment
        if soft_labels is not None:
            soft_labels = soft_labels[:subset_size]

    # 5. Get Transforms
    if transform is None:
        transform = get_transforms(mode=mode)

    return ArtworkDataset(df, transform=transform, soft_labels=soft_labels, mode=mode)
