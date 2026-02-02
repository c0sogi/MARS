import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import configuration and utilities
from library.config import Config
from library.utils import set_seed

# Constants for Age Normalization (derived from training data analysis)
AGE_MEAN = 58.6821
AGE_STD = 10.0354


class BreastCancerDataset(Dataset):
    """
    Custom Dataset for Breast Cancer Detection.
    Implements Spatial Channel Expansion (Image + Age + Implant).
    """

    def __init__(self, df, phase="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Image Loading
        rel_path = row["file_path"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Attempt to load with OpenCV (IMREAD_UNCHANGED to preserve depth if needed)
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # Fail loudly if image is missing or corrupt
        if image is None:
            try:
                # Fallback to PIL
                pil_img = Image.open(full_path)
                image = np.array(pil_img)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load image at {full_path}. Error: {e}"
                )

            if image is None:
                raise ValueError(f"Image loaded as None at {full_path}")

        # 2. Image Preprocessing
        # Ensure Grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # If 4 channels (RGBA), take first 3 then gray or just gray?
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        # Normalize to [0, 1] float32
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            image = image.astype(np.float32) / 255.0

        # Ensure 2D shape (H, W) for Albumentations
        if image.ndim > 2:
            image = image.squeeze()

        # 3. Augmentation
        # Apply geometric augmentations if transform is provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Ensure shape is (H, W) after transform
        if image.ndim == 3:
            image = image.squeeze()

        # 4. Spatial Channel Expansion
        # Construct (3, H, W) tensor: [Image, Age, Implant]
        h, w = image.shape

        # Process Age (Standard Scaling)
        age_raw = row["age"]
        if pd.isna(age_raw):
            age_norm = 0.0  # Mean imputation (0.0 after standardization)
        else:
            age_norm = (age_raw - AGE_MEAN) / AGE_STD

        # Process Implant (Binary)
        implant_raw = row["implant"]
        if pd.isna(implant_raw):
            implant_val = 0.0
        else:
            implant_val = float(implant_raw)

        # Broadcast metadata to spatial dimensions
        age_channel = np.full((h, w), age_norm, dtype=np.float32)
        implant_channel = np.full((h, w), implant_val, dtype=np.float32)

        # Stack channels
        combined_img = np.stack(
            [image, age_channel, implant_channel], axis=0
        )  # Shape: (3, H, W)

        # Convert to Torch Tensor
        image_tensor = torch.from_numpy(combined_img).float()

        # 5. Prepare Output Dictionary
        sample = {
            "image": image_tensor,  # The 3-channel input for the model
            "age": torch.tensor(age_norm, dtype=torch.float32),
            "implant": torch.tensor(implant_val, dtype=torch.float32),
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
            "site_id": row["site_id"] if "site_id" in row else -1,
            "laterality": row["laterality"] if "laterality" in row else "U",
            "view": row["view"] if "view" in row else "U",
        }

        # Add labels for train/val
        if self.phase != "test":
            sample["label"] = torch.tensor(row["cancer"], dtype=torch.float32)
        else:
            sample["prediction_id"] = row["prediction_id"]

        return sample


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    """
    target_h, target_w = Config.IMAGE_SIZE

    if phase == "train":
        return A.Compose(
            [
                # Moderate Geometric Augmentations (Shift, Scale, Rotate, Flip)
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                # Resize
                A.Resize(height=target_h, width=target_w),
            ]
        )
    else:
        return A.Compose(
            [
                # Resize only
                A.Resize(height=target_h, width=target_w),
            ]
        )


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA,
    val_metadata_path=Config.VAL_METADATA,
    test_metadata_path=Config.TEST_METADATA,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Initializes and returns DataLoaders for Train, Val, and Test sets.
    """
    # Load Metadata
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    # Debug Mode
    if debug:
        print(f"[DEBUG] Subsetting datasets to {debug_sample_size} samples.")
        df_train = df_train.head(debug_sample_size)
        df_val = df_val.head(debug_sample_size)
        df_test = df_test.head(debug_sample_size)

    # Initialize Datasets
    train_dataset = BreastCancerDataset(
        df_train, phase="train", transform=get_transforms("train")
    )
    val_dataset = BreastCancerDataset(
        df_val, phase="val", transform=get_transforms("val")
    )
    test_dataset = BreastCancerDataset(
        df_test, phase="test", transform=get_transforms("test")
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
