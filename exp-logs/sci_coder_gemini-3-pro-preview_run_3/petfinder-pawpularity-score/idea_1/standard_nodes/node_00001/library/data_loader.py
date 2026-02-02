import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import set_seed


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Loads images, applies transformations, and extracts metadata features.
    """

    def __init__(self, df, input_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata and file paths.
            input_dir (str): Root directory for input images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Flag to indicate if this is the test set (no targets).
        """
        self.df = df.reset_index(drop=True)
        self.input_dir = input_dir
        self.transform = transform
        self.is_test = is_test

        # Identify feature columns
        # Note: 'Subject Focus' is the column name in the CSV, but sometimes referred to as 'Focus'
        self.feature_cols = [
            "Subject Focus",
            "Eyes",
            "Face",
            "Near",
            "Action",
            "Accessory",
            "Group",
            "Collage",
            "Human",
            "Occlusion",
            "Info",
            "Blur",
        ]

        # Handle case where 'Subject Focus' might be named 'Focus'
        if "Subject Focus" not in self.df.columns and "Focus" in self.df.columns:
            self.feature_cols[0] = "Focus"

        # Verify all feature columns exist
        missing_cols = [c for c in self.feature_cols if c not in self.df.columns]
        if missing_cols:
            # If columns are missing (e.g. in a minimal test csv), we might fill with 0
            # But for this task, we expect them to be present as per description.
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative to input_dir (e.g., "train/id.jpg")
        img_path = os.path.join(self.input_dir, row["file_path"])

        # Read with OpenCV (BGR)
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 3. Extract Metadata Features
        # Ensure we get a float tensor of shape (12,)
        meta_features = row[self.feature_cols].values.astype(np.float32)
        meta_features = torch.tensor(meta_features, dtype=torch.float32)

        # 4. Extract Target and ID
        pet_id = row["Id"]

        if not self.is_test:
            target = row["Pawpularity"]
            target = torch.tensor(target, dtype=torch.float32)
        else:
            # Dummy target for test set
            target = torch.tensor(0.0, dtype=torch.float32)

        return image, meta_features, target, pet_id


def get_transforms(img_size):
    """
    Returns the standard ImageNet normalization and resizing transforms.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataloaders():
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_META_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_META_PATH}")

    df_train = pd.read_csv(Config.TRAIN_META_PATH)
    df_val = pd.read_csv(Config.VAL_META_PATH)
    df_test = pd.read_csv(Config.TEST_META_PATH)

    # 2. Debug Mode - Subset Data
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Using subset of {Config.DEBUG_SUBSET_SIZE} samples."
        )
        df_train = df_train.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SUBSET_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 3. Define Transforms
    transform = get_transforms(Config.IMG_SIZE)

    # 4. Create Datasets
    train_dataset = PawpularityDataset(
        df_train, Config.INPUT_DIR, transform=transform, is_test=False
    )

    val_dataset = PawpularityDataset(
        df_val, Config.INPUT_DIR, transform=transform, is_test=False
    )

    test_dataset = PawpularityDataset(
        df_test, Config.INPUT_DIR, transform=transform, is_test=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
