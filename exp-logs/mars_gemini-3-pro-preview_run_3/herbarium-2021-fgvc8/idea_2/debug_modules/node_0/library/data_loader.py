import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config


class HerbariumDataset(Dataset):
    """
    Custom Dataset for Herbarium 2021 FGVC8 Competition.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, category_id, etc.).
            transforms (albumentations.Compose): Transformations to apply to the image.
            mode (str): 'train', 'val', or 'test'. Determines if labels are returned.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.root_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # file_path in CSV is relative to ./input (e.g., train/images/...)
        file_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images (though verification passed)
            # Return a black image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Basic to tensor if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return logic based on mode
        if self.mode in ["train", "val"]:
            label = row["category_id"]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image and image_id for submission creation
            image_id = row["image_id"]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_transforms(mode="train", img_size=256):
    """
    Returns Albumentations transformations for train/val/test.
    """
    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Val and Test
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): If True, attempts to load calculated sample weights from cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # ==========================================
    # Weighted Random Sampler Setup
    # ==========================================
    weights_cache_path = os.path.join(Config.WORKING_DIR, "train_sample_weights.npy")

    sample_weights = None

    if load_cached_data and os.path.exists(weights_cache_path):
        try:
            sample_weights = np.load(weights_cache_path)
            # Verify length matches
            if len(sample_weights) != len(train_df):
                sample_weights = None
        except Exception:
            sample_weights = None

    if sample_weights is None:
        # Calculate weights
        # 1. Count samples per class
        class_counts = train_df["category_id"].value_counts().sort_index()

        # 2. Calculate weight per class (inverse frequency)
        # We map category_id to its weight.
        # Note: category_ids are integers but might not be contiguous 0..N-1 in some datasets,
        # but here we assume they are valid keys.
        # To be safe, we use a dictionary map.
        class_weights_map = (1.0 / class_counts).to_dict()

        # 3. Map weights to each sample in the dataframe
        # This can be slow, so we use map
        sample_weights = (
            train_df["category_id"].map(class_weights_map).values.astype(np.float64)
        )

        # Save to cache
        np.save(weights_cache_path, sample_weights)

    # Convert to tensor for Sampler
    sample_weights_tensor = torch.from_numpy(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor, num_samples=len(train_df), replacement=True
    )

    # ==========================================
    # Dataset Instantiation
    # ==========================================
    train_dataset = HerbariumDataset(
        train_df, transforms=get_transforms("train", Config.IMG_SIZE), mode="train"
    )

    val_dataset = HerbariumDataset(
        val_df, transforms=get_transforms("val", Config.IMG_SIZE), mode="val"
    )

    test_dataset = HerbariumDataset(
        test_df, transforms=get_transforms("test", Config.IMG_SIZE), mode="test"
    )

    # ==========================================
    # DataLoader Instantiation
    # ==========================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
