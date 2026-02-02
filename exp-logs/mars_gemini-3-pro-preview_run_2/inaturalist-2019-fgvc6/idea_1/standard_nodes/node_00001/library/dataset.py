import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class INatDataset(Dataset):
    """
    PyTorch Dataset for iNaturalist 2019.
    Handles image loading, preprocessing, and label mapping.
    """

    def __init__(self, df, root_dir, transform=None, mode="train", label_map=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Path to the image directory.
            transform (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
            label_map (dict): Mapping from original category_id to model class index (0..N-1).
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.root_dir, row["file_name"])

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images (should not happen given metadata check)
            # Return a black image of expected size to prevent crash
            image = np.zeros((Config.INPUT_SIZE, Config.INPUT_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            category_id = row["category_id"]
            # Map category_id to contiguous index 0..N-1
            label = self.label_map[category_id] if self.label_map else category_id
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test, return image and image_id (needed for submission)
            image_id = row["image_id"]
            return image, image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transformations for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(Config.INPUT_SIZE, Config.INPUT_SIZE)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize shorter side to 256, then CenterCrop
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=256),
                A.CenterCrop(height=Config.INPUT_SIZE, width=Config.INPUT_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_loaders():
    """
    Constructs and returns DataLoaders for train, validation, and test sets.

    Returns:
        train_loader, val_loader, test_loader

    Note: The test_loader.dataset will have an attribute 'idx_to_class'
    for decoding predictions.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 3. Create Label Mappings
    # Ensure consistent mapping from category_id to 0..N-1
    # We use the FULL training set to define the classes before any subsampling.
    unique_categories = sorted(train_df["category_id"].unique())

    # Map: category_id (dataset) -> index (model output 0..1009)
    cat_to_idx = {cat: i for i, cat in enumerate(unique_categories)}

    # Map: index (model output) -> category_id (submission)
    idx_to_cat = {i: cat for i, cat in enumerate(unique_categories)}

    # 2. Handle Debugging (Subsampling)
    if Config.DEBUG_SAMPLE_SIZE is not None:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # 4. Create Datasets
    train_dataset = INatDataset(
        df=train_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train"),
        mode="train",
        label_map=cat_to_idx,
    )

    val_dataset = INatDataset(
        df=val_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="val"),
        mode="val",
        label_map=cat_to_idx,
    )

    test_dataset = INatDataset(
        df=test_df,
        root_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="test"),
        mode="test",
        label_map=None,  # No labels in test
    )

    # Attach the decoder to the test dataset for easy access during inference
    test_dataset.idx_to_class = idx_to_cat
    # Also attach to val dataset if needed for analysis
    val_dataset.idx_to_class = idx_to_cat

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
