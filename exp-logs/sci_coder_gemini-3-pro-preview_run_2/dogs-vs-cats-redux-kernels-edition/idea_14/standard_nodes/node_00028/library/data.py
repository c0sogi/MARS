import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    Reads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, root_dir=Config.INPUT_DIR, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and 'label' (or 'id').
            transforms (albumentations.Compose): Transforms to apply.
            root_dir (str): Root directory for image paths.
            is_test (bool): Whether this is the test set (returns id instead of label).
        """
        self.df = df
        self.transforms = transforms
        self.root_dir = root_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path
        # Metadata paths are relative (e.g., "train/cat.0.jpg")
        img_path = os.path.join(self.root_dir, row["filepath"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (though analysis showed 0 missing)
            # Create a black image to prevent crashing
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            # Return image and ID for submission mapping
            return image, row["id"]
        else:
            # Return image and label
            # Label is float for compatibility with BCEWithLogitsLoss / Mixup
            return image, torch.tensor(row["label"], dtype=torch.float32)


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms based on the mode.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(Config.IMG_SIZE, Config.IMG_SIZE),
                    scale=(Config.RRC_SCALE_MIN, Config.RRC_SCALE_MAX),
                    p=1.0,
                ),
                A.HorizontalFlip(p=Config.HFLIP_PROB),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_data_loaders(fold_id, load_cached_data=True):
    """
    Creates train and validation DataLoaders for a specific fold.
    Merges metadata/train.csv and metadata/val.csv to perform full 5-fold CV.

    Args:
        fold_id (int): The fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Unused here as we load from CSVs, but kept for signature compliance.

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)

    # 2. Merge to form the full dataset
    full_df = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # 3. Create Stratified Folds
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We need to extract the folds. The generator yields indices.
    # We iterate to find the specific fold_id.
    fold_generator = skf.split(full_df, full_df["label"])

    train_idx, val_idx = None, None
    for current_fold, (t_idx, v_idx) in enumerate(fold_generator):
        if current_fold == fold_id:
            train_idx = t_idx
            val_idx = v_idx
            break

    if train_idx is None:
        raise ValueError(f"Fold {fold_id} out of range for {Config.N_FOLDS} folds.")

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    # Debugging: Subset if enabled
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        print(f"[DEBUG] Training on subset of size {len(train_df)}")

    # 4. Create Datasets
    train_dataset = DogCatDataset(
        train_df, transforms=get_transforms(mode="train"), root_dir=Config.INPUT_DIR
    )

    val_dataset = DogCatDataset(
        val_df, transforms=get_transforms(mode="val"), root_dir=Config.INPUT_DIR
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for batch norm stability and mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates the test DataLoader.

    Returns:
        test_loader
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    test_dataset = DogCatDataset(
        test_df,
        transforms=get_transforms(mode="test"),
        root_dir=Config.INPUT_DIR,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
