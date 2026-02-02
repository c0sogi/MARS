import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from timm.data.mixup import Mixup

from library.config import Config
from library.utils import seed_everything


def get_id_map():
    """
    Generates a mapping between the original category_ids and zero-based indices.
    Reads from the training metadata to ensure all classes are covered.

    Returns:
        id2idx (dict): Mapping from category_id to index (0 to N-1).
        idx2id (dict): Mapping from index to category_id.
    """
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_METADATA}")

    df = pd.read_csv(Config.TRAIN_METADATA)
    unique_ids = sorted(df["category_id"].unique())

    id2idx = {cat_id: idx for idx, cat_id in enumerate(unique_ids)}
    idx2id = {idx: cat_id for idx, cat_id in enumerate(unique_ids)}

    return id2idx, idx2id


class SpeciesDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, target_map=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_name, category_id).
            root_dir (str): Root directory for images.
            transform (A.Compose): Albumentations transforms.
            target_map (dict): Mapping from category_id to index.
            is_test (bool): If True, does not look for labels.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.target_map = target_map
        self.is_test = is_test

        # Pre-extract paths and labels for faster access
        self.file_names = df["file_name"].values
        if not self.is_test:
            self.category_ids = df["category_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images to prevent crash
            # Create a black image of default size (e.g., 224x224)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            t = ToTensorV2()
            image = t(image=image)["image"]

        # Return logic
        if self.is_test:
            return image, torch.tensor(0)  # Dummy target for test

        cat_id = self.category_ids[idx]
        label = self.target_map[cat_id] if self.target_map else cat_id

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(phase_config, is_train=False):
    """
    Creates the Albumentations transform pipeline based on the training phase.

    Args:
        phase_config (dict): Configuration dictionary for the current phase.
        is_train (bool): Whether to return training or validation transforms.
    """
    img_size = phase_config["img_size"]

    if is_train:
        # Base transforms
        transforms = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
        ]

        # Phase 2 Specific: Heavier Augmentation (Simulating RandAugment/Pixel-level noise)
        # We only apply these if mixup is NOT active (or if it's explicitly Phase 2)
        # Config logic: Phase 1 uses Mixup (so we keep image augs lighter to avoid over-distortion)
        # Phase 2 disables Mixup but enables "RandAugment and Random Erasing".
        if phase_config.get("name") == "phase_2":
            transforms.extend(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.5,
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=int(img_size * 0.1),
                        max_width=int(img_size * 0.1),
                        p=0.5,
                    ),
                ]
            )

        transforms.extend(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        return A.Compose(transforms)

    else:
        # Validation / Test Transforms
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_mixup_fn(phase_config):
    """
    Returns a Mixup function if active in the configuration.
    """
    if phase_config.get("mixup_active", False):
        return Mixup(
            mixup_alpha=phase_config.get("mixup_alpha", 0.8),
            cutmix_alpha=phase_config.get("cutmix_alpha", 1.0),
            prob=phase_config.get("mixup_prob", 1.0),
            switch_prob=0.5,
            mode="batch",
            label_smoothing=phase_config.get("label_smoothing", 0.1),
            num_classes=Config.NUM_CLASSES,
        )
    return None


def get_loaders(phase_config):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        phase_config (dict): Configuration for the specific training phase.

    Returns:
        train_loader, val_loader, test_loader, mixup_fn
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Debug Mode: Subsample
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        print(
            f"DEBUG MODE: Subsampled data to {len(train_df)} train, {len(val_df)} val."
        )

    # Get Label Mapping
    id2idx, _ = get_id_map()

    # Transforms
    train_transform = get_transforms(phase_config, is_train=True)
    val_transform = get_transforms(phase_config, is_train=False)

    # Datasets
    train_dataset = SpeciesDataset(
        train_df,
        Config.INPUT_DIR,
        transform=train_transform,
        target_map=id2idx,
        is_test=False,
    )

    val_dataset = SpeciesDataset(
        val_df,
        Config.INPUT_DIR,
        transform=val_transform,
        target_map=id2idx,
        is_test=False,
    )

    test_dataset = SpeciesDataset(
        test_df,
        Config.INPUT_DIR,
        transform=val_transform,
        target_map=None,
        is_test=True,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config["batch_size"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=phase_config["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=phase_config["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Mixup Function
    mixup_fn = get_mixup_fn(phase_config)

    return train_loader, val_loader, test_loader, mixup_fn
