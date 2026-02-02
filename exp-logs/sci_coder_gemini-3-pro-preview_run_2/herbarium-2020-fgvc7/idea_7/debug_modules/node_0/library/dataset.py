import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.taxonomy import TaxonomyMapper


class HerbariumDataset(Dataset):
    """
    Custom Dataset for the Herbarium 2020 competition.
    Handles loading images and mapping category IDs to hierarchical labels (Species, Genus, Family).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        transform: A.Compose = None,
        mapper: TaxonomyMapper = None,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, category_id, etc.).
            config (Config): Configuration object.
            transform (A.Compose): Albumentations transform pipeline.
            mapper (TaxonomyMapper): Instance of TaxonomyMapper (required for train/val).
            mode (str): 'train', 'val', or 'test'.
        """
        self.config = config
        self.transform = transform
        self.mode = mode

        # Convert file paths to numpy array for faster indexing
        self.file_paths = df["file_path"].values

        if self.mode in ["train", "val"]:
            if mapper is None:
                raise ValueError("TaxonomyMapper is required for train/val modes.")

            # Extract raw category IDs
            self.category_ids = df["category_id"].values

            # Pre-compute Species Indices (0 to N-1)
            # This avoids dictionary lookups inside the critical __getitem__ path
            self.species_indices = np.array(
                [mapper.species_to_idx[c] for c in self.category_ids], dtype=np.int64
            )

            # Pre-fetch hierarchical mappings as numpy arrays
            # The mapper stores them as tensors, we convert to numpy for CPU-side indexing
            self.s2g = mapper.species_to_genus_map.numpy()
            self.s2f = mapper.species_to_family_map.numpy()

        elif self.mode == "test":
            # For test set, we need image_ids for submission
            self.image_ids = df["image_id"].values

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Construct full path
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.config.INPUT_DIR, file_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback for missing/corrupt images (though metadata check passed)
            # Create a black image of expected size
            img = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.mode in ["train", "val"]:
            # Retrieve labels
            species_idx = self.species_indices[idx]
            genus_idx = self.s2g[species_idx]
            family_idx = self.s2f[species_idx]

            return img, species_idx, genus_idx, family_idx

        else:
            # Return image and ID for submission
            image_id = self.image_ids[idx]
            return img, image_id


def get_transforms(width: int, height: int, mode: str = "train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        width (int): Target width.
        height (int): Target height.
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(height=height, width=width, scale=(0.6, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(config: Config, phase: str = "p1"):
    """
    Creates DataLoaders for training and validation.

    Args:
        config (Config): Configuration object.
        phase (str): 'p1' (Phase 1) or 'p2' (Phase 2) to determine resolution/batch size.

    Returns:
        tuple: (train_loader, val_loader, mapper)
    """
    # 1. Initialize Taxonomy Mapper
    mapper = TaxonomyMapper(config).load_or_build()

    # 2. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)

    # Debug Mode: Subsample data
    if config.DEBUG:
        print(f"DEBUG MODE: Subsampling {config.DEBUG_SAMPLES} images...")
        train_df = train_df.sample(
            n=min(len(train_df), config.DEBUG_SAMPLES), random_state=config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), config.DEBUG_SAMPLES // 5), random_state=config.SEED
        ).reset_index(drop=True)

    # 3. Determine Parameters based on Phase
    if phase == "p1":
        img_size = config.IMG_SIZE_P1
        batch_size = config.BATCH_SIZE_P1
    else:
        img_size = config.IMG_SIZE_P2
        batch_size = config.BATCH_SIZE_P2

    print(
        f"Initializing DataLoaders for Phase {phase}: Size={img_size}, BS={batch_size}"
    )

    # 4. Create Transforms
    train_tf = get_transforms(img_size, img_size, mode="train")
    val_tf = get_transforms(img_size, img_size, mode="val")

    # 5. Create Datasets
    train_ds = HerbariumDataset(
        train_df, config, transform=train_tf, mapper=mapper, mode="train"
    )
    val_ds = HerbariumDataset(
        val_df, config, transform=val_tf, mapper=mapper, mode="val"
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for BatchNorm stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, mapper


def get_test_loader(config: Config, img_size: int):
    """
    Creates DataLoader for the test set.

    Args:
        config (Config): Configuration object.
        img_size (int): Image resolution.

    Returns:
        DataLoader: Test loader.
    """
    test_df = pd.read_csv(config.TEST_CSV)

    # Debug Mode: Subsample test data as well to ensure quick pipeline check
    if config.DEBUG:
        test_df = test_df.sample(
            n=min(len(test_df), config.DEBUG_SAMPLES), random_state=config.SEED
        ).reset_index(drop=True)

    test_tf = get_transforms(img_size, img_size, mode="test")

    test_ds = HerbariumDataset(test_df, config, transform=test_tf, mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE_P2,  # Use P2 batch size (usually safe for inference)
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
