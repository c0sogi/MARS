import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from library.utils import seed_everything
from library.data_mappings import get_species_to_genus_mapping

# Configuration Constants
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_4/"
METADATA_DIR = "./metadata"


class HerbariumDataset(Dataset):
    """
    Dataset class for Herbarium 2020.
    Returns:
        - image: Transformed image tensor.
        - species_id: Target label for species (int).
        - genus_id: Target label for genus (int).
    """

    def __init__(self, df, transform=None, species_to_genus_map=None, is_test=False):
        self.df = df
        self.transform = transform
        self.species_to_genus_map = species_to_genus_map
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(INPUT_DIR, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for missing/corrupt images (should not happen based on analysis)
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            # Return image and ID for submission
            return image, row["image_id"]
        else:
            species_id = row["category_id"]
            # Retrieve genus_id from map; default to 0 or handle error if missing
            # The map is expected to cover all training species.
            genus_id = self.species_to_genus_map.get(species_id, 0)

            return image, species_id, genus_id


def get_transforms(image_size, is_training=True):
    """
    Returns Albumentations transforms based on the image size and phase.
    Implements Progressive Resizing logic.
    """
    if is_training:
        return A.Compose(
            [
                # Resize to the target size for the current phase
                A.Resize(height=image_size, width=image_size),
                # Augmentations for regularization
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Inference Transforms
        # If image_size is 224 (final phase), use the specific Resize(256)->Crop(224) strategy
        if image_size >= 224:
            resize_dim = 256
            crop_dim = 224
        else:
            # For smaller progressive sizes, just resize directly or use a small margin
            resize_dim = image_size
            crop_dim = image_size

        return A.Compose(
            [
                A.Resize(height=resize_dim, width=resize_dim),
                A.CenterCrop(height=crop_dim, width=crop_dim),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    load_cached_data: bool = True,
    sample_limit: int = None,
):
    """
    Creates DataLoaders for training and validation.

    Args:
        image_size (int): Target image resolution (e.g., 160 or 224).
        batch_size (int): Batch size.
        num_workers (int): Number of DataLoader workers.
        load_cached_data (bool): Whether to use cached weights/mappings.
        sample_limit (int, optional): Limit dataset size for debugging.

    Returns:
        train_loader, val_loader, num_classes, num_genera
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Load Metadata
    train_csv_path = os.path.join(METADATA_DIR, "train.csv")
    val_csv_path = os.path.join(METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    if sample_limit:
        train_df = train_df.head(sample_limit)
        val_df = val_df.head(sample_limit)

    # 2. Get Species -> Genus Mapping
    species_to_genus, num_genera = get_species_to_genus_mapping(
        json_path=os.path.join(INPUT_DIR, "nybg2020/train/metadata.json"),
        cache_dir=CACHE_DIR,
        load_cached_data=load_cached_data,
    )

    # Calculate number of species classes
    # We assume classes are 0-indexed and contiguous or we take the max ID
    # The provided metadata analysis shows max ID around 32092 (total 32093 classes)
    # Ideally, we should use the max category_id found in the full dataset + 1
    num_classes = max(train_df["category_id"].max(), val_df["category_id"].max()) + 1

    # 3. Compute/Load Sampling Weights (Inverse Square Root)
    weights_cache_path = os.path.join(CACHE_DIR, "train_weights.npy")

    sample_weights = None

    # Check if cache exists and is valid for current dataframe length
    if load_cached_data and os.path.exists(weights_cache_path):
        try:
            loaded_weights = np.load(weights_cache_path)
            if len(loaded_weights) == len(train_df):
                sample_weights = torch.from_numpy(loaded_weights).double()
                # print(f"Loaded sample weights from {weights_cache_path}")
            else:
                pass  # Cache invalid (size mismatch), recompute
        except Exception:
            pass  # Load failed, recompute

    if sample_weights is None:
        # print("Computing inverse square-root sampling weights...")
        # Count samples per class
        class_counts = train_df["category_id"].value_counts().sort_index()

        # Create a map from category_id to weight: w = 1 / sqrt(count)
        # Handle potential missing classes in sequence by using reindex or dict map
        class_weights_map = (1.0 / np.sqrt(class_counts)).to_dict()

        # Map weights to each sample
        # Default weight 0 for classes not in train (should not happen)
        weights_numpy = train_df["category_id"].map(class_weights_map).fillna(0).values

        # Save to cache
        np.save(weights_cache_path, weights_numpy)
        sample_weights = torch.from_numpy(weights_numpy).double()
        # print(f"Saved sample weights to {weights_cache_path}")

    # 4. Create Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 5. Create Datasets and Loaders
    train_dataset = HerbariumDataset(
        df=train_df,
        transform=get_transforms(image_size, is_training=True),
        species_to_genus_map=species_to_genus,
        is_test=False,
    )

    val_dataset = HerbariumDataset(
        df=val_df,
        transform=get_transforms(image_size, is_training=False),
        species_to_genus_map=species_to_genus,
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Mutually exclusive with shuffle
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, num_classes, num_genera


def get_test_dataloader(image_size: int, batch_size: int, num_workers: int = 4):
    """
    Creates DataLoader for the test set.
    """
    test_csv_path = os.path.join(METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    test_dataset = HerbariumDataset(
        df=test_df,
        transform=get_transforms(image_size, is_training=False),
        species_to_genus_map=None,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
