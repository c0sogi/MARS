import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from library.config import Config
from library.taxonomy import TaxonomyManager


class HerbariumDataset(Dataset):
    """
    Dataset class for the Herbarium 2021 dataset.
    Handles loading images and mapping species labels to family and order labels.
    """

    def __init__(
        self,
        csv_path,
        taxonomy_manager=None,
        transform=None,
        is_test=False,
        debug_size=None,
    ):
        self.csv_path = csv_path
        self.root_dir = Config.INPUT_DIR
        self.transform = transform
        self.is_test = is_test

        # Load metadata
        self.df = pd.read_csv(csv_path)

        # Debug subsampling
        if debug_size is not None and debug_size > 0:
            if len(self.df) > debug_size:
                # Use a fixed seed for reproducibility in debug mode
                self.df = self.df.sample(
                    n=debug_size, random_state=Config.SEED
                ).reset_index(drop=True)

        # If training/validation, map species to family and order
        if not self.is_test:
            if taxonomy_manager is None:
                raise ValueError(
                    "TaxonomyManager must be provided for training/validation sets."
                )

            species_to_family, species_to_order = taxonomy_manager.get_mappings()

            # Ensure category_id exists
            if "category_id" not in self.df.columns:
                raise ValueError(f"category_id column missing in {csv_path}")

            # Vectorized mapping of species to family/order
            self.df["family_id"] = self.df["category_id"].map(species_to_family)
            self.df["order_id"] = self.df["category_id"].map(species_to_order)

            # Verification
            if (
                self.df["family_id"].isnull().any()
                or self.df["order_id"].isnull().any()
            ):
                raise ValueError(
                    "Some category_ids could not be mapped to family/order. Check taxonomy coverage."
                )

            self.df["family_id"] = self.df["family_id"].astype(int)
            self.df["order_id"] = self.df["order_id"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in csv is relative to input dir (e.g. "train/images/...")
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing/corrupt images (robustness)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            t = ToTensorV2()
            image = t(image=image)["image"]

        if self.is_test:
            return image, row["image_id"]
        else:
            return image, row["category_id"], row["family_id"], row["order_id"]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                # Normalize using ImageNet mean/std
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize to target size directly
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(stage=1, debug_sample_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Factory function to create DataLoaders for the specified training stage.

    Args:
        stage (int): 1 for Representation Learning (Instance-Balanced),
                     2 for Classifier Re-balancing (Class-Balanced).
        debug_sample_size (int, optional): Number of samples to use for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, taxonomy_manager)
    """
    # 1. Prepare Taxonomy
    tax_mgr = TaxonomyManager()
    tax_mgr.load(load_cached_data=True)

    # 2. Prepare Transforms
    train_transform = get_transforms("train")
    eval_transform = get_transforms("val")

    # 3. Create Datasets
    train_dataset = HerbariumDataset(
        csv_path=Config.TRAIN_CSV,
        taxonomy_manager=tax_mgr,
        transform=train_transform,
        is_test=False,
        debug_size=debug_sample_size,
    )

    val_dataset = HerbariumDataset(
        csv_path=Config.VAL_CSV,
        taxonomy_manager=tax_mgr,
        transform=eval_transform,
        is_test=False,
        debug_size=debug_sample_size,
    )

    test_dataset = HerbariumDataset(
        csv_path=Config.TEST_CSV,
        taxonomy_manager=None,
        transform=eval_transform,
        is_test=True,
        debug_size=debug_sample_size,
    )

    # 4. Configure Samplers
    train_sampler = None
    shuffle = True

    if stage == 2:
        # Stage 2: Class-Balanced Sampling
        # We want to sample inversely proportional to class frequency
        print("Configuring Class-Balanced Sampler for Stage 2...")

        # Get labels
        labels = train_dataset.df["category_id"].values

        # Calculate counts
        unique_labels, counts = np.unique(labels, return_counts=True)
        count_dict = dict(zip(unique_labels, counts))

        # Calculate weights for each sample: weight = 1.0 / count
        weights = np.array([1.0 / count_dict[label] for label in labels])
        weights = torch.DoubleTensor(weights)

        # Create sampler
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        # Sampler option is mutually exclusive with shuffle
        shuffle = False

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle,
        sampler=train_sampler,
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

    return train_loader, val_loader, test_loader, tax_mgr
