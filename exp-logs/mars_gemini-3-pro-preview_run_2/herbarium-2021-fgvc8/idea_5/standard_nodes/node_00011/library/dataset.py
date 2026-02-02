import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from library.config import Config
from library.utils import load_taxonomy_mapping


class HerbariumDataset(Dataset):
    """
    Custom Dataset for Herbarium 2021 Competition.
    Handles loading images and mapping species to hierarchical labels (Family, Order).
    """

    def __init__(self, df, transform=None, taxonomy_map=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            taxonomy_map (pd.DataFrame, optional): Mapping from category_id to family_id and order_id.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # If training or validation, we need labels
        if self.mode in ["train", "val"]:
            # Ensure taxonomy map is merged if provided
            if taxonomy_map is not None:
                # Merge on category_id to get family_id and order_id
                # We use left merge to preserve the original dataframe order/indices
                self.df = pd.merge(self.df, taxonomy_map, on="category_id", how="left")

            # Extract labels
            self.species_labels = self.df["category_id"].values
            self.family_labels = self.df["family_id"].values
            self.order_labels = self.df["order_id"].values
        else:
            # For test set, we only need image_ids for submission
            self.image_ids = self.df["image_id"].values

        # Pre-compute full paths to avoid string concatenation in __getitem__
        # The file_path in metadata is relative to input dir (e.g., "train/images/...")
        self.file_paths = [
            os.path.join(self.input_dir, fp) for fp in self.df["file_path"].values
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.file_paths[idx]

        # Use OpenCV for loading (faster than PIL in some envs)
        img = cv2.imread(img_path)

        if img is None:
            # Fallback for missing/corrupt images: return a black image
            # This prevents the dataloader from crashing
            img = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            # Albumentations or Torchvision transforms
            # If using torchvision transforms, they expect PIL or Tensor.
            # ToTensor() handles numpy arrays (H, W, C) -> (C, H, W) scaled to [0, 1]
            img = self.transform(img)

        if self.mode in ["train", "val"]:
            # Return tuple of labels for multi-task learning
            labels = (
                self.species_labels[idx],
                self.family_labels[idx],
                self.order_labels[idx],
            )
            return img, labels
        else:
            # Return image_id for submission generation
            return img, self.image_ids[idx]


def get_dataloaders(stage=1, debug_size=Config.DEBUG_SAMPLE_SIZE):
    """
    Constructs DataLoaders for Train, Validation, and Test sets.

    Args:
        stage (int): 1 for Instance-Balanced Sampling, 2 for Class-Balanced Sampling.
        debug_size (int, optional): If set, truncates datasets for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # ---------------------------------------------------------
    # 1. Define Transforms
    # ---------------------------------------------------------
    # Training Transforms: Augmentation
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    )

    # Validation/Test Transforms: Deterministic
    # Resize to slightly larger then center crop, or just resize
    # Standard practice: Resize 256 -> CenterCrop 224
    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD),
        ]
    )

    # ---------------------------------------------------------
    # 2. Load Metadata
    # ---------------------------------------------------------
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    # Load Taxonomy Mapping (Species -> Family, Order)
    taxonomy_map = load_taxonomy_mapping(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Debugging / Subsampling
    # ---------------------------------------------------------
    if debug_size is not None:
        df_train = df_train.iloc[:debug_size]
        df_val = df_val.iloc[:debug_size]
        df_test = df_test.iloc[:debug_size]

    # ---------------------------------------------------------
    # 4. Create Datasets
    # ---------------------------------------------------------
    train_dataset = HerbariumDataset(
        df_train, transform=train_transform, taxonomy_map=taxonomy_map, mode="train"
    )

    val_dataset = HerbariumDataset(
        df_val, transform=val_transform, taxonomy_map=taxonomy_map, mode="val"
    )

    test_dataset = HerbariumDataset(
        df_test, transform=val_transform, taxonomy_map=None, mode="test"
    )

    # ---------------------------------------------------------
    # 5. Define Samplers
    # ---------------------------------------------------------
    train_sampler = None
    shuffle_train = True

    if stage == 2:
        # Stage 2: Class-Balanced Sampling
        # Calculate weights for each sample based on the inverse frequency of its class
        print("Preparing Class-Balanced Sampler for Stage 2...")

        # Get category labels from the dataset
        # Note: We access the internal merged dataframe or the stored labels
        labels = train_dataset.species_labels

        # Calculate class counts
        # We use bincount if labels are strictly 0..N-1 integers, otherwise value_counts
        # Config.NUM_CLASSES is 64500. Labels should be within this range.
        class_counts = np.bincount(labels, minlength=Config.NUM_CLASSES)

        # Avoid division by zero for classes not present in the split (if any)
        class_counts = class_counts.astype(np.float32)
        class_counts[class_counts == 0] = 1.0  # neutral weight for missing classes

        # Calculate weight per class
        class_weights = 1.0 / class_counts

        # Assign weight to each sample
        sample_weights = class_weights[labels]
        sample_weights = torch.from_numpy(sample_weights).double()

        # Create WeightedRandomSampler
        # num_samples=len(sample_weights) ensures the epoch size remains the same
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle_train = (
            False  # Sampler provides shuffling mutually exclusive with shuffle=True
        )

    # ---------------------------------------------------------
    # 6. Create DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=shuffle_train,
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

    return train_loader, val_loader, test_loader
