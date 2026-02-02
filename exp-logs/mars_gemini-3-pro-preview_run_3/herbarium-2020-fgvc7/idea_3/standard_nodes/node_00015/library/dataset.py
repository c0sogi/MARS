import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from library import config, utils


class PlantDataset(Dataset):
    """
    Custom Dataset for Plant Species Classification.
    Returns (image, species_label, genus_label) for train/val.
    Returns (image, image_id, dummy_label) for test.
    """

    def __init__(self, df, transform=None, mapping_df=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.input_dir = config.INPUT_DIR

        # Pre-compute file paths
        self.file_paths = df["file_path"].values

        if not self.is_test:
            if mapping_df is None:
                raise ValueError("mapping_df is required for training/validation sets.")

            # Merge df with mapping_df to get species_idx and genus_idx
            # Ensure category_id is consistent type
            temp_df = self.df.copy()
            temp_df["category_id"] = temp_df["category_id"].astype(int)

            # mapping_df contains ['category_id', 'species_idx', 'genus', 'genus_idx']
            merged_df = pd.merge(temp_df, mapping_df, on="category_id", how="left")

            # Extract labels
            self.species_labels = merged_df["species_idx"].values.astype(np.int64)
            self.genus_labels = merged_df["genus_idx"].values.astype(np.int64)
        else:
            # For test set, we need image_id for submission
            self.image_ids = df["image_id"].values.astype(np.int64)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for broken images (should be rare given analysis)
            # Create a black image of correct size
            image = np.zeros(
                (config.IMG_SIZE[0], config.IMG_SIZE[1], 3), dtype=np.uint8
            )
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        if self.is_test:
            # Return image and ID for submission generation
            # Third element is dummy to keep signature consistent if needed
            image_id = self.image_ids[idx]
            return image, image_id, -1
        else:
            # Return image and targets (Species, Genus)
            species_label = self.species_labels[idx]
            genus_label = self.genus_labels[idx]
            return image, species_label, genus_label


def get_dataloaders(batch_size, num_workers, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Implements Inverse Square-Root Weighted Sampling for the training set.
    """
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # 2. Get Mappings (Species <-> Genus)
    mapping_df, num_species, num_genus = config.get_mappings(
        load_cached=load_cached_data
    )

    # 3. Define Transforms
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 4. Calculate/Load Training Weights for Sampling
    weights_cache_path = os.path.join(config.WORKING_DIR, "train_weights.npy")
    train_weights = None

    if load_cached_data and os.path.exists(weights_cache_path):
        print(f"Loading cached training weights from {weights_cache_path}...")
        loaded_weights = np.load(weights_cache_path)
        # Validate cache integrity
        if len(loaded_weights) == len(train_df):
            train_weights = torch.from_numpy(loaded_weights).double()
        else:
            print("Cached weights size mismatch. Recalculating...")

    if train_weights is None:
        print("Calculating inverse square-root sampling weights...")
        # Merge to get species indices for all training samples
        temp_train = pd.merge(train_df, mapping_df, on="category_id", how="left")
        species_indices = temp_train["species_idx"].values

        # Count frequency of each species
        # We use bincount on indices. Ensure indices are 0..N-1
        class_counts = np.bincount(species_indices, minlength=num_species)

        # Avoid division by zero for classes not in training (should not happen with correct mapping)
        class_counts = class_counts.astype(np.float64)
        class_counts[class_counts == 0] = 1.0

        # Calculate weight per class: 1 / sqrt(count)
        class_weights = 1.0 / np.sqrt(class_counts)

        # Assign weight to each sample
        sample_weights = class_weights[species_indices]

        # Save to cache
        np.save(weights_cache_path, sample_weights)

        train_weights = torch.from_numpy(sample_weights).double()

    # 5. Create Datasets
    train_dataset = PlantDataset(
        train_df, transform=train_transform, mapping_df=mapping_df, is_test=False
    )

    val_dataset = PlantDataset(
        val_df, transform=val_test_transform, mapping_df=mapping_df, is_test=False
    )

    test_dataset = PlantDataset(
        test_df, transform=val_test_transform, mapping_df=None, is_test=True
    )

    # 6. Create DataLoaders
    # Weighted Sampler for Training
    train_sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
