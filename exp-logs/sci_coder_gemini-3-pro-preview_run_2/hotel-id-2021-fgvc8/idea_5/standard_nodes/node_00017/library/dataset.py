import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Sampler
from library.config import Config


def get_label_map(train_df, load_cached_data=True):
    """
    Generates or loads a mapping from original hotel_id to contiguous integers.
    """
    cache_path = os.path.join(Config.working_dir, "classes.npy")

    if load_cached_data and os.path.exists(cache_path):
        unique_ids = np.load(cache_path)
        # Ensure cached classes cover the current dataset
        if not set(train_df["hotel_id"].unique()).issubset(set(unique_ids)):
            unique_ids = np.sort(train_df["hotel_id"].unique())
            np.save(cache_path, unique_ids)
    else:
        unique_ids = np.sort(train_df["hotel_id"].unique())
        np.save(cache_path, unique_ids)

    label_to_idx = {label: idx for idx, label in enumerate(unique_ids)}
    return label_to_idx, unique_ids


class HotelDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        data_root=Config.input_dir,
        is_test=False,
        label_map=None,
    ):
        self.df = df
        self.transform = transform
        self.data_root = data_root
        self.is_test = is_test
        self.label_map = label_map

        # Pre-compute paths and labels to speed up __getitem__
        self.file_paths = df["file_path"].values
        self.image_names = df["image"].values

        if not self.is_test:
            # Map original hotel_ids to contiguous indices
            self.labels = df["hotel_id"].map(self.label_map).values.astype(np.int64)
        else:
            self.labels = np.full(len(df), -1, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata validation ensures existence)
            # Create a black image of correct size
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]
        image_name = self.image_names[idx]

        return image, label, image_name


class PKSampler(Sampler):
    """
    Randomly samples P identities, then K instances of each identity for a batch.
    """

    def __init__(self, dataset, batch_size, samples_per_class):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class

        # Group indices by label
        self.labels = dataset.labels
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

        # Calculate length roughly equivalent to one epoch over the data
        self.length = len(dataset) // batch_size

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            # 1. Sample P classes
            selected_classes = np.random.choice(
                self.unique_labels, self.classes_per_batch, replace=False
            )

            batch_indices = []
            for cls in selected_classes:
                indices = self.label_to_indices[cls]

                # 2. Sample K images from this class
                if len(indices) >= self.samples_per_class:
                    selected_indices = np.random.choice(
                        indices, self.samples_per_class, replace=False
                    )
                else:
                    # If not enough samples, sample with replacement
                    selected_indices = np.random.choice(
                        indices, self.samples_per_class, replace=True
                    )

                batch_indices.extend(selected_indices)

            yield batch_indices


def get_transforms(img_size=Config.image_size):
    """
    Returns training and validation transforms.
    """
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            # Color Jitter to handle lighting variations
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            # CoarseDropout (Cutout) for regularization
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            A.Normalize(mean=Config.mean, std=Config.std),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=Config.mean, std=Config.std),
            ToTensorV2(),
        ]
    )

    return train_transform, val_transform


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        # Keep test small for debug but usually we want to test pipeline
        test_df = test_df.sample(
            n=min(len(test_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    # Generate Label Map
    label_to_idx, unique_ids = get_label_map(train_df, load_cached_data=True)

    # Transforms
    train_transform, val_transform = get_transforms(Config.image_size)

    # Datasets
    train_dataset = HotelDataset(
        train_df, transform=train_transform, is_test=False, label_map=label_to_idx
    )

    val_dataset = HotelDataset(
        val_df, transform=val_transform, is_test=False, label_map=label_to_idx
    )

    test_dataset = HotelDataset(
        test_df, transform=val_transform, is_test=True, label_map=None
    )

    # Sampler for Training
    train_sampler = PKSampler(
        train_dataset,
        batch_size=Config.batch_size,
        samples_per_class=Config.samples_per_class,
    )

    # DataLoaders
    # Note: shuffle must be False when using a custom batch_sampler
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # We also need a loader for the gallery (training set without augmentation) for inference
    # This is used to build the embedding database
    gallery_dataset = HotelDataset(
        train_df,
        transform=val_transform,  # Use val transform (no aug)
        is_test=False,
        label_map=label_to_idx,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, gallery_loader, len(unique_ids)
