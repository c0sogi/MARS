import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data_type (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=Config.MEAN,
                    std=Config.STD,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=Config.MEAN,
                    std=Config.STD,
                ),
                ToTensorV2(),
            ]
        )


def get_label_map(train_df, load_cached_data=True):
    """
    Generates or loads a mapping from raw hotel_id to class index [0, num_classes-1].

    Args:
        train_df (pd.DataFrame): The training metadata.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping {hotel_id: class_index}
        list: Inverse mapping [hotel_id, ...] where index is class_index
    """
    cache_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    unique_ids = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading label encoder from {cache_path}")
            unique_ids = np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")
            unique_ids = None

    if unique_ids is None:
        print("Computing label encoder from training data...")
        unique_ids = np.sort(train_df["hotel_id"].unique())
        np.save(cache_path, unique_ids)
        print(f"Label encoder saved to {cache_path}")

    label_map = {hid: i for i, hid in enumerate(unique_ids)}
    return label_map, unique_ids


class HotelDataset(Dataset):
    def __init__(
        self,
        df,
        transform=None,
        label_map=None,
        data_root=Config.INPUT_DIR,
        is_test=False,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transform (A.Compose): Albumentations transforms.
            label_map (dict): Mapping from hotel_id to class index.
            data_root (str): Root directory containing image folders.
            is_test (bool): If True, ignores targets.
        """
        self.df = df
        self.transform = transform
        self.label_map = label_map
        self.data_root = data_root
        self.is_test = is_test

        # Pre-compute file paths to avoid overhead in __getitem__
        # The metadata file_path is relative to Config.INPUT_DIR
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.hotel_ids = self.df["hotel_id"].values
            # Filter out IDs not in label_map (should not happen if label_map is built from train)
            # For validation, we assume all IDs are present in training set (or handle unseen)
            # Given the task description, we assume closed-set for now or handle errors.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata check passed)
            # Return a black image or raise error.
            # For robustness, we'll return a black image of correct size.
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.is_test:
            return image, rel_path  # Return path/ID for submission mapping

        # Get Label
        raw_id = self.hotel_ids[idx]
        label = self.label_map.get(raw_id, -1)

        if label == -1:
            # Handle potential unseen label in validation
            # For this competition, we assume consistent label space
            label = 0

        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        """Helper to get all mapped labels for the sampler."""
        if self.is_test:
            return []
        return [self.label_map.get(hid, -1) for hid in self.hotel_ids]


class BalancedBatchSampler(Sampler):
    """
    Batch Sampler that ensures each batch contains P unique classes
    and K instances per class (M-per-class sampling).
    """

    def __init__(self, dataset, batch_size, samples_per_class):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class

        # Organize indices by class
        self.labels = dataset.get_labels()
        self.labels_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.labels_to_indices:
                self.labels_to_indices[label] = []
            self.labels_to_indices[label].append(idx)

        self.classes = list(self.labels_to_indices.keys())
        self.n_classes = len(self.classes)

        # Define epoch length roughly equal to standard iteration
        self.n_samples = len(dataset)
        self.n_batches = self.n_samples // self.batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            # Select P classes randomly
            selected_classes = np.random.choice(
                self.classes,
                size=self.classes_per_batch,
                replace=(self.n_classes < self.classes_per_batch),
            )

            batch_indices = []
            for cls in selected_classes:
                cls_indices = self.labels_to_indices[cls]

                # Select K instances from this class
                # If class has fewer than K instances, sample with replacement
                replace = len(cls_indices) < self.samples_per_class
                selected_indices = np.random.choice(
                    cls_indices, size=self.samples_per_class, replace=replace
                )
                batch_indices.extend(selected_indices)

            yield batch_indices


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsamples data.
        load_cached_data (bool): Whether to use cached label encoder.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=Config.DEBUG_SAMPLE_SIZE // 5, random_state=Config.SEED
        ).reset_index(drop=True)
        # Test set is small enough, keep as is or sample

    # Prepare Label Mapping
    label_map, unique_ids = get_label_map(train_df, load_cached_data=load_cached_data)
    num_classes = len(unique_ids)

    # Create Datasets
    train_dataset = HotelDataset(
        train_df, transform=get_transforms("train"), label_map=label_map
    )

    val_dataset = HotelDataset(
        val_df, transform=get_transforms("val"), label_map=label_map
    )

    test_dataset = HotelDataset(test_df, transform=get_transforms("test"), is_test=True)

    # Create Sampler for Training
    train_sampler = BalancedBatchSampler(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        samples_per_class=Config.SAMPLES_PER_CLASS,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader, num_classes, unique_ids
