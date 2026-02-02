import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from library.config import Config
from library.utils import get_logger
from library.data_processing import (
    load_dicom,
    crop_breast_roi,
    resize_and_normalize,
    process_and_cache_data,
    collate_bag_fn,
)

logger = get_logger("dataset")


class MammographyBagDataset(Dataset):
    """
    Dataset that loads bags of mammograms (e.g., CC and MLO views) for a specific breast.
    Groups images by patient_id and laterality.
    """

    def __init__(self, df, input_dir=Config.INPUT_DIR, is_train=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing bag-level information.
            input_dir (str): Root directory for images.
            is_train (bool): Whether to return targets (train/val) or bag IDs (test).
        """
        self.df = df
        self.input_dir = input_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_paths = row["file_path"]

        images = []
        # Iterate over all images in the bag
        for rel_path in file_paths:
            full_path = os.path.join(self.input_dir, rel_path)

            # Load and process using library functions
            # 1. Load DICOM
            img = load_dicom(full_path)

            # 2. Crop ROI
            img = crop_breast_roi(img)

            # 3. Resize and Normalize (returns Tensor)
            img_tensor = resize_and_normalize(img)
            images.append(img_tensor)

        # Stack images: (Num_Views, C, H, W)
        if len(images) > 0:
            images_stack = torch.stack(images)
        else:
            # Fallback for empty bag (should not happen based on metadata)
            images_stack = torch.zeros(
                (1, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
            )

        if self.is_train:
            # Prepare Targets
            cancer = torch.tensor(row["cancer"], dtype=torch.float32)

            # Density (handle missing/placeholder, mapped to 0-3 in processing)
            density_val = row.get("density_label", -1)
            density = torch.tensor(density_val, dtype=torch.long)

            # Biopsy (handle missing, assume 0)
            biopsy_val = row.get("biopsy", 0.0)
            biopsy = torch.tensor(biopsy_val, dtype=torch.float32)

            targets = {"cancer": cancer, "density": density, "biopsy": biopsy}
            return images_stack, targets
        else:
            # For inference, return bag_id to map predictions back to submission
            return images_stack, row["bag_id"]


class BalancedBagSampler(Sampler):
    """
    Sampler that ensures a 50/50 ratio of positive and negative bags in each epoch.
    It oversamples the minority class (cancer=1) to match the number of negative samples.
    """

    def __init__(self, df, target_col="cancer"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing the targets.
            target_col (str): Name of the target column.
        """
        self.df = df
        self.target_col = target_col

        self.indices = np.arange(len(df))
        self.labels = df[target_col].values

        self.pos_indices = self.indices[self.labels == 1]
        self.neg_indices = self.indices[self.labels == 0]

        # Define epoch length:
        # We want to see all negative examples at least once per epoch.
        # We balance this by sampling an equal number of positive examples (with replacement).
        self.num_neg = len(self.neg_indices)
        self.num_pos = len(self.pos_indices)

        self.epoch_length = self.num_neg * 2

        logger.info(f"BalancedBagSampler initialized.")
        logger.info(f"  Positives: {self.num_pos}, Negatives: {self.num_neg}")
        logger.info(f"  Epoch Length: {self.epoch_length} (50/50 ratio)")

    def __iter__(self):
        # 1. Sample negatives (all of them, shuffled)
        neg_samples = np.random.permutation(self.neg_indices)

        # 2. Sample positives (replace=True to match number of negatives)
        if self.num_pos > 0:
            pos_samples = np.random.choice(self.pos_indices, self.num_neg, replace=True)
        else:
            # Fallback if no positives exist (e.g., debug run on subset)
            logger.warning(
                "No positive samples found in dataset. Sampling only negatives."
            )
            pos_samples = np.random.choice(self.neg_indices, self.num_neg, replace=True)

        # 3. Combine
        combined = np.concatenate([neg_samples, pos_samples])

        # 4. Shuffle the combined list to mix positives and negatives
        np.random.shuffle(combined)

        return iter(combined.tolist())

    def __len__(self):
        return self.epoch_length


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates the creation of DataLoaders for Train, Validation, and Test.
    Uses process_and_cache_data to handle metadata grouping and caching.

    Args:
        load_cached_data (bool): Whether to use existing parquet caches.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    logger.info("Preparing DataLoaders...")

    # 1. Process Metadata (Group by Bag)
    # Train
    train_df = process_and_cache_data(
        Config.TRAIN_META_PATH,
        Config.TRAIN_BAG_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Validation
    val_df = process_and_cache_data(
        Config.VAL_META_PATH,
        Config.VAL_BAG_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # Test
    test_df = process_and_cache_data(
        Config.TEST_META_PATH,
        Config.TEST_BAG_CACHE,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    # Debugging: Reduce dataset size if configured
    if Config.DEBUG:
        logger.info(
            f"DEBUG mode: Truncating datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Create Datasets
    train_dataset = MammographyBagDataset(train_df, is_train=True)
    val_dataset = MammographyBagDataset(val_df, is_train=True)
    test_dataset = MammographyBagDataset(test_df, is_train=False)

    # 3. Create Sampler for Training
    # This ensures balanced batches (50% cancer, 50% non-cancer)
    train_sampler = BalancedBagSampler(train_df)

    # 4. Create DataLoaders
    # Note: We use collate_bag_fn to handle variable number of images per bag

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_fn,
        pin_memory=True,
    )

    logger.info("DataLoaders ready.")
    return train_loader, val_loader, test_loader
