import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from library.config import Config
from library.data_processing import load_dicom, crop_breast_roi, preprocess_image


def prepare_bag_dataframe(csv_path, split_name, load_cached_data=True):
    """
    Groups metadata by patient_id and laterality to form bags.
    Implements caching using Parquet to avoid re-processing.
    """
    cache_file = f"{split_name}_bag_cache.parquet"
    cache_path = os.path.join(Config.WORKING_DIR, cache_file)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split_name} data from {cache_path}")
        try:
            df_grouped = pd.read_parquet(cache_path)
            return df_grouped
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing.")

    # 2. Process from scratch
    print(f"Processing {split_name} metadata from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    if "cancer" not in df.columns:
        df["cancer"] = 0  # Default for test set

    # Create prediction_id if not present (for training data consistency)
    if "prediction_id" not in df.columns:
        df["prediction_id"] = df["patient_id"].astype(str) + "_" + df["laterality"]

    # Group by bag (Patient + Laterality)
    # We aggregate file_paths into a list, and take the max of cancer (if any image is cancer, bag is cancer)
    # We also keep the first prediction_id
    group_cols = ["patient_id", "laterality"]

    # Define aggregation dictionary
    agg_dict = {
        "file_path": list,
        "cancer": "max",
        "prediction_id": "first",
        "site_id": "first",  # Keep some metadata
    }

    # Perform grouping
    df_grouped = df.groupby(group_cols).agg(agg_dict).reset_index()

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    try:
        df_grouped.to_parquet(cache_path, index=False)
        print(f"Saved {split_name} bag cache to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df_grouped


class BreastBagDataset(Dataset):
    """
    Dataset that yields a 'bag' of images for a single breast (Patient + Laterality).
    """

    def __init__(self, df, transforms=None, debug=False):
        self.df = df
        self.transforms = transforms

        if debug:
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)

        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_paths = row["file_path"]
        label = row["cancer"]
        pred_id = row["prediction_id"]

        images_list = []

        for rel_path in file_paths:
            full_path = os.path.join(self.input_dir, rel_path)

            # 1. Load DICOM
            img = load_dicom(full_path)
            if img is None:
                continue

            # 2. Crop ROI
            img = crop_breast_roi(img)

            # 3. Preprocess (Resize, Normalize to [0,1] float32, 3 channels)
            # preprocess_image returns (H, W, 3)
            img = preprocess_image(img, Config.IMG_HEIGHT, Config.IMG_WIDTH)

            # 4. Augmentations
            if self.transforms:
                # Albumentations expects image key
                augmented = self.transforms(image=img)
                img_tensor = augmented["image"]
            else:
                # Fallback to ToTensor logic if no transforms provided
                # Convert (H, W, 3) -> (3, H, W)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

            images_list.append(img_tensor)

        if len(images_list) == 0:
            # Handle case where all images failed to load
            # Return a zero tensor bag of shape (1, 3, H, W)
            # This prevents batch collation failure
            images_list.append(
                torch.zeros(
                    (3, Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=torch.float32
                )
            )

        # Stack images: (N, 3, H, W) where N is number of views
        bag_tensor = torch.stack(images_list, dim=0)

        return bag_tensor, torch.tensor(label, dtype=torch.float32), pred_id


def collate_bag_batch(batch):
    """
    Custom collate function for MIL batches.
    Since each bag has a different number of images (N), we cannot stack them
    into a single (B, N, C, H, W) tensor directly.

    Returns:
        images (List[Tensor]): List of B tensors, each shape (N_i, 3, H, W)
        labels (Tensor): Shape (B,)
        ids (List[str]): List of prediction IDs
    """
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    ids = [item[2] for item in batch]

    return images, labels, ids


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),  # Converts to (C, H, W)
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_balanced_sampler(df):
    """
    Creates a WeightedRandomSampler to balance positive and negative bags.
    """
    labels = df["cancer"].values
    class_counts = np.bincount(labels.astype(int))

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Calculate weights: inverse of frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[labels.astype(int)]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(df), replacement=True
    )
    return sampler


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create DataLoaders for Train, Val, and Test.
    """
    # 1. Prepare DataFrames
    df_train = prepare_bag_dataframe(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    df_val = prepare_bag_dataframe(Config.VAL_METADATA_PATH, "val", load_cached_data)
    df_test = prepare_bag_dataframe(Config.TEST_METADATA_PATH, "test", load_cached_data)

    # 2. Define Transforms
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # 3. Create Datasets
    train_dataset = BreastBagDataset(
        df_train, transforms=train_transforms, debug=Config.DEBUG
    )
    val_dataset = BreastBagDataset(
        df_val, transforms=val_transforms, debug=Config.DEBUG
    )
    test_dataset = BreastBagDataset(
        df_test, transforms=val_transforms, debug=Config.DEBUG
    )

    # 4. Create Sampler for Training
    train_sampler = get_balanced_sampler(df_train) if not Config.DEBUG else None
    shuffle_train = train_sampler is None

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=shuffle_train,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_batch,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_bag_batch,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
