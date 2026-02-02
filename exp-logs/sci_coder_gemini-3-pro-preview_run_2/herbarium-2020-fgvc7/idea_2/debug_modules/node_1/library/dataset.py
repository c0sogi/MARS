import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed


class PlantDataset(Dataset):
    def __init__(self, df, data_root, transform=None, label_map=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            data_root (str): Root directory for images.
            transform (albumentations.Compose): Transformations to apply.
            label_map (dict): Dictionary mapping original category_id to model label index.
        """
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.label_map = label_map

        # Determine if we have labels
        self.has_labels = "category_id" in self.df.columns

        # Store classes list for inverse mapping if map is provided
        self.classes = None
        if self.label_map:
            # Create a sorted list of class IDs where index corresponds to the model label
            self.classes = sorted(
                self.label_map.keys(), key=lambda k: self.label_map[k]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in csv is relative to input dir (e.g., nybg2020/train/...)
        img_path = os.path.join(self.data_root, row["file_path"])

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (though verification script showed 0 missing)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided
            base_transform = A.Compose(
                [
                    A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                    A.Normalize(mean=Config.MEAN, std=Config.STD),
                    ToTensorV2(),
                ]
            )
            image = base_transform(image=image)["image"]

        # Handle Labels
        if self.has_labels:
            cat_id = row["category_id"]
            if self.label_map:
                label = self.label_map[cat_id]
            else:
                label = cat_id

            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test set, return image and image_id (for submission)
            image_id = row["image_id"]
            return image, torch.tensor(image_id, dtype=torch.long)


def get_transforms(data_split):
    """
    Returns albumentations transforms for the specified data split.

    Args:
        data_split (str): 'train', 'val', or 'test'.
    """
    if data_split == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        train_csv (str): Path to train metadata CSV.
        val_csv (str): Path to val metadata CSV.
        test_csv (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets the data for debugging.
        load_cached_data (bool): Whether to use cached label mapping.

    Returns:
        tuple: (train_loader, val_loader, test_loader, classes)
               classes is a list of original category_ids corresponding to model indices.
    """
    set_seed()

    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # --- Label Mapping Caching Logic ---
    cache_path = os.path.join(Config.WORKING_DIR, "label_map.npy")
    unique_cats = None

    if load_cached_data and os.path.exists(cache_path):
        try:
            unique_cats = np.load(cache_path)
            # Ensure cached map matches expected number of classes (avoids stale debug cache)
            if len(unique_cats) != Config.NUM_CLASSES:
                unique_cats = None
        except Exception:
            unique_cats = None

    if unique_cats is None:
        # Compute unique categories from training data
        unique_cats = sorted(train_df["category_id"].unique())
        unique_cats = np.array(unique_cats)
        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_path, unique_cats)

    # Create mapping: category_id -> label_index (0..N-1)
    label_map = {cat_id: idx for idx, cat_id in enumerate(unique_cats)}

    # Verify consistency with Config
    if len(unique_cats) != Config.NUM_CLASSES:
        # If debug is on, this mismatch is expected
        if not debug:
            print(
                f"Warning: Number of classes in data ({len(unique_cats)}) "
                f"does not match Config.NUM_CLASSES ({Config.NUM_CLASSES})."
            )

    # Debugging: Subset data
    if debug:
        subset_size = Config.DEBUG_SUBSET_SIZE
        train_df = train_df.iloc[:subset_size]
        val_df = val_df.iloc[:subset_size]
        test_df = test_df.iloc[:subset_size]
        print(f"DEBUG MODE: Subsetting data to {subset_size} rows.")

    # Instantiate Datasets
    train_dataset = PlantDataset(
        train_df,
        Config.INPUT_DIR,
        transform=get_transforms("train"),
        label_map=label_map,
    )

    val_dataset = PlantDataset(
        val_df, Config.INPUT_DIR, transform=get_transforms("val"), label_map=label_map
    )

    test_dataset = PlantDataset(
        test_df,
        Config.INPUT_DIR,
        transform=get_transforms("test"),
        label_map=None,  # No labels in test
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, unique_cats.tolist()
