import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_label_encoder(train_df, load_cached_data=True):
    """
    Handles the creation and caching of the label encoder (mapping hotel_id to index).
    Strictly follows the caching logic requirements:
    - Checks for cached file.
    - If missing or forced reload, computes from scratch and saves.
    - Uses .npy format.
    """
    cache_dir = Config.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "label_encoder.npy")

    unique_ids = None

    # 1. Try to load cached data
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                unique_ids = np.load(cache_path)
                # Verify consistency if needed, or trust the cache
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                unique_ids = None
        else:
            pass  # Cache doesn't exist

    # 2. Compute if not loaded
    if unique_ids is None:
        # Get unique hotel IDs from training data
        unique_ids = np.sort(train_df["hotel_id"].unique())

        # Save to cache
        np.save(cache_path, unique_ids)
        print(f"Label encoder computed and saved to {cache_path}")
    else:
        print(f"Label encoder loaded from {cache_path}")

    # Create mapping dictionaries
    # id_to_idx: Maps original hotel_id to 0..N-1
    # idx_to_id: Maps 0..N-1 back to original hotel_id
    id_to_idx = {val: i for i, val in enumerate(unique_ids)}

    return unique_ids, id_to_idx


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline based on the split.

    Train: Resize(256) -> RandomCrop(224) -> HorizontalFlip -> Normalize -> ToTensor
    Val/Test: Resize(224) -> Normalize -> ToTensor
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(Config.image_size, Config.image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.image_size, Config.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    def __init__(self, df, data_root, transform=None, label_map=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            data_root (str): Root directory containing images (e.g., ./input).
            transform (albumentations.Compose): Transformations to apply.
            label_map (dict): Dictionary mapping hotel_id to integer labels.
            is_test (bool): Whether this is the test set (no targets).
        """
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.label_map = label_map
        self.is_test = is_test

        # Pre-compute full paths to avoid overhead in __getitem__
        # Metadata file_path is relative to input directory
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.hotel_ids = self.df["hotel_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata check guarantees existence)
            # Create a black image to prevent crash
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return logic
        if self.is_test:
            # For test, we might need the image ID to map predictions back,
            # but usually the DataLoader order is preserved.
            # Returning just the image is standard for inference loops that iterate sequentially.
            return image
        else:
            # Get target
            original_id = self.hotel_ids[idx]
            label = self.label_map[original_id]
            return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(load_cached_data=True):
    """
    Orchestrates the loading of metadata, creation of datasets, and dataloaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    # 2. Label Encoding
    # Only train_df is used to build the encoder to ensure we don't leak info,
    # but we must assume val/test classes are covered or handle unknowns.
    # In this competition, classes are fixed.
    # Cite debug_lesson_3: Derive Global Mappings from Full Datasets Before Subsetting
    unique_ids, id_to_idx = get_label_encoder(
        train_df, load_cached_data=load_cached_data
    )

    # Update Config with actual number of classes found
    Config.n_classes = len(unique_ids)

    # 3. Handle Debug Mode
    if Config.debug:
        print("Debug mode active: Sampling subset of data.")
        train_df = train_df.sample(
            n=min(2000, len(train_df)), random_state=Config.seed
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(500, len(val_df)), random_state=Config.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(100, len(test_df)), random_state=Config.seed
        ).reset_index(drop=True)

    # 4. Create Datasets
    train_dataset = HotelDataset(
        df=train_df,
        data_root=Config.input_root,
        transform=get_transforms("train"),
        label_map=id_to_idx,
        is_test=False,
    )

    val_dataset = HotelDataset(
        df=val_df,
        data_root=Config.input_root,
        transform=get_transforms("val"),
        label_map=id_to_idx,
        is_test=False,
    )

    test_dataset = HotelDataset(
        df=test_df,
        data_root=Config.input_root,
        transform=get_transforms("test"),
        label_map=None,
        is_test=True,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, unique_ids
