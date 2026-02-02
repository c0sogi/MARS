import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from library.config import Config


def get_transforms(img_size, mode="train"):
    """
    Returns the Albumentations transformations for the specified mode.

    Args:
        img_size (int): The target image size (height and width).
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed Albumentations transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # CoarseDropout for distributed feature learning
                (
                    A.CoarseDropout(**Config.COARSE_DROPOUT_PARAMS)
                    if Config.USE_COARSE_DROPOUT
                    else A.NoOp()
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test (TTA is handled in inference loop usually, but here we just do standard resize/norm)
        # Note: If TTA is required inside the dataset, it would be a separate mode, but standard practice
        # is to apply TTA on the model forward pass or wrap the dataset.
        # Here we provide standard deterministic transforms for validation/test.
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )


def get_dataframes(load_cached_data=True):
    """
    Loads metadata, processes labels for the 2-class decomposition,
    creates a 5-fold split, and handles caching.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (full_train_df, test_df)
            full_train_df contains 'fold', 'target_rust', 'target_scab' columns.
    """
    train_cache_path = os.path.join(Config.IDEA_DIR, "full_train_processed.parquet")
    test_cache_path = os.path.join(Config.IDEA_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        try:
            full_train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return full_train_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Process from scratch
    # Load provided metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Concatenate train and val to create a full development set for CV
    full_train_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Generate Binary Targets for Multi-Label Decomposition
    # Rust Target: 1 if 'rust' or 'multiple_diseases' is 1
    # Scab Target: 1 if 'scab' or 'multiple_diseases' is 1
    full_train_df["target_rust"] = full_train_df.apply(
        lambda x: 1.0 if (x["rust"] == 1 or x["multiple_diseases"] == 1) else 0.0,
        axis=1,
    )
    full_train_df["target_scab"] = full_train_df.apply(
        lambda x: 1.0 if (x["scab"] == 1 or x["multiple_diseases"] == 1) else 0.0,
        axis=1,
    )

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    full_train_df["fold"] = -1

    # We use 'stratify_label' provided in metadata for stratification
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["stratify_label"])
    ):
        full_train_df.loc[val_idx, "fold"] = fold

    # Construct full paths for images
    # The metadata contains 'file_path' relative to INPUT_DIR
    # We update it to be the absolute path or relative to current working directory correctly
    full_train_df["abs_file_path"] = full_train_df["file_path"].apply(
        lambda x: os.path.join(Config.INPUT_DIR, x)
    )
    test_df["abs_file_path"] = test_df["file_path"].apply(
        lambda x: os.path.join(Config.INPUT_DIR, x)
    )

    # Debug Mode: Subset data
    if Config.DEBUG:
        full_train_df = full_train_df.sample(
            n=100, random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(n=20, random_state=Config.SEED).reset_index(drop=True)

    # 3. Save to cache
    os.makedirs(Config.IDEA_DIR, exist_ok=True)
    full_train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return full_train_df, test_df


class AppleDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-fetch paths to avoid overhead
        self.file_paths = self.df["abs_file_path"].values
        self.image_ids = self.df["image_id"].values

        if self.mode != "test":
            self.targets_rust = self.df["target_rust"].values.astype(np.float32)
            self.targets_scab = self.df["target_scab"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)

        if image is None:
            # Fallback for missing images (should not happen based on EDA)
            # Create a black image of expected size
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.mode == "test":
            return image, self.image_ids[idx]
        else:
            # Return stacked targets: [rust_prob, scab_prob]
            target = torch.tensor(
                [self.targets_rust[idx], self.targets_scab[idx]], dtype=torch.float32
            )
            return image, target


def get_loaders(fold, model_config, load_cached_data=True):
    """
    Creates DataLoaders for a specific fold and model configuration.

    Args:
        fold (int): The current fold index (0 to NUM_FOLDS-1).
        model_config (dict): Configuration dictionary for the specific model
                             (must contain 'img_size' and 'batch_size').
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    full_train_df, test_df = get_dataframes(load_cached_data=load_cached_data)

    # Split into Train and Validation based on fold
    train_df = full_train_df[full_train_df["fold"] != fold].reset_index(drop=True)
    val_df = full_train_df[full_train_df["fold"] == fold].reset_index(drop=True)

    img_size = model_config["img_size"]
    batch_size = model_config["batch_size"]

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms(img_size, mode="train"), mode="train"
    )
    val_dataset = AppleDataset(
        val_df, transforms=get_transforms(img_size, mode="val"), mode="val"
    )
    test_dataset = AppleDataset(
        test_df, transforms=get_transforms(img_size, mode="test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
