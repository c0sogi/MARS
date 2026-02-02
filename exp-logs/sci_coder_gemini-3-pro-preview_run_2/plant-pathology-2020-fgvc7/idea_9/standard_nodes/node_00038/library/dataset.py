import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything


def get_transforms(data_type, image_size):
    """
    Returns the Albumentations transform pipeline for the specified data type.

    Args:
        data_type (str): 'train', 'valid', or 'test'.
        image_size (int): The spatial resolution for resizing.

    Returns:
        A.Compose: The transform pipeline.
    """
    if data_type == "train":
        # Map Config parameters to Albumentations 2.x API for CoarseDropout
        cd_params = Config.coarse_dropout_params
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.CoarseDropout(
                    num_holes_range=(cd_params["min_holes"], cd_params["max_holes"]),
                    hole_height_range=(
                        cd_params["min_height"],
                        cd_params["max_height"],
                    ),
                    hole_width_range=(cd_params["min_width"], cd_params["max_width"]),
                    fill_value=cd_params["fill_value"],
                    p=cd_params["p"],
                ),
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
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class AppleDataset(Dataset):
    """
    PyTorch Dataset for Apple Disease Detection.
    """

    def __init__(self, df, transforms=None, debug=False):
        self.df = df
        self.transforms = transforms
        self.debug = debug

        # If debug is enabled, reduce dataset size
        if self.debug:
            self.df = self.df.sample(
                n=min(len(self.df), 50), random_state=Config.seed
            ).reset_index(drop=True)

        self.image_ids = self.df["image_id"].values
        self.file_paths = self.df["full_path"].values

        # Check if targets are present (Training mode)
        self.has_targets = all(col in self.df.columns for col in Config.target_columns)
        if self.has_targets:
            self.targets = self.df[Config.target_columns].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_path = self.file_paths[idx]

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Prepare targets
        if self.has_targets:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, target, image_id
        else:
            # Return dummy target for test set to keep signature consistent or just image/id
            # Returning dummy target allows using same loop structure if needed,
            # but usually test loop ignores targets.
            dummy_target = torch.tensor(
                np.zeros(len(Config.target_columns)), dtype=torch.float32
            )
            return image, dummy_target, image_id


def prepare_data(load_cached_data=True):
    """
    Loads metadata, processes targets, and handles caching.
    Returns the full training dataframe and the test dataframe.
    """
    os.makedirs(Config.working_dir, exist_ok=True)

    train_cache_path = os.path.join(Config.working_dir, "full_train_processed.parquet")
    test_cache_path = os.path.join(Config.working_dir, "test_processed.parquet")

    # --- Process Training Data ---
    if load_cached_data and os.path.exists(train_cache_path):
        full_train_df = pd.read_parquet(train_cache_path)
    else:
        # Load metadata
        if not os.path.exists(Config.train_metadata_path) or not os.path.exists(
            Config.val_metadata_path
        ):
            raise FileNotFoundError("Metadata files not found.")

        train_meta = pd.read_csv(Config.train_metadata_path)
        val_meta = pd.read_csv(Config.val_metadata_path)

        # Combine to form full training set for CV
        full_train_df = pd.concat([train_meta, val_meta], ignore_index=True)

        # Construct full paths
        full_train_df["full_path"] = full_train_df["file_path"].apply(
            lambda x: os.path.join(Config.input_dir, x)
        )

        # Create Binary Targets
        # Logic: Target is 1 if specific disease is present OR multiple_diseases is present
        # Metadata columns: 'rust', 'scab', 'multiple_diseases' are one-hot (0 or 1)
        full_train_df["rust"] = (
            full_train_df["rust"] + full_train_df["multiple_diseases"]
        )
        full_train_df["scab"] = (
            full_train_df["scab"] + full_train_df["multiple_diseases"]
        )

        # Clip to 1.0 just in case (though logical OR on 0/1 is sufficient)
        full_train_df["rust"] = full_train_df["rust"].clip(upper=1.0)
        full_train_df["scab"] = full_train_df["scab"].clip(upper=1.0)

        # Save to cache
        full_train_df.to_parquet(train_cache_path, index=False)

    # --- Process Test Data ---
    if load_cached_data and os.path.exists(test_cache_path):
        test_df = pd.read_parquet(test_cache_path)
    else:
        if not os.path.exists(Config.test_metadata_path):
            raise FileNotFoundError("Test metadata file not found.")

        test_df = pd.read_csv(Config.test_metadata_path)
        test_df["full_path"] = test_df["file_path"].apply(
            lambda x: os.path.join(Config.input_dir, x)
        )
        test_df.to_parquet(test_cache_path, index=False)

    return full_train_df, test_df


def get_dataloaders(
    fold, image_size, batch_size=Config.batch_size, load_cached_data=True
):
    """
    Creates DataLoaders for a specific fold and image size.

    Args:
        fold (int): The fold index (0 to num_folds-1).
        image_size (int): Image resolution.
        batch_size (int): Batch size.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.seed)

    # Load DataFrames
    full_train_df, test_df = prepare_data(load_cached_data=load_cached_data)

    # Stratified K-Fold Split
    # We split based on 'stratify_label' which is present in the metadata
    skf = StratifiedKFold(
        n_splits=Config.num_folds, shuffle=True, random_state=Config.seed
    )

    # Create fold column
    full_train_df["fold"] = -1
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(full_train_df, full_train_df["stratify_label"])
    ):
        full_train_df.loc[val_idx, "fold"] = fold_idx

    # Select data for current fold
    train_df = full_train_df[full_train_df["fold"] != fold].reset_index(drop=True)
    val_df = full_train_df[full_train_df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = AppleDataset(
        train_df, transforms=get_transforms("train", image_size), debug=Config.debug
    )

    val_dataset = AppleDataset(
        val_df, transforms=get_transforms("valid", image_size), debug=Config.debug
    )

    test_dataset = AppleDataset(
        test_df, transforms=get_transforms("test", image_size), debug=Config.debug
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
