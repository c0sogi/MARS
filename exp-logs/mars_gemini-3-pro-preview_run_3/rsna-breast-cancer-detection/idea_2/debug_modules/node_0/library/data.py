import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import load_dicom_and_process


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for Breast Cancer Detection.
    Handles DICOM loading, processing, and augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transforms to apply.
            mode (str): 'train' (returns img, label) or 'test' (returns img, prediction_id).
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Load and process image (returns H, W, 1 in [0, 1])
        # Uses the utility function provided in library.utils
        img = load_dicom_and_process(
            file_path,
            img_size=Config.IMG_SIZE,
            norm_min_percentile=Config.NORM_MIN_PERCENTILE,
            norm_max_percentile=Config.NORM_MAX_PERCENTILE,
        )

        # Convert (H, W, 1) to (H, W, 3) for EfficientNet backbone
        # We repeat the single grayscale channel 3 times
        img = np.repeat(img, 3, axis=-1)

        # Albumentations expects numpy array (H, W, C)
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]
        else:
            # Fallback conversion to tensor (C, H, W)
            img = torch.from_numpy(img.transpose(2, 0, 1))

        if self.mode == "test":
            # For inference, we need the prediction_id to map predictions back to the submission file
            return img, row["prediction_id"]
        else:
            # For training/validation, return image and binary label
            # Label is returned as a float32 tensor of shape (1,) for BCEWithLogitsLoss
            label = row["cancer"] if "cancer" in row else 0.0
            return img, torch.tensor([label], dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                # Simulates variations in patient positioning
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Simulates variations in scan exposure/contrast
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Only convert to Tensor
        return A.Compose([ToTensorV2()])


def get_dataloaders(stage=1, debug=Config.DEBUG, load_cached_data=True):
    """
    Creates DataLoaders for Train (Stage 1 or 2), Validation, and Test.

    Args:
        stage (int): 1 for Balanced Sampling (Representation Learning),
                     2 for Natural Distribution (Calibration).
        debug (bool): If True, subsets data for quick testing.
        load_cached_data (bool): If True, attempts to load DataFrames from parquet cache.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """

    # -------------------------------------------------------------------------
    # 1. Load Metadata (with Caching)
    # -------------------------------------------------------------------------
    # Define cache paths based on debug state to avoid mixing full/subset data
    cache_suffix = "_debug" if debug else ""
    cache_path_train = os.path.join(
        Config.WORKING_DIR, f"train_meta{cache_suffix}.parquet"
    )
    cache_path_val = os.path.join(Config.WORKING_DIR, f"val_meta{cache_suffix}.parquet")
    cache_path_test = os.path.join(
        Config.WORKING_DIR, f"test_meta{cache_suffix}.parquet"
    )

    # Check if cache exists and loading is requested
    if (
        load_cached_data
        and os.path.exists(cache_path_train)
        and os.path.exists(cache_path_val)
        and os.path.exists(cache_path_test)
    ):
        df_train = pd.read_parquet(cache_path_train)
        df_val = pd.read_parquet(cache_path_val)
        df_test = pd.read_parquet(cache_path_test)
    else:
        # Load from original CSVs
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)

        # Apply Debug Subsetting
        if debug:
            df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
            df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
            df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

        # Save to cache for future runs
        df_train.to_parquet(cache_path_train)
        df_val.to_parquet(cache_path_val)
        df_test.to_parquet(cache_path_test)

    # -------------------------------------------------------------------------
    # 2. Prepare Datasets
    # -------------------------------------------------------------------------
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = BreastCancerDataset(
        df_train, transforms=train_transforms, mode="train"
    )
    val_dataset = BreastCancerDataset(df_val, transforms=val_transforms, mode="train")
    test_dataset = BreastCancerDataset(df_test, transforms=val_transforms, mode="test")

    # -------------------------------------------------------------------------
    # 3. Configure Samplers (Stage-dependent)
    # -------------------------------------------------------------------------
    train_sampler = None
    shuffle = True

    if stage == 1:
        # Stage 1: Balanced Sampling (1:1)
        # We use WeightedRandomSampler to oversample the minority class
        targets = df_train["cancer"].values.astype(int)
        class_counts = np.bincount(targets)

        # Safety check for debug mode where a class might be missing
        if len(class_counts) < 2:
            shuffle = True
            train_sampler = None
        else:
            # Weight = 1 / count
            class_weights = 1.0 / class_counts
            # Assign weight to each sample corresponding to its class
            sample_weights = class_weights[targets]

            train_sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(sample_weights),
                replacement=True,
            )
            # Shuffle must be False when using a sampler
            shuffle = False

    elif stage == 2:
        # Stage 2: Natural Distribution
        # We use standard shuffling to let the model learn the true prevalence (calibration)
        shuffle = True
        train_sampler = None

    # -------------------------------------------------------------------------
    # 4. Create DataLoaders
    # -------------------------------------------------------------------------
    # Determine batch size based on stage
    train_bs = Config.STAGE1_BATCH_SIZE if stage == 1 else Config.STAGE2_BATCH_SIZE
    # Validation/Test can use larger batches (no gradients)
    val_bs = Config.STAGE2_BATCH_SIZE

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to maintain stable statistics
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
