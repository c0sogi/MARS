import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from library.config import Config
from library.utils import read_dicom, get_all_file_lists


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Non-rigid transformations for medical imaging
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class MGMTDataset(Dataset):
    def __init__(self, df, file_lists_df, phase="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing BraTS21ID and targets.
            file_lists_df (pd.DataFrame): Dataframe containing lists of file paths for each subject.
            phase (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transforms to apply.
        """
        self.df = df.reset_index(drop=True)
        # Convert file lists dataframe to a dictionary for faster lookup by BraTS21ID
        self.file_lists = file_lists_df.set_index("BraTS21ID").to_dict("index")
        self.phase = phase
        self.transform = transform

        # Modalities to use as channels
        self.modalities = ["FLAIR", "T1wCE", "T2w"]

    def __len__(self):
        return len(self.df)

    def _load_slice_at_depth(self, subject_id, relative_depth):
        """
        Loads a 3-channel image (FLAIR, T1wCE, T2w) at a specific relative depth.
        """
        channels = []
        subject_files = self.file_lists.get(subject_id, {})

        for mod in self.modalities:
            files = subject_files.get(f"{mod}_files", [])

            img = None
            if len(files) > 0:
                # Calculate index based on relative depth
                num_files = len(files)
                # Ensure index is within bounds [0, num_files-1]
                idx = int(relative_depth * num_files)
                idx = min(max(idx, 0), num_files - 1)

                file_path = files[idx]
                img = read_dicom(file_path, size=Config.IMG_SIZE)

            # Handle missing files or read errors
            if img is None:
                img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)

            channels.append(img)

        # Stack to create (H, W, 3)
        # OpenCV reads are usually (H, W), so stacking creates (H, W, 3)
        img_stacked = np.stack(channels, axis=-1)
        return img_stacked

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Get target if available
        label = (
            torch.tensor(row["MGMT_value"], dtype=torch.float32)
            if "MGMT_value" in row
            else torch.tensor(-1.0)
        )

        # Determine depths to sample
        voi_start, voi_end = Config.VOI_RANGE

        if self.phase == "train":
            # Stochastic Sampling: Pick ONE random depth within VOI
            depth = random.uniform(voi_start, voi_end)

            image = self._load_slice_at_depth(subject_id, depth)

            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                # Fallback to tensor conversion
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

            return image, label, subject_id

        else:
            # Inference Ensemble: Pick N equidistant depths
            depths = np.linspace(voi_start, voi_end, Config.NUM_INFERENCE_SLICES)

            images = []
            for d in depths:
                img = self._load_slice_at_depth(subject_id, d)

                if self.transform:
                    augmented = self.transform(image=img)
                    img_tensor = augmented["image"]
                else:
                    img_tensor = (
                        torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    )

                images.append(img_tensor)

            # Stack images: (N_SLICES, C, H, W)
            images_stacked = torch.stack(images)

            return images_stacked, label, subject_id


def get_dataloaders(
    train_metadata_path=Config.TRAIN_METADATA_PATH,
    val_metadata_path=Config.VAL_METADATA_PATH,
    test_metadata_path=Config.TEST_METADATA_PATH,
    load_cached_data=True,
):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_metadata_path (str): Path to training metadata CSV.
        val_metadata_path (str): Path to validation metadata CSV.
        test_metadata_path (str): Path to test metadata CSV.
        load_cached_data (bool): Whether to use cached file lists.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    # Debugging: Subset data if DEBUG is enabled
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        # Keep test set full usually, but for strict debug maybe subset
        # df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Get File Lists (Cached)
    # We combine train and val for file list generation to ensure coverage if splitting logic changes,
    # though usually they are separate. The utility handles caching by split name.

    # Train files
    train_files_df = get_all_file_lists(
        df_train, load_cached_data=load_cached_data, split_name="train"
    )

    # Val files
    val_files_df = get_all_file_lists(
        df_val, load_cached_data=load_cached_data, split_name="val"
    )

    # Test files
    test_files_df = get_all_file_lists(
        df_test, load_cached_data=load_cached_data, split_name="test"
    )

    # 3. Create Datasets
    train_dataset = MGMTDataset(
        df_train, train_files_df, phase="train", transform=get_transforms("train")
    )

    val_dataset = MGMTDataset(
        df_val, val_files_df, phase="val", transform=get_transforms("val")
    )

    test_dataset = MGMTDataset(
        df_test, test_files_df, phase="test", transform=get_transforms("test")
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader
