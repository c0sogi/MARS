import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import load_case_data


def get_transforms(mode="train"):
    """
    Returns the albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'valid'.

    Returns:
        A.ReplayCompose: The transform pipeline capable of replaying parameters.
    """
    if mode == "train":
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,
                ),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class RSNADataset(Dataset):
    """
    Dataset for Training and Validation.
    Loads cached 2.5D volumes and applies volumetric-consistent augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Target columns
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load volume: (SEQ_LEN, H, W, 3)
        # We use cached data for training/validation for speed
        volume = load_case_data(study_id, image_dir, load_cached_data=True)

        # Prepare storage for augmented volume
        # Output shape: (SEQ_LEN, C, H, W)
        seq_len = volume.shape[0]
        images_list = []

        if self.transforms:
            # Apply transform to the first slice to generate parameters
            data = self.transforms(image=volume[0])
            images_list.append(data["image"])
            replay_params = data["replay"]

            # Replay the exact same transform for the rest of the sequence
            for i in range(1, seq_len):
                data = self.transforms.replay(replay_params, image=volume[i])
                images_list.append(data["image"])
        else:
            # Fallback if no transforms (should not happen based on design)
            base_tf = get_transforms("valid")
            for i in range(seq_len):
                data = base_tf(image=volume[i])
                images_list.append(data["image"])

        # Stack to create tensor: (SEQ_LEN, C, H, W)
        images = torch.stack(images_list)

        # Positional Encoding: Normalized depth (0 to 1)
        # Shape: (SEQ_LEN, 1)
        positions = torch.linspace(0, 1, seq_len).unsqueeze(1)

        # Targets
        labels = torch.tensor(row[self.target_cols].values.astype(np.float32))

        return {
            "images": images,
            "positions": positions,
            "targets": labels,
            "study_id": study_id,
        }


class TestDataset(Dataset):
    """
    Dataset for Inference.
    Loads raw DICOMs on-the-fly (no caching assumed for hidden test set).
    """

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Load volume: (SEQ_LEN, H, W, 3)
        # For inference, we force load_cached_data=False to ensure we read from the
        # actual test directories which might not be cached.
        volume = load_case_data(study_id, image_dir, load_cached_data=False)

        seq_len = volume.shape[0]
        images_list = []

        if self.transforms:
            # Apply transform (normalization) to first slice
            data = self.transforms(image=volume[0])
            images_list.append(data["image"])
            replay_params = data["replay"]

            # Replay for consistency (though usually just normalization for test)
            for i in range(1, seq_len):
                data = self.transforms.replay(replay_params, image=volume[i])
                images_list.append(data["image"])

        images = torch.stack(images_list)
        positions = torch.linspace(0, 1, seq_len).unsqueeze(1)

        return {"images": images, "positions": positions, "study_id": study_id}


def cache_dataset(df):
    """
    Iterates through the dataframe and triggers the caching mechanism
    for all studies.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    print(f"Caching {len(df)} studies to {Config.CACHE_DIR}...")

    for idx, row in df.iterrows():
        study_id = row["StudyInstanceUID"]
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # This function handles the check/compute/save logic internally
        load_case_data(study_id, image_dir, load_cached_data=True)


def get_dataloaders(debug=False):
    """
    Prepare DataLoaders for training and validation.

    Args:
        debug (bool): If True, subsamples the dataset for quick debugging.

    Returns:
        train_loader, val_loader
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG MODE: Using {len(train_df)} train and {len(val_df)} val samples.")

    # Cache Data (Pre-process)
    # We only cache the training and validation sets used in this run
    cache_dataset(train_df)
    cache_dataset(val_df)

    # Create Datasets
    train_dataset = RSNADataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )

    val_dataset = RSNADataset(val_df, transforms=get_transforms("valid"), mode="valid")

    # Create Loaders
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
        drop_last=False,
    )

    return train_loader, val_loader
