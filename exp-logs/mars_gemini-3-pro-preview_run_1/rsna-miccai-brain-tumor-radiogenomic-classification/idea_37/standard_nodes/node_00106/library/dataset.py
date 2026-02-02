import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.geometry_utils import (
    process_subject_geometry,
    read_dicom_image,
    natural_sort_key,
)


class ARVSDataset(Dataset):
    """
    Aligned-Relative Volumetric Stack (ARVS) Dataset.
    Loads 9 channels per subject: 3 modalities x 3 relative depth offsets.
    """

    def __init__(self, metadata_df, geometry_df, transforms=None, mode="train"):
        """
        Args:
            metadata_df: DataFrame containing paths and labels.
            geometry_df: DataFrame containing calculated file indices for each modality/offset.
            transforms: Albumentations composition.
            mode: 'train', 'val', or 'test'.
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        # Merge geometry info into metadata for faster access
        self.data = pd.merge(self.metadata_df, geometry_df, on="BraTS21ID", how="left")
        self.transforms = transforms
        self.mode = mode

        # Cache for file lists to reduce filesystem I/O overhead if RAM permits
        # Given 220GB RAM, we could cache paths, but listing is safer for consistency.
        # We will list directories dynamically to ensure robustness.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject_id = row["BraTS21ID"]

        channels = []

        # Iterate through modalities in the order defined in Config
        for mod in Config.MODALITIES:
            # 1. Get directory path
            rel_path = row[f"{mod}_path"]
            full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

            # 2. List and sort files (Must match geometry_utils logic exactly)
            if os.path.exists(full_dir_path):
                files = glob.glob(os.path.join(full_dir_path, "*.dcm"))
                files.sort(key=lambda f: natural_sort_key(os.path.basename(f)))
            else:
                files = []

            num_files = len(files)

            # 3. Retrieve indices for this modality's offsets
            # Columns are named like "flair_idx_0", "flair_idx_1", "flair_idx_2"
            for i in range(len(Config.RELATIVE_OFFSETS)):
                col_name = f"{mod}_idx_{i}"
                file_idx = int(row[col_name])

                # Safety check
                if num_files > 0:
                    # Clamp index just in case
                    file_idx = max(0, min(file_idx, num_files - 1))
                    img_path = files[file_idx]

                    # 4. Load Image
                    img = read_dicom_image(img_path)
                    if img is None:
                        # Fallback: black image
                        img = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)
                    else:
                        # Resize to target size
                        img = cv2.resize(
                            img, Config.IMAGE_SIZE, interpolation=cv2.INTER_AREA
                        )
                        img = img.astype(np.float32)

                        # 5. Independent Min-Max Normalization
                        min_val = img.min()
                        max_val = img.max()
                        if max_val > min_val:
                            img = (img - min_val) / (max_val - min_val)
                        else:
                            img = np.zeros_like(img)
                else:
                    # Missing modality fallback
                    img = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)

                channels.append(img)

        # Stack channels: (H, W, 9)
        # Order: [Mod1_Offset1, Mod1_Offset2, Mod1_Offset3, Mod2_Offset1, ...]
        image_stack = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=image_stack)
            image_stack = augmented["image"]
        else:
            # Default to tensor conversion if no transforms provided
            image_stack = torch.from_numpy(image_stack.transpose(2, 0, 1))

        # Return logic based on mode
        if self.mode in ["train", "val"]:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image_stack, label
        else:
            return image_stack, str(subject_id)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Strictly excludes translation and scaling to preserve CoM alignment.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                # Spatial Augmentations (Coupled across channels)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                # Elastic & Grid (Non-rigid deformations)
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    ],
                    p=0.3,
                ),
                # Normalization is handled manually in Dataset per channel
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                ToTensorV2(),
            ]
        )


def get_dataloader(split, batch_size=None, num_workers=None, load_cached_geometry=True):
    """
    Factory function to create DataLoaders.

    Args:
        split: 'train', 'val', or 'test'.
        batch_size: Batch size (defaults to Config).
        num_workers: Number of workers (defaults to Config).
        load_cached_geometry: Whether to use cached geometry parquet files.

    Returns:
        DataLoader instance.
    """
    # Defaults
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    # 1. Load Metadata
    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        mode = "train"
        shuffle = True
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        mode = "val"
        shuffle = False
    elif split == "test":
        metadata_path = Config.TEST_METADATA_PATH
        mode = "test"
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # 2. Process Geometry (Calculate CoM and Offsets)
    # This handles caching internally within the library function
    print(f"Preparing geometry for {split} set...")
    geometry_df = process_subject_geometry(df, load_cached_data=load_cached_geometry)

    # 3. Create Dataset
    transforms = get_transforms(mode=mode)
    dataset = ARVSDataset(
        metadata_df=df, geometry_df=geometry_df, transforms=transforms, mode=mode
    )

    # 4. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        drop_last=(split == "train"),  # Drop last incomplete batch only during training
    )

    return loader
