import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import read_and_normalize_dicom


def get_transforms(phase):
    """
    Returns the Albumentations transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composed transformations.
    """
    if phase == "train":
        return A.Compose(
            [
                # Non-rigid augmentations as per SIL-Net design
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                # Ensure resizing happens if not already handled perfectly by utils
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE), ToTensorV2()]
        )


def get_sorted_files(dir_path):
    """
    Returns a sorted list of DICOM files in a directory.
    """
    if not os.path.exists(dir_path):
        return []
    # Sort by filename (which usually contains instance number)
    # Using simple string sort or extracting number if needed.
    # Standard BraTS/RSNA filenames are like Image-1.dcm, Image-10.dcm.
    # We sort by the integer value in the filename to ensure correct depth order.
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]

    def extract_number(filename):
        # Extract number from 'Image-123.dcm'
        try:
            # Remove extension
            name = os.path.splitext(filename)[0]
            # Split by '-' or just take digits
            num = "".join(filter(str.isdigit, name))
            return int(num) if num else 0
        except:
            return 0

    files.sort(key=extract_number)
    return files


def generate_instance_metadata(df_metadata, phase, load_cached_data=True):
    """
    Generates instance-level metadata by expanding subject-level metadata.
    Selects slices at offsets [-2, 0, +2] from the median index for each modality independently.

    Args:
        df_metadata (pd.DataFrame): Subject-level metadata.
        phase (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        pd.DataFrame: Instance-level metadata.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"cached_instances_{phase}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading cached instance metadata for {phase} from {cache_path}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating...")

    print(f"Generating instance metadata for {phase}...")

    instance_rows = []

    # Process each subject
    for idx, row in df_metadata.iterrows():
        sid = row["BraTS21ID"]

        # Get paths (relative to input dir, so we join with INPUT_DIR)
        flair_dir = os.path.join(Config.INPUT_DIR, row["flair_path"])
        t1wce_dir = os.path.join(Config.INPUT_DIR, row["t1wce_path"])
        t2w_dir = os.path.join(Config.INPUT_DIR, row["t2w_path"])

        # Get sorted file lists independently
        flair_files = get_sorted_files(flair_dir)
        t1wce_files = get_sorted_files(t1wce_dir)
        t2w_files = get_sorted_files(t2w_dir)

        # Skip if any modality is missing files
        if not flair_files or not t1wce_files or not t2w_files:
            continue

        # Determine lengths and median indices
        len_flair = len(flair_files)
        len_t1wce = len(t1wce_files)
        len_t2w = len(t2w_files)

        mid_flair = len_flair // 2
        mid_t1wce = len_t1wce // 2
        mid_t2w = len_t2w // 2

        # Generate instances for defined offsets
        for offset in Config.SLICE_OFFSETS:
            # Calculate indices with clamping
            idx_flair = max(0, min(len_flair - 1, mid_flair + offset))
            idx_t1wce = max(0, min(len_t1wce - 1, mid_t1wce + offset))
            idx_t2w = max(0, min(len_t2w - 1, mid_t2w + offset))

            # Construct full file paths
            # Note: We store paths relative to INPUT_DIR to keep dataframe clean,
            # or full paths. Let's store full paths for easier loading in Dataset.
            inst_data = {
                "BraTS21ID": sid,
                "flair_file": os.path.join(flair_dir, flair_files[idx_flair]),
                "t1wce_file": os.path.join(t1wce_dir, t1wce_files[idx_t1wce]),
                "t2w_file": os.path.join(t2w_dir, t2w_files[idx_t2w]),
                "instance_offset": offset,
            }

            if "MGMT_value" in row:
                inst_data["MGMT_value"] = row["MGMT_value"]

            instance_rows.append(inst_data)

    df_instances = pd.DataFrame(instance_rows)

    # Save to cache
    try:
        df_instances.to_parquet(cache_path)
        print(f"Saved instance metadata to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return df_instances


class BrainTumorDataset(Dataset):
    """
    Dataset class for SIL-Net.
    Loads 3 independent channels (FLAIR, T1wCE, T2w) for a specific instance.
    """

    def __init__(self, df, phase, transform=None):
        self.df = df
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load images using the utility function (returns float32 [0, 1])
        flair = read_and_normalize_dicom(row["flair_file"])
        t1wce = read_and_normalize_dicom(row["t1wce_file"])
        t2w = read_and_normalize_dicom(row["t2w_file"])

        # Stack channels: (H, W, 3)
        image = np.stack([flair, t1wce, t2w], axis=-1)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen based on get_transforms)
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return dict or tuple depending on phase
        if self.phase == "test":
            return image, row["BraTS21ID"]
        else:
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image, label


def get_dataloader(
    df, phase, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Factory function to create DataLoaders.
    """
    transforms = get_transforms(phase)
    dataset = BrainTumorDataset(df, phase, transform=transforms)

    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(
            phase == "train"
        ),  # Drop last incomplete batch in training to maintain batch stats
    )

    return loader
