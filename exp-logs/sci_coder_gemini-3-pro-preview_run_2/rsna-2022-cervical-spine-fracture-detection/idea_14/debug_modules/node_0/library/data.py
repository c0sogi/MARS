import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything(Config.SEED)


def get_transforms(mode="train"):
    """
    Returns the albumentations transformation pipeline.
    Uses ReplayCompose to ensure consistent geometric transforms across the Z-axis sequence.
    """
    if mode == "train":
        return A.ReplayCompose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def cache_study_paths(metadata_df, root_dir, cache_name, load_cached_data=True):
    """
    Scans directories to map StudyInstanceUID to sorted list of file paths.
    Caches the result to a parquet file to speed up subsequent loads.
    """
    cache_dir = Config.OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached paths from {cache_path}...")
        df_paths = pd.read_parquet(cache_path)
        # Convert back to dictionary: {StudyUID: [path1, path2, ...]}
        # We assume the parquet is sorted by SliceNum during creation
        path_map = (
            df_paths.groupby("StudyInstanceUID")["FilePath"].apply(list).to_dict()
        )
        return path_map

    print(f"Scanning directories in {root_dir} to build path cache...")

    data_records = []
    study_uids = metadata_df["StudyInstanceUID"].unique()

    # We use the 'image_path' from metadata to locate the folder
    # metadata image_path is relative to input root, e.g. "train_images/1.2.3..."
    # We need to construct the full path.

    # Create a lookup for relative path to save time
    study_to_relpath = dict(
        zip(metadata_df["StudyInstanceUID"], metadata_df["image_path"])
    )

    for uid in study_uids:
        rel_path = study_to_relpath.get(uid)
        if not rel_path:
            continue

        full_dir_path = os.path.join(Config.DATA_ROOT, rel_path)

        if not os.path.exists(full_dir_path):
            continue

        # List all dcm files
        # We use os.scandir for speed
        files = []
        try:
            with os.scandir(full_dir_path) as it:
                for entry in it:
                    if entry.name.endswith(".dcm") and entry.is_file():
                        files.append(entry.name)
        except OSError:
            continue

        # Sort by slice number (filename is usually "1.dcm", "10.dcm", etc.)
        # Extract integer from filename
        try:
            # Create tuples (slice_num, full_path)
            sorted_files = []
            for f in files:
                slice_num = int(os.path.splitext(f)[0])
                full_file_path = os.path.join(full_dir_path, f)
                sorted_files.append((slice_num, full_file_path))

            # Sort
            sorted_files.sort(key=lambda x: x[0])

            # Add to records
            for s_num, f_path in sorted_files:
                data_records.append(
                    {"StudyInstanceUID": uid, "FilePath": f_path, "SliceNum": s_num}
                )
        except ValueError:
            # Fallback if filenames are not integers
            continue

    # Create DataFrame and Save
    df_paths = pd.DataFrame(data_records)
    if not df_paths.empty:
        df_paths.to_parquet(cache_path, index=False)

    path_map = df_paths.groupby("StudyInstanceUID")["FilePath"].apply(list).to_dict()
    return path_map


def load_dicom_slice(path):
    """
    Reads a DICOM file and normalizes it to 0-255 uint8.
    Uses percentile clipping for robustness against artifacts.
    """
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Robust Min-Max Scaling with Percentile Clipping
        # This handles extreme outliers often found in raw CT data
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)

        img = np.clip(img, p1, p99)

        # Avoid division by zero
        denom = p99 - p1
        if denom == 0:
            denom = 1

        img = (img - p1) / denom
        img = (img * 255).astype(np.uint8)

        return img
    except Exception as e:
        # Return black image on failure
        return np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)


class CervicalSpineDataset(Dataset):
    def __init__(self, metadata_df, path_map, transform=None, seq_len=Config.SEQ_LEN):
        self.metadata = metadata_df.reset_index(drop=True)
        self.path_map = path_map
        self.transform = transform
        self.seq_len = seq_len

        # Targets
        self.labels = (
            self.metadata[Config.TARGET_COLS].values.astype(np.float32)
            if "patient_overall" in self.metadata.columns
            else None
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Get all slice paths for this study
        paths = self.path_map.get(uid, [])

        # Handle missing data
        if len(paths) == 0:
            # Return zero tensor if no images found
            return torch.zeros(
                (self.seq_len, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
            ), (
                torch.tensor(self.labels[idx])
                if self.labels is not None
                else torch.zeros(len(Config.TARGET_COLS))
            )

        num_slices = len(paths)

        # Uniform sampling of indices
        # We want exactly seq_len indices distributed across the volume
        if num_slices >= self.seq_len:
            indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)
        else:
            # If fewer slices than seq_len, we must repeat
            indices = np.linspace(0, num_slices - 1, self.seq_len).astype(int)

        # Prepare sequence container
        # Shape: (Seq, Channels, H, W) -> (96, 3, 384, 384)
        sequence_imgs = []

        # Volumetric Augmentation Strategy:
        # We capture the augmentation parameters from the first slice and replay them for the rest.
        replay_data = None

        for i, center_idx in enumerate(indices):
            # 2.5D Stacking: (z-1, z, z+1)
            # Clamp indices to valid range
            idx_prev = max(0, center_idx - 1)
            idx_next = min(num_slices - 1, center_idx + 1)

            path_prev = paths[idx_prev]
            path_curr = paths[center_idx]
            path_next = paths[idx_next]

            # Load images
            img_prev = load_dicom_slice(path_prev)
            img_curr = load_dicom_slice(path_curr)
            img_next = load_dicom_slice(path_next)

            # Stack to (H, W, 3)
            img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

            # Apply Augmentation
            if self.transform:
                if i == 0:
                    # First slice: Apply transform and store parameters (replay_data)
                    augmented = self.transform(image=img_stack)
                    sequence_imgs.append(augmented["image"])
                    # If the transform is ReplayCompose, it populates 'replay' in the result
                    if "replay" in augmented:
                        replay_data = augmented["replay"]
                else:
                    # Subsequent slices: Replay the exact same transform
                    if replay_data:
                        augmented = A.ReplayCompose.replay(replay_data, image=img_stack)
                        sequence_imgs.append(augmented["image"])
                    else:
                        # Fallback if not ReplayCompose (e.g. just ToTensor)
                        augmented = self.transform(image=img_stack)
                        sequence_imgs.append(augmented["image"])
            else:
                # No transform (should not happen given get_transforms)
                img_tensor = (
                    torch.from_numpy(img_stack.transpose(2, 0, 1)).float() / 255.0
                )
                sequence_imgs.append(img_tensor)

        # Stack sequence: (Seq, C, H, W)
        sequence_tensor = torch.stack(sequence_imgs)

        if self.labels is not None:
            target = torch.tensor(self.labels[idx])
            return sequence_tensor, target
        else:
            return sequence_tensor


def get_dataloaders(load_cached_data=True):
    """
    Factory function to create training and validation dataloaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if Config.DEBUG:
        train_df = train_df.head(Config.BATCH_SIZE * 2)
        val_df = val_df.head(Config.BATCH_SIZE * 2)
        print(
            f"DEBUG Mode: Reduced train size to {len(train_df)} and val size to {len(val_df)}"
        )

    # Cache Paths (Train and Val images are in the same root usually, or split)
    # The metadata 'image_path' points to the correct folder.
    # We scan the entire train_images folder once.
    # Note: Config.TRAIN_IMAGES_DIR is defined.

    # We combine train and val dfs to build the cache for all necessary studies
    combined_meta = pd.concat([train_df, val_df], ignore_index=True)

    # Build/Load Cache
    path_map = cache_study_paths(
        combined_meta,
        Config.DATA_ROOT,
        "train_paths_cache",
        load_cached_data=load_cached_data,
    )

    # Datasets
    train_dataset = CervicalSpineDataset(
        train_df, path_map, transform=get_transforms("train")
    )

    val_dataset = CervicalSpineDataset(
        val_df, path_map, transform=get_transforms("val")
    )

    # DataLoaders
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


def get_test_dataloader(load_cached_data=True):
    """
    Factory function for the test set dataloader.
    """
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Build/Load Cache for Test
    path_map = cache_study_paths(
        test_df, Config.DATA_ROOT, "test_paths_cache", load_cached_data=load_cached_data
    )

    test_dataset = CervicalSpineDataset(
        test_df, path_map, transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test usually processes one by one or small batch
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
