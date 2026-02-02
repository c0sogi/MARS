import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Tuple, Optional

from library.config import Config
from library.utils import seed_everything


def load_dicom(path: str, size: Tuple[int, int]) -> np.ndarray:
    """
    Reads a DICOM file, applies bone windowing, normalizes, and resizes.

    Bone Window: Center (WL) = 500, Width (WW) = 2000
    Range: [-500, 1500] -> [0, 1]
    """
    try:
        dicom = pydicom.dcmread(path, stop_before_pixels=False)
        data = dicom.pixel_array.astype(np.float32)

        # Apply RescaleSlope and RescaleIntercept if present
        slope = getattr(dicom, "RescaleSlope", 1.0)
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        data = data * slope + intercept

        # Bone Windowing
        window_center = 500
        window_width = 2000
        min_value = window_center - (window_width / 2)
        max_value = window_center + (window_width / 2)

        data = np.clip(data, min_value, max_value)
        data = (data - min_value) / (max_value - min_value)

        # Resize
        if data.shape != size:
            data = cv2.resize(data, size, interpolation=cv2.INTER_LINEAR)

        return data
    except Exception as e:
        # Fallback for corrupt or missing files: return black image
        # print(f"Warning: Failed to load {path}: {e}")
        return np.zeros(size, dtype=np.float32)


def get_study_paths(
    metadata_df: pd.DataFrame, cache_path: str, load_cached_data: bool = True
) -> Dict[str, List[str]]:
    """
    Retrieves and sorts DICOM file paths for each study.
    Uses Parquet caching to speed up subsequent runs.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert DataFrame back to dict
            study_map = cache_df.set_index("StudyInstanceUID")["paths"].to_dict()
            # Ensure paths are lists (parquet might store as array)
            study_map = {k: list(v) for k, v in study_map.items()}
            return study_map
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Recomputing...")

    # 2. Compute from scratch
    study_map = {}

    # Ensure working directory exists for cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    study_uids = metadata_df["StudyInstanceUID"].unique()

    # We need to map StudyInstanceUID to the actual directory path
    # The metadata contains 'image_path' which is relative to input dir
    # We create a lookup for path
    path_lookup = metadata_df.set_index("StudyInstanceUID")["image_path"].to_dict()

    for uid in study_uids:
        rel_path = path_lookup.get(uid)
        if not rel_path:
            continue

        full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(full_dir_path):
            study_map[uid] = []
            continue

        # List all files
        try:
            files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
            # Sort by instance number (filename integer)
            # e.g., '10.dcm' -> 10
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            full_file_paths = [os.path.join(full_dir_path, f) for f in files]
            study_map[uid] = full_file_paths
        except Exception:
            study_map[uid] = []

    # 3. Save to cache
    try:
        # Convert dict to DataFrame for parquet storage
        cache_data = [{"StudyInstanceUID": k, "paths": v} for k, v in study_map.items()]
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    return study_map


class RSNADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        study_paths: Dict[str, List[str]],
        transform: Optional[A.ReplayCompose] = None,
        phase: str = "train",
    ):
        self.df = df.reset_index(drop=True)
        self.study_paths = study_paths
        self.transform = transform
        self.phase = phase
        self.seq_length = Config.SEQ_LENGTH
        self.image_size = Config.IMAGE_SIZE

        # Pre-check valid studies (must have at least 1 image)
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            uid = row["StudyInstanceUID"]
            if uid in self.study_paths and len(self.study_paths[uid]) > 0:
                self.valid_indices.append(idx)

        # If debugging, slice the dataset
        if Config.DEBUG:
            self.valid_indices = self.valid_indices[: Config.DEBUG_SAMPLE_SIZE]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map logical index to dataframe index
        df_idx = self.valid_indices[idx]
        row = self.df.iloc[df_idx]
        uid = row["StudyInstanceUID"]
        paths = self.study_paths[uid]
        num_slices = len(paths)

        # Uniform sampling of indices
        # We want exactly SEQ_LENGTH slices
        if num_slices >= self.seq_length:
            indices = np.linspace(0, num_slices - 1, self.seq_length)
        else:
            # If fewer slices than sequence length, we must interpolate/duplicate
            # But usually CT scans have > 100 slices.
            # We use linspace which handles this by repeating indices if needed (though unlikely with linspace on small range)
            # Actually linspace on small range [0, 10] with num=96 gives fractional steps.
            # We round to nearest int.
            indices = np.linspace(0, num_slices - 1, self.seq_length)

        indices = np.round(indices).astype(int)

        # Load images
        image_stack = []

        # We need to apply consistent augmentation across the sequence
        # Generate replay parameters for the first image, then apply to all
        replay_data = None

        for i, slice_idx in enumerate(indices):
            # 2.5D Stacking: (z-1, z, z+1)
            # Handle boundary conditions by clamping
            z_indices = [
                max(0, slice_idx - 1),
                slice_idx,
                min(num_slices - 1, slice_idx + 1),
            ]

            channels = []
            for z in z_indices:
                path = paths[z]
                img = load_dicom(path, self.image_size)
                channels.append(img)

            # Stack to (H, W, 3)
            img_25d = np.stack(channels, axis=-1)

            # Apply transforms
            if self.transform:
                if i == 0:
                    # First slice: compute params and store replay
                    res = self.transform(image=img_25d)
                    img_25d = res["image"]
                    replay_data = res["replay"]
                else:
                    # Subsequent slices: replay params
                    res = A.ReplayCompose.replay(replay_data, image=img_25d)
                    img_25d = res["image"]
            else:
                # Just ToTensor if no transform provided (though usually ToTensor is part of transform)
                # If transform is None, we assume manual ToTensor or raw numpy
                # But we should ensure output is tensor.
                # Assuming transform handles ToTensorV2.
                pass

            image_stack.append(img_25d)

        # Stack sequence: (Seq, C, H, W)
        # Albumentations ToTensorV2 converts (H, W, C) -> (C, H, W)
        # So image_stack is a list of (C, H, W) tensors
        if torch.is_tensor(image_stack[0]):
            images = torch.stack(image_stack)  # (Seq, C, H, W)
        else:
            # Fallback if transform didn't convert to tensor
            images = np.stack(image_stack)
            images = torch.from_numpy(images).permute(0, 3, 1, 2)

        # Get Labels
        if self.phase != "test":
            labels = row[Config.TARGET_COLS].values.astype(np.float32)
            return images, torch.tensor(labels)
        else:
            return images, torch.tensor([])  # Empty tensor for test


def get_transforms(phase: str) -> A.ReplayCompose:
    """
    Returns Albumentations transforms.
    Uses ReplayCompose to ensure consistency across the sequence.
    """
    if phase == "train":
        return A.ReplayCompose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                ToTensorV2(),
            ]
        )
    else:
        return A.ReplayCompose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Creates Train and Validation DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Get Study Paths (Cached)
    # We combine dfs to build cache for all relevant studies at once or handle separately
    # Let's handle separately to match Config paths
    train_study_paths = get_study_paths(
        train_df, Config.TRAIN_CACHE_PATH, load_cached_data
    )
    val_study_paths = get_study_paths(val_df, Config.VAL_CACHE_PATH, load_cached_data)

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = RSNADataset(
        train_df, train_study_paths, transform=train_transform, phase="train"
    )
    val_dataset = RSNADataset(
        val_df, val_study_paths, transform=val_transform, phase="val"
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


def get_test_dataloader(load_cached_data: bool = True) -> DataLoader:
    """
    Creates Test DataLoader.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_study_paths = get_study_paths(
        test_df, Config.TEST_CACHE_PATH, load_cached_data
    )

    test_transform = get_transforms("test")

    test_dataset = RSNADataset(
        test_df, test_study_paths, transform=test_transform, phase="test"
    )

    return DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
