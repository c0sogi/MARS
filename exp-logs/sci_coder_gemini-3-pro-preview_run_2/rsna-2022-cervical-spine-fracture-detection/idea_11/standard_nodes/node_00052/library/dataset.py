import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library.config import Config
from library.utils import load_dicom


def get_image_paths(metadata_df, mode, load_cached_data=True):
    """
    Retrieves or generates a cache of sorted slice filenames for each study.
    """
    cache_file = os.path.join(Config.WORKING_DIR, f"{mode}_paths_cache.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            cached_df = pd.read_parquet(cache_file)
            # Convert DataFrame back to dict: StudyUID -> list of files
            path_map = pd.Series(
                cached_df.file_paths.values, index=cached_df.StudyInstanceUID
            ).to_dict()
            return path_map
        except Exception as e:
            print(
                f"Warning: Failed to load cache {cache_file}. Recomputing. Error: {e}"
            )

    # 2. Compute from scratch
    path_map = {}

    # Iterate over unique studies in the metadata
    unique_studies = metadata_df[["StudyInstanceUID", "image_path"]].drop_duplicates()

    for _, row in unique_studies.iterrows():
        study_id = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_dir_path = os.path.join(Config.DATA_ROOT, rel_path)

        if os.path.exists(full_dir_path):
            try:
                # List all .dcm files
                files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]

                # Sort numerically by slice number (filename is usually '1.dcm', '10.dcm', etc.)
                # We assume filename structure is '{slice_number}.dcm'
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))

                path_map[study_id] = files
            except Exception:
                # Fallback if sorting fails or directory access issues
                path_map[study_id] = []
        else:
            path_map[study_id] = []

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Convert dict to DataFrame for Parquet storage
    # PyArrow handles lists in columns efficiently
    save_df = pd.DataFrame(
        {
            "StudyInstanceUID": list(path_map.keys()),
            "file_paths": list(path_map.values()),
        }
    )
    save_df.to_parquet(cache_file)

    return path_map


class CervicalSpineDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", transform=None, load_cached_data=True
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transform pipeline.
            load_cached_data (bool): Whether to use cached file paths.
        """
        self.mode = mode
        self.df = pd.read_csv(metadata_path)

        # Load dictionary mapping StudyUID -> List of sorted filenames
        self.path_map = get_image_paths(self.df, mode, load_cached_data)

        # Define Transforms
        if transform is None:
            if mode == "train":
                # Use ReplayCompose to apply identical geometric transforms across the Z-stack
                self.transform = A.ReplayCompose(
                    [
                        A.Rotate(limit=15, p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                        ),
                        A.HorizontalFlip(p=0.5),
                        # Normalize: Input is [0, 1] float, so max_pixel_value=1.0
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=1.0,
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225),
                            max_pixel_value=1.0,
                        ),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Get sorted list of slice files
        slice_files = self.path_map.get(study_id, [])
        num_slices = len(slice_files)

        # Determine sampling indices
        seq_len = Config.SEQ_LEN

        if num_slices == 0:
            # Fallback for empty/missing directories
            indices = np.zeros(seq_len, dtype=int)
            slice_files = ["dummy.dcm"]
        else:
            # Uniform sampling across the Z-axis
            indices = np.linspace(0, num_slices - 1, seq_len).astype(int)

        full_dir_path = os.path.join(Config.DATA_ROOT, row["image_path"])

        image_stack = []
        replay_data = None

        # Iterate through sampled indices to build the sequence
        for i, slice_idx in enumerate(indices):
            # 2.5D Stacking: Load z-1, z, z+1
            # Clamp to valid range [0, num_slices - 1]
            neighbors = [slice_idx - 1, slice_idx, slice_idx + 1]
            neighbors = [max(0, min(num_slices - 1, z)) for z in neighbors]

            channels = []
            for z in neighbors:
                if num_slices == 0:
                    img = np.zeros(Config.IMAGE_SIZE, dtype=np.float32)
                else:
                    file_path = os.path.join(full_dir_path, slice_files[z])
                    img = load_dicom(file_path, size=Config.IMAGE_SIZE)
                channels.append(img)

            # Stack to (H, W, 3)
            img_25d = np.stack(channels, axis=-1)

            # Apply Augmentation
            if self.mode == "train":
                if i == 0:
                    # First frame: Generate random parameters and store replay data
                    augmented = self.transform(image=img_25d)
                    replay_data = augmented["replay"]
                    img_tensor = augmented["image"]
                else:
                    # Subsequent frames: Replay exact same parameters
                    augmented = self.transform.replay(replay_data, image=img_25d)
                    img_tensor = augmented["image"]
            else:
                # Validation/Test: Deterministic transform
                augmented = self.transform(image=img_25d)
                img_tensor = augmented["image"]

            image_stack.append(img_tensor)

        # Stack into sequence tensor: (Seq_Len, 3, H, W)
        # ToTensorV2 moves channels first, so list elements are (3, H, W)
        volume = torch.stack(image_stack)

        # Return Data
        if "patient_overall" in row:
            # Training/Validation Mode: Return Labels
            # Columns: C1..C7, patient_overall
            label_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            labels = row[label_cols].values.astype(np.float32)
            return volume, torch.tensor(labels)
        else:
            # Test Mode: Return Study ID for submission mapping
            return volume, study_id
