import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_slice_number(filename):
    """
    Extracts the integer slice number from a DICOM filename (e.g., '100.dcm' -> 100).
    Returns -1 if extraction fails.
    """
    try:
        return int(os.path.splitext(filename)[0])
    except ValueError:
        return -1


def cache_image_paths(metadata_df, subset_name, load_cached_data=True):
    """
    Scans directories for DICOM files and caches the sorted paths in a Parquet file.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        subset_name (str): Name of the subset (e.g., 'train', 'val', 'test') for naming the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping of StudyInstanceUID -> List of sorted filenames.
    """
    cache_file = os.path.join(Config.working_dir, f"{subset_name}_paths_cache.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading {subset_name} image paths from cache: {cache_file}")
            cached_df = pd.read_parquet(cache_file)
            # Convert back to dict: UID -> list of files
            path_map = cached_df.set_index("StudyInstanceUID")["file_paths"].to_dict()

            # Basic validation
            if path_map and len(path_map) > 0:
                first_key = next(iter(path_map))
                if isinstance(path_map[first_key], (list, np.ndarray)):
                    return path_map
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Scanning directories for {subset_name}...")
    path_map = {}

    for _, row in metadata_df.iterrows():
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_dir_path = os.path.join(Config.input_dir, rel_path)

        if os.path.exists(full_dir_path):
            try:
                files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
                # Sort naturally by slice number to ensure Z-axis continuity
                files.sort(key=get_slice_number)
                path_map[uid] = files
            except OSError:
                path_map[uid] = []
        else:
            path_map[uid] = []

    # 3. Save to cache
    print(f"Saving {subset_name} paths to cache: {cache_file}")
    try:
        # Create a DataFrame for storage
        cache_df = pd.DataFrame(
            {
                "StudyInstanceUID": list(path_map.keys()),
                "file_paths": list(path_map.values()),
            }
        )
        cache_df.to_parquet(cache_file, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache file: {e}")

    return path_map


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline for the specified phase.
    Uses ReplayCompose for training to ensure consistency across the 2.5D sequence.
    """
    if phase == "train":
        return A.ReplayCompose(
            [
                A.Resize(Config.image_size[0], Config.image_size[1]),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.image_size[0], Config.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class RSNADataset(Dataset):
    def __init__(self, metadata_df, image_paths_map, phase="train", transform=None):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata with UIDs and Targets.
            image_paths_map (dict): Pre-computed map of UID -> list of filenames.
            phase (str): 'train', 'val', or 'test'.
            transform (A.Compose or A.ReplayCompose): Albumentations transforms.
        """
        self.df = metadata_df
        self.image_paths_map = image_paths_map
        self.phase = phase
        self.transform = transform

        # Prepare targets if available (Train/Val)
        # Config.target_cols: ["C1", "C2", ..., "C7", "patient_overall"]
        self.has_labels = "patient_overall" in self.df.columns
        if self.has_labels:
            self.labels = self.df[Config.target_cols].values.astype(np.float32)
        else:
            # Test set placeholders
            self.labels = np.zeros(
                (len(self.df), len(Config.target_cols)), dtype=np.float32
            )

        # Pre-calculate full directory paths to avoid overhead in __getitem__
        self.dir_paths = [
            os.path.join(Config.input_dir, p) for p in self.df["image_path"].values
        ]
        self.uids = self.df["StudyInstanceUID"].values

    def __len__(self):
        return len(self.df)

    def load_slice(self, path):
        """
        Reads a DICOM file, converts to uint8 (0-255).
        Handles missing files by returning a black image.
        """
        if not os.path.exists(path):
            return np.zeros(
                (Config.image_size[0], Config.image_size[1]), dtype=np.uint8
            )

        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # Robust Min-Max Normalization to 0-255
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
                img = (img * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

            return img
        except Exception:
            # Fallback for corrupted files
            return np.zeros(
                (Config.image_size[0], Config.image_size[1]), dtype=np.uint8
            )

    def __getitem__(self, idx):
        uid = self.uids[idx]
        dir_path = self.dir_paths[idx]
        all_files = self.image_paths_map.get(uid, [])

        num_files = len(all_files)
        seq_len = Config.seq_len

        # Determine indices to sample
        if num_files == 0:
            # Handle empty study gracefully
            indices = np.zeros(seq_len, dtype=int)
            all_files = ["dummy.dcm"]
            num_files = 1
        else:
            # Uniform sampling for both train and test to ensure Z-axis coverage
            # We use linspace to select 'seq_len' slices evenly distributed across the scan
            indices = np.linspace(0, num_files - 1, seq_len).round().astype(int)

        # Dataset-Encapsulated Augmentation:
        # Sample parameters once using a dummy image to ensure consistency across the sequence
        replay_data = None
        if self.phase == "train" and isinstance(self.transform, A.ReplayCompose):
            dummy_img = np.zeros(
                (Config.image_size[0], Config.image_size[1], 3), dtype=np.uint8
            )
            # This call generates the random parameters (rotation, shift, etc.)
            res = self.transform(image=dummy_img)
            replay_data = res["replay"]

        sequence_tensors = []

        for i in indices:
            # 2.5D Stacking: (i-1, i, i+1)
            # Handle boundary conditions by clamping to valid range
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(num_files - 1, i + 1)

            # Construct paths
            path_prev = os.path.join(dir_path, all_files[idx_prev])
            path_curr = os.path.join(dir_path, all_files[idx_curr])
            path_next = os.path.join(dir_path, all_files[idx_next])

            # Load slices (I/O intensive)
            img_prev = self.load_slice(path_prev)
            img_curr = self.load_slice(path_curr)
            img_next = self.load_slice(path_next)

            # Stack to (H, W, 3)
            img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)

            # Apply Transforms
            if self.phase == "train" and replay_data is not None:
                # Apply the consistent transform parameters
                augmented = A.ReplayCompose.replay(replay_data, image=img_stack)
                img_tensor = augmented["image"]  # Shape: (3, H, W) via ToTensorV2
            elif self.transform:
                # Val/Test (deterministic Compose)
                augmented = self.transform(image=img_stack)
                img_tensor = augmented["image"]
            else:
                # Fallback without transforms
                img_tensor = (
                    torch.from_numpy(img_stack.transpose(2, 0, 1)).float() / 255.0
                )

            sequence_tensors.append(img_tensor)

        # Stack sequence: [seq_len, 3, H, W]
        sequence = torch.stack(sequence_tensors)

        # Labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return sequence, label
