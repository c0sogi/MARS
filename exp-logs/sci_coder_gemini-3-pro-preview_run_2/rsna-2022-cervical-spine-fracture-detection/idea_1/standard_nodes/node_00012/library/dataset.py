import os
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_study_paths, load_dicom_and_process


class UnifiedRandomAffine:
    """
    Applies the same random affine transformation and horizontal flip
    to all slices in a 3D volume (Bag).
    Expects input tensor of shape (S, C, H, W).
    """

    def __init__(self, degrees=15, translate=0.1, scale=0.1, p_flip=0.5):
        self.degrees = degrees
        self.translate = translate  # fraction of img size
        self.scale = scale  # deviation from 1.0 (e.g. 0.1 means 0.9-1.1)
        self.p_flip = p_flip

    def __call__(self, x):
        # x shape: (S, C, H, W)

        # 1. Random Horizontal Flip
        if random.random() < self.p_flip:
            x = TF.hflip(x)

        # 2. Random Affine
        # Sample parameters once for the whole bag (Cite solution_lesson_node_00008)
        angle = random.uniform(-self.degrees, self.degrees)

        if self.translate > 0:
            max_dx = self.translate * x.shape[-1]
            max_dy = self.translate * x.shape[-2]
            tx = random.uniform(-max_dx, max_dx)
            ty = random.uniform(-max_dy, max_dy)
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if self.scale > 0:
            s = random.uniform(1.0 - self.scale, 1.0 + self.scale)
        else:
            s = 1.0

        # TF.affine applies the transformation to the last 2 dimensions (H, W)
        # It preserves the leading dimensions (S, C)
        x = TF.affine(x, angle=angle, translate=translations, scale=s, shear=0)

        return x


class RSNADataset(Dataset):
    """
    PyTorch Dataset for Cervical Spine Fracture Detection.
    Loads CT scans as a bag of 2D slices with uniform subsampling.
    """

    def __init__(self, df, config, transform=None, load_cached_paths=True):
        """
        Args:
            df (pd.DataFrame): Metadata DataFrame containing StudyInstanceUID and labels.
            config (Config): Configuration object with hyperparameters.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_cached_paths (bool): Whether to use cached file paths to speed up initialization.
        """
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform

        # Define target columns
        self.target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        # Check if labels exist in the dataframe (Train/Val vs Test)
        self.has_labels = all(col in self.df.columns for col in self.target_cols)

        # Prepare file paths with caching mechanism
        self.study_paths = self._prepare_file_paths(load_cached_paths)

    def _prepare_file_paths(self, load_cached):
        """
        Loads or generates a dictionary mapping StudyInstanceUID to a sorted list of DICOM file paths.
        Uses Parquet for caching to avoid repetitive os.listdir calls.
        """
        cache_path = os.path.join(self.config.CACHE_DIR, "paths_cache.parquet")
        cached_data = {}

        # 1. Try to load existing cache
        if load_cached and os.path.exists(cache_path):
            try:
                # Read parquet and convert back to dict
                df_cache = pd.read_parquet(cache_path)
                cached_data = df_cache.set_index("StudyInstanceUID")["paths"].to_dict()
            except Exception as e:
                print(f"Warning: Failed to load path cache ({e}). Recomputing...")
                cached_data = {}

        # 2. Identify missing UIDs (studies not in cache)
        required_uids = self.df["StudyInstanceUID"].unique()
        missing_uids = [uid for uid in required_uids if uid not in cached_data]

        # 3. Compute paths for missing studies and update cache
        if missing_uids:
            # Create a lookup for relative image paths from the dataframe
            # We drop duplicates to ensure one path per UID
            uid_to_relpath = (
                self.df.drop_duplicates("StudyInstanceUID")
                .set_index("StudyInstanceUID")["image_path"]
                .to_dict()
            )

            new_entries = []
            for uid in missing_uids:
                rel_path = uid_to_relpath.get(uid)
                if rel_path:
                    full_dir_path = os.path.join(self.config.INPUT_DIR, rel_path)
                    paths = get_study_paths(full_dir_path)
                    cached_data[uid] = paths
                else:
                    cached_data[uid] = []

            # Save updated cache to disk
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Convert dict to DataFrame for Parquet storage
            save_df = pd.DataFrame(
                [{"StudyInstanceUID": k, "paths": v} for k, v in cached_data.items()]
            )
            save_df.to_parquet(cache_path)

        return cached_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["StudyInstanceUID"]

        # Retrieve file paths
        paths = self.study_paths.get(uid, [])

        # Initialize tensor container
        # Shape: (NUM_SLICES, Channels, H, W)
        tensor_shape = (
            self.config.NUM_SLICES,
            self.config.IN_CHANNELS,
            self.config.IMG_SIZE,
            self.config.IMG_SIZE,
        )

        if len(paths) == 0:
            # Return zero tensor if no images found
            images = torch.zeros(tensor_shape, dtype=torch.float32)
        else:
            # Uniform Subsampling
            # Select N slices evenly spaced across the Z-axis
            num_files = len(paths)
            indices = np.linspace(0, num_files - 1, self.config.NUM_SLICES).astype(int)
            selected_paths = [paths[i] for i in indices]

            # Load and process each slice
            slice_list = []
            for p in selected_paths:
                img = load_dicom_and_process(
                    p,
                    size=self.config.IMG_SIZE,
                    window_center=self.config.BONE_WINDOW_CENTER,
                    window_width=self.config.BONE_WINDOW_WIDTH,
                )
                slice_list.append(img)

            # Stack slices: (NUM_SLICES, H, W)
            volume = np.stack(slice_list, axis=0)

            # Add channel dimension: (NUM_SLICES, 1, H, W)
            volume = np.expand_dims(volume, axis=1)

            images = torch.from_numpy(volume).float()

        # Apply optional transforms
        if self.transform:
            images = self.transform(images)

        # Retrieve Labels
        if self.has_labels:
            labels = row[self.target_cols].values.astype(np.float32)
            labels = torch.from_numpy(labels)
        else:
            # Return dummy labels for test set
            labels = torch.zeros(len(self.target_cols), dtype=torch.float32)

        return images, labels
