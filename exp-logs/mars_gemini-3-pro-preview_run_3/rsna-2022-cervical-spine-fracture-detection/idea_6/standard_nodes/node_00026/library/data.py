import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_dicom, seed_everything

# =========================================================================
# Preprocessing & Caching Logic
# =========================================================================


def window_image(img, center=500, width=2000):
    """
    Applies a windowing function to the HU image to highlight bone structures.
    Standard Bone Window: WL=500, WW=2000.
    Maps range [center - width/2, center + width/2] to [0, 255].
    """
    lower = center - width // 2
    upper = center + width // 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / width
    img = img * 255.0
    return img.astype(np.uint8)


def preprocess_scan(image_dir):
    """
    Loads all DICOM files from a directory, sorts them, applies windowing,
    resizes, and stacks them into a 3D volume.

    Args:
        image_dir (str): Path to the directory containing DICOM files.

    Returns:
        np.ndarray: 3D array of shape (Depth, H, W) in uint8 format.
                    Returns None if no valid images found.
    """
    # List all files
    files = glob.glob(os.path.join(image_dir, "*"))
    if not files:
        return None

    # Sort files by instance number (filename usually contains instance number)
    # Heuristic: try to extract number from filename, else sort alphabetically
    try:
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        files.sort()

    slices = []
    for f in files:
        img = load_dicom(f)
        # Check if valid (load_dicom returns zeros on failure, but we might want to skip)
        # Here we assume load_dicom handles basic errors.

        # Windowing
        img = window_image(img)

        # Resize to Config dimensions
        if img.shape[:2] != Config.IMAGE_SIZE:
            img = cv2.resize(img, Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

        slices.append(img)

    if not slices:
        return None

    # Stack along depth
    volume = np.stack(slices, axis=0)  # (D, H, W)
    return volume


def cache_dataset(metadata_df, cache_dir, load_cached_data=True):
    """
    Iterates through the metadata dataframe and caches preprocessed volumes.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'StudyInstanceUID' and 'image_path'.
        cache_dir (str): Directory to save .npy files.
        load_cached_data (bool): If True, skips processing if file exists.
    """
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Checking/Caching dataset in {cache_dir}...")

    # Get unique studies
    unique_studies = metadata_df[["StudyInstanceUID", "image_path"]].drop_duplicates()

    for _, row in unique_studies.iterrows():
        uid = row["StudyInstanceUID"]
        rel_path = row["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        save_path = os.path.join(cache_dir, f"{uid}.npy")

        if load_cached_data and os.path.exists(save_path):
            continue

        # Process
        volume = preprocess_scan(full_path)

        if volume is not None:
            np.save(save_path, volume)
        else:
            # Create a dummy volume if scan is completely broken to prevent crash
            # Shape: (Config.NUM_SLICES, H, W)
            dummy = np.zeros((Config.NUM_SLICES, *Config.IMAGE_SIZE), dtype=np.uint8)
            np.save(save_path, dummy)
            print(f"Warning: Failed to load scan {uid}. Saved dummy volume.")


# =========================================================================
# Augmentation
# =========================================================================


class VolumetricTransforms:
    """
    Applies deterministic geometric transformations to a 4D volume (S, C, H, W).
    Ensures the same transformation is applied to every slice in the sequence
    to preserve 3D structural consistency for the context module.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, volume):
        """
        Args:
            volume (torch.Tensor or np.ndarray): Shape (S, C, H, W) or (S, H, W)
        Returns:
            Transformed volume.
        """
        if np.random.rand() > self.prob:
            return volume

        # Generate parameters once
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.85, 1.15)

        h, w = volume.shape[-2], volume.shape[-1]
        center = (w // 2, h // 2)

        # Translation factors (relative to size)
        tx = np.random.uniform(-0.1, 0.1) * w
        ty = np.random.uniform(-0.1, 0.1) * h

        # Construct Affine Matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply to each slice
        # Assuming volume is numpy (S, C, H, W)
        is_torch = torch.is_tensor(volume)
        if is_torch:
            volume = volume.numpy()

        # Handle (S, C, H, W)
        if volume.ndim == 4:
            s, c, _, _ = volume.shape
            # Reshape to (S*C, H, W) for faster processing if possible,
            # or just loop. Looping is safer for memory.
            output = np.zeros_like(volume)
            for i in range(s):
                for j in range(c):
                    output[i, j] = cv2.warpAffine(
                        volume[i, j],
                        M,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
            volume = output

        # Handle (S, H, W)
        elif volume.ndim == 3:
            s, _, _ = volume.shape
            output = np.zeros_like(volume)
            for i in range(s):
                output[i] = cv2.warpAffine(
                    volume[i],
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            volume = output

        if is_torch:
            volume = torch.from_numpy(volume)

        return volume


# =========================================================================
# Dataset
# =========================================================================


class CervicalSpineDataset(Dataset):
    """
    Dataset class for Cervical Spine Fracture Detection.
    Loads cached 3D volumes, samples slices, constructs 2.5D inputs,
    and applies volumetric-consistent augmentations.
    """

    def __init__(self, df, cache_dir, mode="train", transforms=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            cache_dir (str): Directory containing cached .npy files.
            mode (str): 'train', 'val', or 'test'.
            transforms (callable, optional): Augmentation pipeline.
        """
        self.df = df
        self.cache_dir = cache_dir
        self.mode = mode
        self.transforms = transforms

        # Pre-calculate indices for efficiency
        self.uids = self.df["StudyInstanceUID"].values

        # For training/validation, extract labels
        if self.mode != "test":
            self.labels = self.df[
                ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
            ].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        file_path = os.path.join(self.cache_dir, f"{uid}.npy")

        # Load Volume
        if os.path.exists(file_path):
            try:
                volume = np.load(file_path)  # (D, H, W)
            except Exception:
                # Fallback
                volume = np.zeros(
                    (Config.NUM_SLICES, *Config.IMAGE_SIZE), dtype=np.uint8
                )
        else:
            # Fallback if cache missing
            volume = np.zeros((Config.NUM_SLICES, *Config.IMAGE_SIZE), dtype=np.uint8)

        current_depth = volume.shape[0]

        # --- Uniform Sampling ---
        # We want exactly Config.NUM_SLICES
        if current_depth == 0:
            indices = np.zeros(Config.NUM_SLICES, dtype=int)
        else:
            indices = np.linspace(0, current_depth - 1, Config.NUM_SLICES).astype(int)

        # --- 2.5D Stack Construction ---
        # For each index i, we want [i-1, i, i+1]
        # We construct this from the loaded volume

        # Create padded volume for boundary handling
        # Pad 1 slice before and 1 slice after
        padded_volume = np.pad(volume, ((1, 1), (0, 0), (0, 0)), mode="edge")

        # Adjust indices to account for padding (shift by +1)
        shifted_indices = indices + 1

        # Gather slices
        # Shape: (NUM_SLICES, 3, H, W)
        stack_slices = []
        for idx_val in shifted_indices:
            # Extract z-1, z, z+1
            # Slicing from idx_val-1 to idx_val+2 gives 3 slices
            chunk = padded_volume[idx_val - 1 : idx_val + 2]
            stack_slices.append(chunk)

        # Stack into numpy array
        img_stack = np.stack(stack_slices, axis=0)  # (S, 3, H, W)

        # --- Augmentation ---
        if self.mode == "train" and self.transforms:
            img_stack = self.transforms(img_stack)

        # --- Normalization ---
        # Convert to float32 and [0, 1]
        img_stack = img_stack.astype(np.float32) / 255.0

        # ImageNet Normalization
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        # Input shape is (S, 3, H, W)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        img_stack = (img_stack - mean) / std

        # Convert to Tensor
        img_tensor = torch.from_numpy(img_stack)  # (S, 3, H, W)

        # --- Labels ---
        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        else:
            # Return UID for submission matching if needed, or just image
            return img_tensor, uid
