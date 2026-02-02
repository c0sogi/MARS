import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class CervicalSpineDataset(Dataset):
    """
    Dataset class for Cervical Spine Fracture Detection.
    Loads cached 3D volumes, converts them to 2.5D stacks, applies consistent
    volumetric augmentation, and returns tensors ready for the model.
    """

    def __init__(self, metadata_df, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata (StudyInstanceUID, labels).
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label returning.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.mode = mode
        self.cache_dir = Config.CACHE_DIR
        self.image_size = Config.IMAGE_SIZE
        self.num_slices = Config.NUM_SLICES

        # Augmentation parameters
        self.rot_range = 15.0  # degrees
        self.scale_range = 0.15  # +/- 15%
        self.shift_range = 0.1  # +/- 10%

        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # 1. Load Volume
        volume = self._load_volume(study_uid)

        # 2. Augmentation (Train only)
        # We apply augmentation on the [D, H, W] volume before making 2.5D stacks
        # to save compute, or after?
        # Ideally, we apply affine to the spatial dimensions.
        # Since 2.5D takes neighbors, spatial transform must be consistent.
        if self.mode == "train":
            volume = self.apply_consistent_augmentation(volume)

        # 3. Construct 2.5D Stacks
        # Output shape: [D, H, W, 3] -> [D, 3, H, W]
        volume_25d = self.make_2_5d_slice(volume)

        # 4. Normalize and To Tensor
        # Convert to float [0, 1]
        volume_25d = volume_25d.astype(np.float32) / 255.0

        # Normalize
        volume_25d = (volume_25d - self.mean) / self.std

        # Transpose to [D, C, H, W]
        volume_25d = np.transpose(volume_25d, (0, 3, 1, 2))

        # Convert to torch tensor
        data_tensor = torch.from_numpy(volume_25d).float()

        # 5. Get Labels
        if self.mode in ["train", "val"]:
            # Order: C1, C2, C3, C4, C5, C6, C7, patient_overall
            labels = [
                row["C1"],
                row["C2"],
                row["C3"],
                row["C4"],
                row["C5"],
                row["C6"],
                row["C7"],
                row["patient_overall"],
            ]
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            return data_tensor, label_tensor
        else:
            # Test mode, return study_uid for submission mapping
            return data_tensor, study_uid

    def _load_volume(self, study_uid):
        """
        Loads the preprocessed .npy volume from cache.
        Returns a numpy array of shape (D, H, W).
        """
        path = os.path.join(self.cache_dir, f"{study_uid}.npy")

        if os.path.exists(path):
            try:
                volume = np.load(path)
                # Ensure correct shape [64, 256, 256]
                if volume.shape != (self.num_slices, self.image_size, self.image_size):
                    # Fallback resize if cache is inconsistent
                    # (Though preprocessor should handle this)
                    pass
                return volume
            except Exception:
                pass

        # Fallback: Return zeros if file missing or corrupt
        return np.zeros(
            (self.num_slices, self.image_size, self.image_size), dtype=np.uint8
        )

    def apply_consistent_augmentation(self, volume):
        """
        Generates a single affine transform and applies it to every slice in the volume.
        Fuses rotation, scaling, and shifting.

        Args:
            volume (np.ndarray): Shape (D, H, W)

        Returns:
            np.ndarray: Augmented volume (D, H, W)
        """
        D, H, W = volume.shape
        center = (W // 2, H // 2)

        # Generate random parameters
        angle = np.random.uniform(-self.rot_range, self.rot_range)
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        tx = np.random.uniform(-self.shift_range, self.shift_range) * W
        ty = np.random.uniform(-self.shift_range, self.shift_range) * H

        # Construct Affine Matrix
        # Get rotation + scale matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Add translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply to all slices
        augmented_slices = []
        for i in range(D):
            # warpAffine expects (W, H)
            slice_img = volume[i]
            aug_slice = cv2.warpAffine(
                slice_img,
                M,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            augmented_slices.append(aug_slice)

        return np.stack(augmented_slices)

    def make_2_5d_slice(self, volume):
        """
        Constructs 2.5D inputs.
        For each slice i, creates a 3-channel image [slice_i-1, slice_i, slice_i+1].

        Args:
            volume (np.ndarray): Shape (D, H, W)

        Returns:
            np.ndarray: Shape (D, H, W, 3)
        """
        D, H, W = volume.shape

        # Pad volume with one slice at top and bottom to handle boundaries easily
        # Replicate border slices
        padded = np.zeros((D + 2, H, W), dtype=volume.dtype)
        padded[1:-1] = volume
        padded[0] = volume[0]
        padded[-1] = volume[-1]

        # Stack
        # Channel 0: z-1 (indices 0 to D-1 in padded)
        # Channel 1: z   (indices 1 to D   in padded)
        # Channel 2: z+1 (indices 2 to D+1 in padded)

        c0 = padded[0:D]
        c1 = padded[1 : D + 1]
        c2 = padded[2 : D + 2]

        # Stack along last axis -> (D, H, W, 3)
        volume_25d = np.stack([c0, c1, c2], axis=-1)

        return volume_25d
