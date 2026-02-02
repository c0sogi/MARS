import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.dicom_utils import load_scan, sort_slices, pixels_to_hu, apply_window


class CervicalSpineDataset(Dataset):
    def __init__(
        self, metadata_path, phase="train", load_cached_data=True, transform=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            phase (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load/save processed volumes to cache.
            transform (A.Compose): Optional override for transforms.
        """
        self.df = pd.read_csv(metadata_path)
        self.phase = phase
        self.load_cached_data = load_cached_data
        self.root_dir = Config.INPUT_DIR

        # Define transforms if not provided
        if transform is None:
            self.transform = self._get_transforms()
        else:
            self.transform = transform

    def _get_transforms(self):
        """
        Returns the Albumentations transform pipeline based on the phase.
        Using ReplayCompose is handled in __getitem__ logic manually or via A.ReplayCompose
        if we were applying it to a list, but here we iterate.
        We define the base composition here.
        """
        if self.phase == "train":
            return A.Compose(
                [
                    # Cite solution_lesson_node_00042: Single affine transformation to preserve high-frequency features
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.5
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _process_and_cache(self, study_id, rel_path):
        """
        Loads DICOMs, processes them (Sort -> HU -> Window -> Resize), and caches as .npy.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

        # 1. Try to load from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                return volume
            except Exception as e:
                print(f"Error loading cache for {study_id}: {e}. Recomputing...")

        # 2. Compute from scratch
        full_path = os.path.join(self.root_dir, rel_path)

        # Load and Sort
        slices = load_scan(full_path)
        slices = sort_slices(slices)

        if not slices:
            # Fallback for empty directories (should not happen based on metadata check)
            return np.zeros(
                (Config.SEQ_LENGTH, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                dtype=np.uint8,
            )

        # Convert to HU
        volume_hu = pixels_to_hu(slices)

        # Apply Bone Window (Level 400, Width 1800) -> uint8 [0, 255]
        volume_windowed = apply_window(
            volume_hu, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH
        )

        # Resize each slice to target size (256, 256)
        # volume_windowed is (Depth, H, W)
        resized_slices = []
        for i in range(volume_windowed.shape[0]):
            sl = volume_windowed[i]
            # cv2.resize expects (W, H)
            sl_resized = cv2.resize(
                sl, Config.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR
            )
            resized_slices.append(sl_resized)

        volume_final = np.stack(resized_slices)

        # 3. Save to cache
        if self.load_cached_data:
            try:
                np.save(cache_path, volume_final)
            except Exception as e:
                print(f"Failed to save cache for {study_id}: {e}")

        return volume_final

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]
        image_path = row["image_path"]

        # 1. Load Volume (D, H, W)
        volume = self._process_and_cache(study_id, image_path)
        depth = volume.shape[0]

        # 2. Uniform Sampling
        # We need exactly SEQ_LENGTH (64) slices.
        if depth == 0:
            # Handle edge case of empty volume
            indices = np.zeros(Config.SEQ_LENGTH, dtype=int)
        else:
            indices = np.linspace(0, depth - 1, Config.SEQ_LENGTH).astype(int)

        # 3. Construct 2.5D Stacks and Apply Transforms
        # To ensure volumetric consistency (same transform for all slices),
        # we use ReplayCompose logic manually or via Albumentations.
        # Here we use the ReplayCompose wrapper approach if available,
        # or simply generate params once.
        # Since we are iterating, the standard way with Albumentations ReplayCompose:

        stacked_images = []

        # Prepare the ReplayCompose wrapper just for this item if training
        if self.phase == "train":
            transform_replay = A.ReplayCompose(self.transform.transforms)
            replay_data = None
        else:
            transform_replay = self.transform
            replay_data = None

        for i, slice_idx in enumerate(indices):
            # 2.5D Stacking: (z-1, z, z+1)
            # Handle boundary conditions by clamping
            idx_prev = max(0, slice_idx - 1)
            idx_curr = slice_idx
            idx_next = min(depth - 1, slice_idx + 1)

            # Get slices (H, W)
            # If depth is 0 (empty volume handled above), this loop runs on zeros indices
            if depth > 0:
                s_prev = volume[idx_prev]
                s_curr = volume[idx_curr]
                s_next = volume[idx_next]
            else:
                s_prev = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
                s_curr = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)
                s_next = np.zeros(Config.IMAGE_SIZE, dtype=np.uint8)

            # Stack to (H, W, 3)
            img_stack = np.stack([s_prev, s_curr, s_next], axis=-1)

            # Apply Transform
            if self.phase == "train":
                if i == 0:
                    # First slice: apply and record params
                    augmented = transform_replay(image=img_stack)
                    img_tensor = augmented["image"]
                    replay_data = augmented["replay"]
                else:
                    # Subsequent slices: replay exact params
                    augmented = A.ReplayCompose.replay(replay_data, image=img_stack)
                    img_tensor = augmented["image"]
            else:
                # Val/Test: Just normalize/tensor
                augmented = self.transform(image=img_stack)
                img_tensor = augmented["image"]

            stacked_images.append(img_tensor)

        # Stack into (Seq, C, H, W) -> (64, 3, 256, 256)
        input_tensor = torch.stack(stacked_images)

        # 4. Prepare Labels
        # Test set might not have labels
        labels = {}
        if "patient_overall" in row:
            labels["patient_overall"] = torch.tensor(
                row["patient_overall"], dtype=torch.float32
            )

            # C1-C7
            c_labels = []
            for k in range(1, 8):
                col_name = f"C{k}"
                val = row[col_name] if col_name in row else 0
                c_labels.append(val)
            labels["vertebrae"] = torch.tensor(c_labels, dtype=torch.float32)
        else:
            # Dummy labels for inference
            labels["patient_overall"] = torch.tensor(0.0, dtype=torch.float32)
            labels["vertebrae"] = torch.zeros(7, dtype=torch.float32)

        return {"image": input_tensor, "labels": labels, "study_id": study_id}
