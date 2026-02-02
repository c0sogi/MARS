import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_oof_preds


class CyclicRoll(A.ImageOnlyTransform):
    """
    Applies a cyclic shift to the image along the time axis (width).
    Useful for audio spectrograms where time translation invariance is desired.
    """

    def __init__(self, shift_limit=0.5, fixed_shift=None, always_apply=False, p=0.5):
        """
        Args:
            shift_limit (float): Maximum fraction of width to shift (for random mode).
            fixed_shift (float, optional): Exact fraction of width to shift (for TTA).
                                           If provided, random shift is disabled.
            p (float): Probability of applying the transform (ignored if fixed_shift is set).
        """
        super(CyclicRoll, self).__init__(always_apply, p)
        self.shift_limit = shift_limit
        self.fixed_shift = fixed_shift

    def apply(self, img, shift=0, **params):
        """
        Args:
            img (numpy.ndarray): Image to roll. Shape (H, W, C).
            shift (int): Number of pixels to shift.
        """
        return np.roll(img, shift, axis=1)

    def get_params(self):
        return {
            "shift": 0
        }  # Placeholder, logic handled in update_params or apply_to_image logic usually
        # But for Albumentations ImageOnlyTransform, we calculate params in get_transform_init_args_names
        # or rely on `get_params_dependent_on_targets`.
        # Simpler approach for custom logic:

    def get_transform_init_args_names(self):
        return ("shift_limit", "fixed_shift")

    def apply_with_params(self, params, **kwargs):
        # This wrapper is needed for newer albumentations or just use standard apply
        # We override the call behavior by calculating shift inside apply if needed,
        # but standard way is to calculate in get_params.
        # Let's stick to the simplest implementation compatible with apply.
        return super().apply_with_params(params, **kwargs)

    def update_params(self, params, **kwargs):
        img = kwargs["image"]
        h, w = img.shape[:2]

        if self.fixed_shift is not None:
            shift_pixels = int(w * self.fixed_shift)
        else:
            # Random shift
            limit_pixels = int(w * self.shift_limit)
            shift_pixels = np.random.randint(-limit_pixels, limit_pixels)

        params.update({"shift": shift_pixels})
        return params


def get_transforms(mode="train", tta_shift=None):
    """
    Generates the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        tta_shift (float, optional): Specific shift fraction for TTA (0.0 to 1.0).

    Returns:
        A.Compose: The transformation pipeline.
    """
    transforms = []

    # 1. Resize to fixed dimensions (Freq x Time)
    # 224 x 448 preserves aspect ratio better for 10s clips
    transforms.append(A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH))

    # 2. Cyclic Roll (Time Shift)
    if mode == "train":
        # Random cyclic roll during training
        transforms.append(CyclicRoll(shift_limit=0.5, p=0.5))
    elif mode == "test" and tta_shift is not None:
        # Deterministic cyclic roll for TTA
        # We set p=1.0 to ensure it always applies
        transforms.append(CyclicRoll(fixed_shift=tta_shift, always_apply=True))

    # 3. Normalization (ImageNet stats)
    transforms.append(A.Normalize(mean=Config.MEAN, std=Config.STD))

    # 4. Convert to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Handles loading of filtered spectrograms, hard labels, and optional soft labels for distillation.
    """

    def __init__(
        self, metadata_df, transforms=None, soft_labels_path=None, mode="train"
    ):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing 'rec_id', 'file_path_spec', and label columns.
            transforms (A.Compose): Albumentations transforms to apply.
            soft_labels_path (str, optional): Path to Parquet file containing OOF predictions (soft targets).
            mode (str): 'train', 'val', or 'test'. Used for behavior toggles.
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.soft_labels = None

        # Identify label columns
        self.label_cols = [c for c in self.metadata.columns if c.startswith("species_")]

        # Load Soft Labels if provided (for Distillation)
        if soft_labels_path:
            # We need to extract the rec_ids to query the soft labels correctly
            rec_ids = self.metadata["rec_id"].values

            # Load aligned soft targets
            # Returns numpy array of shape [N, num_classes]
            self.soft_labels = load_oof_preds(soft_labels_path, rec_ids)

            if Config.DEBUG:
                print(
                    f"Loaded soft labels from {soft_labels_path} with shape {self.soft_labels.shape}"
                )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Load Image
        # Construct full path
        # Metadata contains relative paths, need to prepend INPUT_DIR
        # Note: Config.INPUT_DIR is "./input"
        img_path = os.path.join(Config.INPUT_DIR, row["file_path_spec"])

        # Load as grayscale (spectrograms are single channel BMPs)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            # Fallback for missing files (should not happen based on EDA, but safety first)
            # Create a blank image
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Convert to Pseudo-RGB (3 channels)
        # If image is already 3 channels (rare but possible), this handles it safely
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 3. Get Hard Labels (Ground Truth)
        # Shape: [num_classes]
        hard_label = row[self.label_cols].values.astype(np.float32)

        # 4. Get Soft Labels (Distillation Target)
        soft_label = np.zeros_like(hard_label)  # Default placeholder
        if self.soft_labels is not None:
            soft_label = self.soft_labels[idx].astype(np.float32)

        # 5. Return Dictionary
        # Returning a dict is flexible and clear
        return {
            "image": image,
            "targets": torch.tensor(hard_label, dtype=torch.float32),
            "soft_targets": (
                torch.tensor(soft_label, dtype=torch.float32)
                if self.soft_labels is not None
                else torch.empty(0)
            ),
            "rec_id": row["rec_id"],
        }
