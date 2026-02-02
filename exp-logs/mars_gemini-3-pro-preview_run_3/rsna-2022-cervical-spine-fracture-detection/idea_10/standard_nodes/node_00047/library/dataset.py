import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_and_preprocess_scan


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Uses ReplayCompose to ensure consistency across the volume slices.
    """
    if mode == "train":
        return A.ReplayCompose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # cv2.BORDER_CONSTANT
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Only Normalize and ToTensor
        return A.ReplayCompose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class FractureDataset(Dataset):
    def __init__(
        self, metadata_path, image_root_dir, transform=None, mode="train", debug=False
    ):
        """
        Args:
            metadata_path (str): Path to the CSV file containing metadata.
            image_root_dir (str): Path to the directory containing study folders (e.g., train_images).
            transform (albumentations.ReplayCompose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.df = pd.read_csv(metadata_path)
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.mode = mode

        if debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Target columns
        self.target_cols = [f"C{i}" for i in range(1, 8)]  # C1-C7
        self.patient_col = "patient_overall"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]

        # Load 2.5D volume: Shape (64, 256, 256, 3), dtype uint8
        # Caching is handled inside load_and_preprocess_scan via Config.CACHE_DIR
        # This returns a numpy array of uint8
        volume = load_and_preprocess_scan(
            study_uid, self.image_root_dir, load_cached_data=True
        )

        # Apply Volumetric-Consistent Augmentations
        # We iterate through the sequence.
        # For the first slice, we generate params. For others, we replay.

        processed_slices = []
        replay_params = None

        if self.transform is not None:
            for i in range(volume.shape[0]):
                img_slice = volume[i]  # (256, 256, 3)

                if i == 0:
                    # Apply transform and capture params
                    data = self.transform(image=img_slice)
                    replay_params = data["replay"]
                    processed_slices.append(data["image"])
                else:
                    # Replay transform to ensure geometric consistency
                    data = self.transform.replay(replay_params, image=img_slice)
                    processed_slices.append(data["image"])
        else:
            # Fallback if no transform provided (should not happen if using get_transforms)
            basic_tf = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            for i in range(volume.shape[0]):
                processed_slices.append(basic_tf(image=volume[i])["image"])

        # Stack slices to create sequence tensor
        # Result Shape: (64, 3, 256, 256)
        volume_tensor = torch.stack(processed_slices)

        # Prepare targets
        if self.mode in ["train", "val"]:
            # Vertebrae targets: C1-C7
            targets = row[self.target_cols].values.astype(np.float32)
            # Patient target
            patient_target = np.float32(row[self.patient_col])

            return {
                "image": volume_tensor,
                "targets": torch.tensor(targets),
                "patient_target": torch.tensor(patient_target),
                "study_uid": study_uid,
            }
        else:
            # Test mode
            return {"image": volume_tensor, "study_uid": study_uid}
