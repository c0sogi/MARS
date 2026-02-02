import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from library import config
from library import dicom_processing


class RSNADataset(Dataset):
    """
    Dataset for GLr genetic subtype prediction.
    Constructs a 24-channel volumetric input using a Dual-Stride strategy.

    Input Structure (24 Channels):
    - Group 1 (Local, Stride 2): [FLAIR, T1w, T1wCE, T2w] (3 slices each -> 12 channels)
    - Group 2 (Context, Stride 10): [FLAIR, T1w, T1wCE, T2w] (3 slices each -> 12 channels)
    """

    def __init__(self, split="train", transform=None, debug=False):
        self.split = split
        self.transform = transform
        self.debug = debug

        # Load Metadata based on split
        if split == "train":
            self.df = pd.read_csv(config.TRAIN_METADATA_PATH)
        elif split == "val":
            self.df = pd.read_csv(config.VAL_METADATA_PATH)
        elif split == "test":
            self.df = pd.read_csv(config.TEST_METADATA_PATH)
        else:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
            )

        # Debug mode: reduce dataset size
        if self.debug:
            self.df = self.df.head(config.MAX_DEBUG_SAMPLES)

    def __len__(self):
        return len(self.df)

    def _get_sorted_files(self, rel_folder_path):
        """
        Retrieves and numerically sorts DICOM files from a directory.
        """
        full_path = os.path.join(config.INPUT_DIR, rel_folder_path)
        if not os.path.exists(full_path):
            return []

        files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]

        # Sort numerically by the image number (Image-N.dcm)
        try:
            files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
        except Exception:
            files.sort()  # Fallback to string sort if naming convention differs

        return [os.path.join(full_path, f) for f in files]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = str(row["BraTS21ID"])

        # Get target (default to 0.0 for test set)
        target = 0.0
        if "MGMT_value" in row:
            target = float(row["MGMT_value"])

        # 1. Prepare File Lists & ROI Anchor
        # We rely on FLAIR for the primary ROI detection
        flair_path_rel = row["path_FLAIR"]
        flair_files = self._get_sorted_files(flair_path_rel)

        # Get Anchor Index using cached logic
        anchor_idx_flair = dicom_processing.get_roi_anchor(
            brats_id, flair_files, load_cached_data=True
        )

        # Calculate Relative Depth of the anchor in FLAIR
        num_slices_flair = len(flair_files)
        if num_slices_flair > 0:
            rel_depth = anchor_idx_flair / num_slices_flair
        else:
            rel_depth = 0.5

        # 2. Construct Input Tensor
        channels = []

        # Iterate through modalities in fixed order
        for mod in config.MODALITIES:  # ["FLAIR", "T1w", "T1wCE", "T2w"]
            if mod == "FLAIR":
                mod_files = flair_files
            else:
                mod_files = self._get_sorted_files(row[f"path_{mod}"])

            num_slices = len(mod_files)

            # Handle empty directories
            if num_slices == 0:
                zeros = np.zeros(
                    (config.IMG_SIZE, config.IMG_SIZE, config.NUM_SLICES_PER_GROUP),
                    dtype=np.float32,
                )
                channels.append(zeros)
                continue

            # Project Anchor to current modality
            anchor_idx_mod = int(rel_depth * num_slices)

            # Get Indices using Fixed Stride
            # Cite solution_lesson_node_00037: Fixed Stride
            indices = dicom_processing.get_stride_indices(anchor_idx_mod, num_slices)

            # Load and Preprocess Slices
            mod_imgs = []
            for i in indices:
                f_path = mod_files[i]
                img = dicom_processing.read_dicom_robust(f_path)
                img = dicom_processing.preprocess_slice(img)
                mod_imgs.append(img)

            # Stack slices for this modality: [H, W, 3]
            mod_stack = np.stack(mod_imgs, axis=-1)
            channels.append(mod_stack)

        # 3. Assemble Final Volume
        # Result: [FLAIR_3, T1w_3, T1wCE_3, T2w_3] -> 12 Channels
        image_volume = np.concatenate(channels, axis=-1)

        # 4. Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image_volume)
            image_volume = augmented["image"]
        else:
            # Default to ToTensor (C, H, W)
            image_volume = torch.from_numpy(image_volume.transpose(2, 0, 1))

        return image_volume, torch.tensor(target, dtype=torch.float32)


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.
    Includes specific augmentations mentioned in the Idea:
    - Horizontal/Vertical Flip
    - Random Rotation (+/- 15 deg) with Zero Padding
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotation with constant (zero) padding to avoid reflection artifacts
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose([ToTensorV2()])
    else:
        return A.Compose([ToTensorV2()])
