import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import INPUT_DIR, IMG_SIZE, ROI_DEPTHS
from library.data_processing import (
    load_dicom_slice,
    normalize_minmax,
    get_relative_indices,
)


class RNWIVDataset(Dataset):
    """
    Relative-Norm Weight-Inflated Volumetric (RN-WIV) Dataset.

    Constructs a 9-channel input tensor by sampling 3 modalities (FLAIR, T1wCE, T2w)
    at 3 relative depths (40%, 50%, 60%) of the brain ROI.
    """

    def __init__(self, df, roi_df, transform=None, is_train=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing subject IDs and paths.
            roi_df (pd.DataFrame): Dataframe containing ROI start/end/count for each subject/modality.
            transform (albumentations.Compose): Augmentation pipeline.
            is_train (bool): Flag indicating training mode.
        """
        self.df = df.reset_index(drop=True)
        self.roi_df = roi_df.set_index("BraTS21ID")
        self.transform = transform
        self.is_train = is_train

        # Modalities and depths defined by the RN-WIV strategy
        # Order: 40% (FLAIR, T1wCE, T2w), 50% (FLAIR, T1wCE, T2w), 60% (FLAIR, T1wCE, T2w)
        self.modalities = ["flair", "t1wce", "t2w"]
        self.depths = ROI_DEPTHS  # Expected to be [0.4, 0.5, 0.6]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["BraTS21ID"]

        # Retrieve ROI data from the provided dataframe (cached externally)
        if sid in self.roi_df.index:
            roi_data = self.roi_df.loc[sid]
        else:
            # Fallback for safety, though pipeline should ensure coverage
            roi_data = {
                f"{m}_{k}": 0
                for m in self.modalities
                for k in ["start", "end", "count"]
            }

        channels = []

        # Iterate depth-first to create the channel structure:
        # [D1_M1, D1_M2, D1_M3, D2_M1, D2_M2, D2_M3, D3_M1, D3_M2, D3_M3]
        for depth in self.depths:
            for mod in self.modalities:
                # 1. Get Path Information
                rel_path = row[f"{mod}_path"]
                full_path = os.path.join(INPUT_DIR, rel_path)

                # 2. Get ROI Information
                start = roi_data[f"{mod}_start"]
                end = roi_data[f"{mod}_end"]
                count = roi_data[f"{mod}_count"]

                # 3. Calculate Target Index
                # get_relative_indices returns a list; we pass a single-item list
                indices = get_relative_indices(start, end, count, [depth])
                target_idx = indices[0]

                # 4. Load Image
                # We must list directory to map index to filename (Image-X.dcm)
                img = None
                if os.path.exists(full_path):
                    # Sorting ensures alignment with the index logic
                    files = sorted(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")],
                        key=lambda x: int(x.split("-")[-1].split(".")[0]),
                    )

                    if files and target_idx < len(files):
                        file_path = os.path.join(full_path, files[target_idx])
                        img = load_dicom_slice(file_path)

                # 5. Normalize and Resize
                if img is None:
                    # Handle missing files or read errors with empty slice
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                else:
                    img = normalize_minmax(img)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                channels.append(img)

        # Stack channels to create (H, W, 9) volume
        image = np.stack(channels, axis=-1)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]  # Albumentations ToTensorV2 returns (C, H, W)
        else:
            # Manual conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1))

        # Return Image and Target
        if "MGMT_value" in row:
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return image, target
        else:
            return image, torch.tensor(-1.0, dtype=torch.float32)


def get_transforms(phase):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Strategy:
    - Train: Flips, Rotation, Elastic/Grid distortions.
             Strictly EXCLUDES translation/shifting to preserve centroid alignment.
    - Test/Val: ToTensorV2 only.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Spatially-preserved distortions
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
