import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import INPUT_DIR, IMG_SIZE
from library.data_processing import (
    load_dicom_slice,
    normalize_minmax,
)


class RNWIVDataset(Dataset):
    """
    Simplified 2.5D Dataset (Cite solution_lesson_node_00002, solution_lesson_node_00036).
    Selects the middle slice of FLAIR, T1wCE, and T2w modalities independently.
    Constructs a 3-channel input tensor.
    """

    def __init__(self, df, transform=None, is_train=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing subject IDs and paths.
            transform (albumentations.Compose): Augmentation pipeline.
            is_train (bool): Flag indicating training mode.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        # Cite solution_lesson_node_00031: Early Fusion (stacking channels)
        self.modalities = ["flair", "t1wce", "t2w"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        channels = []

        for mod in self.modalities:
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            img = None
            if os.path.exists(full_path):
                # Cite solution_lesson_node_00036: Independent heuristics (median index)
                files = sorted(
                    [f for f in os.listdir(full_path) if f.endswith(".dcm")],
                    key=lambda x: int(x.split("-")[-1].split(".")[0]),
                )

                if files:
                    # Select middle slice
                    mid_idx = len(files) // 2
                    file_path = os.path.join(full_path, files[mid_idx])
                    img = load_dicom_slice(file_path)

            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            else:
                # Cite solution_lesson_node_00023: Independent Channel Normalization
                img = normalize_minmax(img)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            channels.append(img)

        # Stack to (H, W, 3)
        image = np.stack(channels, axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))

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
