import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.data_utils import load_patient_images


class LungDataset(Dataset):
    """
    Dataset for DP-SDAN model.
    Loads dual-view CT scans (Axial/Coronal) and clinical metadata.
    Handles baseline alignment for longitudinal prediction.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Custom transforms. If None, defaults are used.
        """
        self.mode = mode
        self.transform = transform

        # 1. Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.df = self._prepare_train_metadata(self.df)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.df = self._prepare_train_metadata(self.df)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            # Test CSV already has Baseline_FVC, Baseline_Percent, etc.
            # Ensure column naming consistency
            self.df["Weeks"] = self.df["Predict_Week"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Define Augmentations
        if self.transform is None:
            self.transform = self._get_transforms()

    def _prepare_train_metadata(self, df):
        """
        For training/val data, we need to identify the baseline visit (Week ~ 0)
        to extract static priors (Baseline FVC, Baseline Percent) and attach them
        to every row for that patient.
        """
        # Sort by Patient and Weeks to ensure order
        df = df.sort_values(["Patient", "Weeks"])

        # Identify baseline rows: The row with Weeks closest to 0 for each patient
        # We'll use the first row per patient as baseline (usually the earliest visit)
        # In this dataset, the CT is acquired at baseline.
        baseline_df = df.drop_duplicates(subset=["Patient"], keep="first").copy()

        # Select relevant columns to merge back
        baseline_cols = {
            "Patient": "Patient",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Weeks": "Baseline_Week",
            "Age": "Baseline_Age",
            "Sex": "Baseline_Sex",
            "SmokingStatus": "Baseline_SmokingStatus",
        }
        baseline_df = baseline_df[list(baseline_cols.keys())].rename(
            columns=baseline_cols
        )

        # Merge baseline info onto the full history
        df = pd.merge(df, baseline_df, on="Patient", how="left")

        return df

    def _get_transforms(self):
        """
        Returns Albumentations transforms.
        """
        # ImageNet Normalization Constants
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.mode == "train":
            return A.Compose(
                [
                    # Spatial Augmentations (No intensity changes)
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                    ),
                    # Normalize and Convert to Tensor
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    # Validation/Test: Only Normalize
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )

    def _process_tabular_features(self, row):
        """
        Constructs the 6-dim dense tabular vector.
        Features: [Age, Sex, Percent, Smoke_Ex, Smoke_Never, Smoke_Current]
        Uses Baseline values for consistency.
        """
        # 1. Age (Scaled)
        # Approx range 50-90. Scale: (x - 50) / 50
        age = (row["Baseline_Age"] - 50.0) / 50.0

        # 2. Sex (Binary)
        # Male: 0, Female: 1
        sex = 0.0 if row["Baseline_Sex"] == "Male" else 1.0

        # 3. Percent (Scaled)
        # Approx range 30-150. Scale: (x - 50) / 50
        percent = (row["Baseline_Percent"] - 50.0) / 50.0

        # 4. SmokingStatus (One-Hot: 3 dims)
        # Order: Ex-smoker, Never smoked, Currently smokes
        status = row["Baseline_SmokingStatus"]
        smoke_ex = 1.0 if status == "Ex-smoker" else 0.0
        smoke_never = 1.0 if status == "Never smoked" else 0.0
        smoke_current = 1.0 if status == "Currently smokes" else 0.0

        # Construct vector
        vector = np.array(
            [age, sex, percent, smoke_ex, smoke_never, smoke_current], dtype=np.float32
        )
        return vector

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # ===========================
        # 1. Load Images
        # ===========================
        # Load from cache or compute (handled by data_utils)
        # dicom_dir is relative path e.g., "train/ID..."
        dicom_dir = row["dicom_dir"]

        # Returns (224, 224, 3) numpy arrays in [0, 1] range
        img_ax, img_cor = load_patient_images(
            patient_id, dicom_dir, load_cached_data=True
        )

        # ===========================
        # 2. Augmentations
        # ===========================
        # Albumentations expects dict
        # We apply the SAME spatial transform parameters to both views?
        # Ideally, they are independent views, so independent augmentation is fine/better for regularization.

        # Apply transform to Axial
        aug_ax = self.transform(image=img_ax)["image"]  # Returns Tensor (C, H, W)

        # Apply transform to Coronal
        aug_cor = self.transform(image=img_cor)["image"]  # Returns Tensor (C, H, W)

        # ===========================
        # 3. Tabular Features
        # ===========================
        # Dense vector for MLP (6-dim)
        tab_dense = torch.tensor(
            self._process_tabular_features(row), dtype=torch.float32
        )

        # Meta scalars for Anchored Head and Loss
        # FVC = Base + alpha * delta_week
        baseline_fvc = float(row["Baseline_FVC"])
        current_week = float(row["Weeks"])
        baseline_week = float(row["Baseline_Week"])
        delta_week = current_week - baseline_week

        # Target FVC (Ground Truth)
        # For test set, FVC might be a dummy value (e.g. 2000), but we don't use it for backprop.
        target_fvc = float(row["FVC"])

        return {
            "img_axial": aug_ax,  # (3, 224, 224)
            "img_coronal": aug_cor,  # (3, 224, 224)
            "tab_dense": tab_dense,  # (6,)
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "delta_week": torch.tensor(delta_week, dtype=torch.float32),
            "target_fvc": torch.tensor(target_fvc, dtype=torch.float32),
            "patient_id": patient_id,
            "patient_week_id": (
                f"{patient_id}_{int(current_week)}" if self.mode == "test" else ""
            ),
        }
