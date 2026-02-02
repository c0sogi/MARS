import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Paths, Data
from library.image_processing import process_patient


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Decline Prediction.

    Features:
    - Loads Axial and Coronal Tri-Slab images via library.image_processing.
    - Handles baseline feature extraction for temporal prediction.
    - Applies spatial augmentations for training.
    - Returns a dictionary compatible with the Pyramid Dual-Axis Attention Network.
    """

    def __init__(self, df, mode="train", cache_images=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing patient metadata.
            mode (str): 'train', 'val', or 'test'.
            cache_images (bool): Whether to use disk caching for processed images.
        """
        self.mode = mode
        self.cache_images = cache_images

        # Preprocess DataFrame to ensure Baseline features are available
        self.df = self._prepare_data(df.copy())

        # Define Augmentations
        self.transform = self._get_transforms()

    def _prepare_data(self, df):
        """
        Prepares the DataFrame by ensuring Baseline columns exist.

        For Train/Val:
            Identifies the baseline visit (min weeks) for each patient and
            merges baseline features (FVC, Percent, Age, etc.) to all rows.

        For Test:
            The metadata/test.csv already contains Baseline_ columns.
            We just need to ensure column naming consistency.
        """
        if self.mode in ["train", "val"]:
            # 1. Sort by Patient and Weeks to ensure order
            df = df.sort_values(["Patient", "Weeks"])

            # 2. Identify Baseline rows (first visit per patient)
            # We assume the visit with the minimum 'Weeks' value is the baseline
            baseline_df = df.loc[df.groupby("Patient")["Weeks"].idxmin()]

            # 3. Select relevant columns and rename to Baseline_...
            baseline_cols = [
                "Patient",
                "FVC",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
                "Weeks",
            ]
            baseline_df = baseline_df[baseline_cols].copy()

            rename_map = {
                "FVC": "Baseline_FVC",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
                "Weeks": "Baseline_Weeks",
            }
            baseline_df = baseline_df.rename(columns=rename_map)

            # 4. Merge back to original dataframe
            # We use left join to propagate baseline info to every visit
            df = pd.merge(df, baseline_df, on="Patient", how="left")

            # 5. Calculate Time Delta (Weeks from baseline)
            df["Time_Delta"] = df["Weeks"] - df["Baseline_Weeks"]

        elif self.mode == "test":
            # Test metadata already has Baseline_ columns
            # We need to map 'Predict_Week' to 'Weeks' for consistency if not present
            if "Predict_Week" in df.columns and "Weeks" not in df.columns:
                df["Weeks"] = df["Predict_Week"]

            # Calculate Time Delta
            # Note: Baseline_Week is present in test.csv
            df["Time_Delta"] = df["Weeks"] - df["Baseline_Week"]

            # Ensure Baseline_Weeks exists for consistency
            if "Baseline_Weeks" not in df.columns:
                df["Baseline_Weeks"] = df["Baseline_Week"]

        # Reset index to be safe
        return df.reset_index(drop=True)

    def _get_transforms(self):
        """
        Returns Albumentations transforms based on mode.
        """
        if self.mode == "train":
            return A.Compose(
                [
                    # Spatial Augmentations only (No intensity changes)
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    # Normalization
                    A.Normalize(mean=Data.IMAGENET_MEAN, std=Data.IMAGENET_STD),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    # Validation/Test: Only Normalize
                    A.Normalize(mean=Data.IMAGENET_MEAN, std=Data.IMAGENET_STD),
                    ToTensorV2(),
                ]
            )

    def _encode_tabular(self, row):
        """
        Encodes tabular features into a normalized tensor.
        Features: Age, Sex, SmokingStatus, Percent (Baseline values).
        """
        # 1. Sex Encoding (Male: 0, Female: 1)
        sex = 0.0 if row["Baseline_Sex"] == "Male" else 1.0

        # 2. SmokingStatus Encoding
        # Ex-smoker: 0.0, Never smoked: 0.5, Currently smokes: 1.0
        ss = row["Baseline_SmokingStatus"]
        if ss == "Ex-smoker":
            smoke = 0.0
        elif ss == "Never smoked":
            smoke = 0.5
        else:
            smoke = 1.0

        # 3. Age Normalization (Robust scaling: (x - 65) / 15)
        age = (float(row["Baseline_Age"]) - 65.0) / 15.0

        # 4. Percent Normalization (Robust scaling: (x - 80) / 20)
        percent = (float(row["Baseline_Percent"]) - 80.0) / 20.0

        return torch.tensor([age, sex, smoke, percent], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal Tri-Slabs)
        # process_patient handles caching and loading
        dicom_dir = row["dicom_dir"]
        axial_img, coronal_img = process_patient(
            patient_id, dicom_dir, load_cached_data=self.cache_images
        )

        # 2. Apply Augmentations
        # Albumentations expects HWC uint8 or float, process_patient returns float32 [0,1]
        # We apply the same transform to both views independently
        # Note: Ideally, spatial transforms should be consistent if they were 3D,
        # but for independent views, independent augmentation is acceptable regularization.

        # Axial
        aug_axial = self.transform(image=axial_img)["image"]
        # Coronal
        aug_coronal = self.transform(image=coronal_img)["image"]

        # 3. Prepare Tabular Features
        tabular_feats = self._encode_tabular(row)

        # 4. Prepare Scalar Metadata
        time_delta = torch.tensor([row["Time_Delta"]], dtype=torch.float32)
        baseline_fvc = torch.tensor([row["Baseline_FVC"]], dtype=torch.float32)

        # 5. Target
        # For test set, FVC might be dummy, but we return it anyway
        target = torch.tensor([row["FVC"]], dtype=torch.float32)

        return {
            "axial": aug_axial,  # (3, 224, 224)
            "coronal": aug_coronal,  # (3, 224, 224)
            "tabular": tabular_feats,  # (4,)
            "time_delta": time_delta,  # (1,)
            "baseline_fvc": baseline_fvc,  # (1,)
            "target": target,  # (1,)
            "patient_week": f"{patient_id}_{row['Weeks']}",  # For tracking/submission
        }
