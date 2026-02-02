import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.dicom_processing import generate_dual_view_tri_slabs


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Prediction.

    Features:
    - Loads Axial and Coronal Tri-Slab images (cached).
    - Manages Baseline extraction for longitudinal data.
    - Tokenizes tabular features for the Granular-Tabular Network.
    - Returns scalar priors for skip connections.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose, optional): Custom transforms.
        """
        self.mode = mode
        self.root_dir = Config.INPUT_ROOT

        # 1. Load Metadata and Prepare Dataframe
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.df = self._prepare_train_val_data(self.df)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.df = self._prepare_train_val_data(self.df)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            # Test CSV already has Baseline_FVC, Baseline_Percent, Baseline_Week, Predict_Week
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Define Augmentations
        if transform is None:
            if mode == "train":
                self.transform = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                        ),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = transform

        # 3. Feature Encoders
        self.sex_mapper = {"Male": 0, "Female": 1}
        self.smoke_mapper = {"Ex-smoker": 0, "Never smoker": 1, "Currently smokes": 2}

    def _prepare_train_val_data(self, df):
        """
        Identifies the baseline visit (min Weeks) for each patient and
        merges baseline info (FVC, Percent, Week) back to the dataframe.
        """
        # Sort by Patient and Weeks to ensure first is baseline
        df = df.sort_values(["Patient", "Weeks"])

        # Extract baseline rows
        baseline_df = df.groupby("Patient").first().reset_index()

        # Select relevant columns and rename
        baseline_cols = {
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Weeks": "Baseline_Week",
            "Age": "Baseline_Age",  # Use baseline age for consistency
        }
        baseline_df = baseline_df[["Patient"] + list(baseline_cols.keys())]
        baseline_df = baseline_df.rename(columns=baseline_cols)

        # Merge back to original dataframe
        # We use a left join to propagate baseline info to all visits
        df = pd.merge(df, baseline_df, on="Patient", how="left")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # ==========================================
        # 1. Image Loading & Processing
        # ==========================================
        # Construct full path to DICOM directory
        # metadata contains relative path in 'dicom_dir'
        dicom_dir = os.path.join(self.root_dir, row["dicom_dir"])

        # Load Dual-View Tri-Slabs (Cached)
        # Returns (224, 224, 3) uint8 arrays
        axial_img, coronal_img = generate_dual_view_tri_slabs(
            patient_id, dicom_dir, load_cached_data=True
        )

        # Apply Augmentations
        # We apply the same transform pipeline but independently (random seeds differ)
        # This acts as regularization for the dual-path network
        if self.transform:
            aug_axial = self.transform(image=axial_img)["image"]
            aug_coronal = self.transform(image=coronal_img)["image"]
        else:
            # Fallback to simple tensor conversion
            aug_axial = torch.from_numpy(axial_img.transpose(2, 0, 1)).float() / 255.0
            aug_coronal = (
                torch.from_numpy(coronal_img.transpose(2, 0, 1)).float() / 255.0
            )

        # ==========================================
        # 2. Tabular Token Preparation
        # ==========================================
        # We use Baseline values for features to ensure consistency during inference

        # Age (Continuous) -> Normalize
        # Use Baseline_Age if available (from _prepare_train_val_data), else row['Age']
        age_raw = row.get("Baseline_Age", row.get("Age", Config.AGE_MEAN))
        age_norm = (age_raw - Config.AGE_MEAN) / Config.AGE_STD

        # Percent (Continuous) -> Normalize
        # Must use Baseline_Percent as current Percent is unknown at test time
        pct_raw = row["Baseline_Percent"]
        pct_norm = (pct_raw - Config.PERCENT_MEAN) / Config.PERCENT_STD

        # Categorical
        sex_enc = self.sex_mapper.get(row["Sex"], 0)
        smoke_enc = self.smoke_mapper.get(row["SmokingStatus"], 0)

        # ==========================================
        # 3. Priors & Time Delta
        # ==========================================
        # Time Delta: The time elapsed since the baseline measurement
        if self.mode == "test":
            current_week = row["Predict_Week"]
        else:
            current_week = row["Weeks"]

        baseline_week = row["Baseline_Week"]
        time_delta = float(current_week - baseline_week)

        # Scalar Priors for Skip Connection
        # We pass raw values; the model handles them or uses them for linear projection
        baseline_fvc = float(row["Baseline_FVC"])
        baseline_pct = float(row["Baseline_Percent"])

        # ==========================================
        # 4. Target
        # ==========================================
        if self.mode == "test":
            target = 0.0  # Dummy
        else:
            target = float(row["FVC"])

        return {
            "axial": aug_axial,  # (3, 224, 224)
            "coronal": aug_coronal,  # (3, 224, 224)
            "age": torch.tensor(age_norm, dtype=torch.float32),
            "sex": torch.tensor(sex_enc, dtype=torch.long),
            "smoke": torch.tensor(smoke_enc, dtype=torch.long),
            "percent": torch.tensor(pct_norm, dtype=torch.float32),
            "priors": torch.tensor([baseline_fvc, baseline_pct], dtype=torch.float32),
            "time_delta": torch.tensor(time_delta, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{current_week}"
            ),
        }
