import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.data_processing import process_patient


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Prediction.
    Handles loading of dual-view Tri-Slab images (Axial/Coronal) and tabular metadata.
    Implements caching via the library.data_processing module.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing patient metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transform pipeline.
                                             If None, defaults are created based on mode.
        """
        self.mode = mode
        self.df = df.copy()

        # --- Preprocessing & Column Unification ---
        if self.mode in ["train", "val"]:
            # In training, we need to identify the Baseline FVC for the slope calculation.
            # We assume the visit closest to Week 0 is the baseline.
            # The 'Weeks' column is already relative to baseline CT.

            # 1. Identify baseline FVC for each patient
            # We create a temporary column 'abs_week' to find the record closest to 0
            self.df["abs_week"] = self.df["Weeks"].abs()

            # Sort by patient and distance to week 0
            # We take the first entry for each patient as the baseline
            baseline_df = (
                self.df.sort_values(["Patient", "abs_week"])
                .groupby("Patient")
                .first()
                .reset_index()
            )

            # Map Baseline FVC back to the main dataframe
            patient_base_fvc = dict(zip(baseline_df["Patient"], baseline_df["FVC"]))
            self.df["Baseline_FVC"] = self.df["Patient"].map(patient_base_fvc)

            # Rename columns to match unified schema
            # Target is FVC, Input Time is Weeks
            self.df["Relative_Week"] = self.df["Weeks"]

        elif self.mode == "test":
            # In test, columns are prefixed with Baseline_
            # We need to map them to the standard names expected by the feature extractor

            # Calculate Relative Week
            self.df["Relative_Week"] = (
                self.df["Predict_Week"] - self.df["Baseline_Week"]
            )

            # Map features
            rename_map = {
                "Baseline_Age": "Age",
                "Baseline_Sex": "Sex",
                "Baseline_SmokingStatus": "SmokingStatus",
                "Baseline_Percent": "Percent",
                # Baseline_FVC is already named correctly in test.csv
            }
            self.df = self.df.rename(columns=rename_map)

            # Ensure FVC column exists (dummy for test) if not present
            if "FVC" not in self.df.columns:
                self.df["FVC"] = 2000.0  # Dummy value

        # --- Feature Normalization Constants (from EDA) ---
        self.age_mean = 67.58
        self.age_std = 6.63
        self.pct_mean = 76.91
        self.pct_std = 19.20

        # --- Augmentation Setup ---
        if transform is None:
            if self.mode == "train":
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                        ),
                        A.Normalize(mean=Config.MEAN, std=Config.STD),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(mean=Config.MEAN, std=Config.STD),
                        ToTensorV2(),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal)
        # process_patient handles caching internally
        # It returns numpy arrays (H, W, 3) in uint8 [0-255]
        dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
        try:
            axial_img, coronal_img = process_patient(
                patient_id, dicom_dir, load_cached_data=True
            )
        except Exception as e:
            # Fallback for corrupt data: return black images
            axial_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            coronal_img = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
            )

        # 2. Apply Augmentations
        # Note: We apply transforms independently.
        # While spatial consistency is ideal, orthogonal views (Axial/Coronal)
        # do not share a 2D coordinate system, so independent aug is acceptable/beneficial regularization.
        if self.transform:
            res_ax = self.transform(image=axial_img)
            axial_tensor = res_ax["image"]

            res_cor = self.transform(image=coronal_img)
            coronal_tensor = res_cor["image"]
        else:
            # Fallback to simple tensor conversion
            axial_tensor = (
                torch.from_numpy(axial_img.transpose(2, 0, 1)).float() / 255.0
            )
            coronal_tensor = (
                torch.from_numpy(coronal_img.transpose(2, 0, 1)).float() / 255.0
            )

        # 3. Process Tabular Features
        # Features: [Age, Sex, Smoking_Ex, Smoking_Never, Smoking_Current, Percent]

        # Age (Standardized)
        age = (float(row["Age"]) - self.age_mean) / self.age_std

        # Percent (Standardized)
        percent = (float(row["Percent"]) - self.pct_mean) / self.pct_std

        # Sex (Binary: Male=0, Female=1)
        sex = 1.0 if row["Sex"] == "Female" else 0.0

        # SmokingStatus (One-Hot)
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        ss = row["SmokingStatus"]
        smk_ex = 1.0 if ss == "Ex-smoker" else 0.0
        smk_never = 1.0 if ss == "Never smoked" else 0.0
        smk_current = 1.0 if ss == "Currently smokes" else 0.0

        # Construct vector
        # Dim = 1 + 1 + 1 + 3 = 6
        tabular_vector = np.array(
            [age, sex, smk_ex, smk_never, smk_current, percent], dtype=np.float32
        )
        tabular_tensor = torch.from_numpy(tabular_vector)

        # 4. Targets and Metadata
        target_fvc = float(row["FVC"])
        base_fvc = float(row["Baseline_FVC"])
        relative_week = float(row["Relative_Week"])

        return {
            "axial": axial_tensor,  # (3, 224, 224)
            "coronal": coronal_tensor,  # (3, 224, 224)
            "tabular": tabular_tensor,  # (6,)
            "fvc": torch.tensor(target_fvc, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "week": torch.tensor(relative_week, dtype=torch.float32),
            "patient_id": patient_id,
        }
