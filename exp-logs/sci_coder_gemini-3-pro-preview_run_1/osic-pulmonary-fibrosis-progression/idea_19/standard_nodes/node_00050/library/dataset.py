import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.dicom_processing import DicomProcessor


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Decline Prediction.

    Features:
    - Loads Dual-View (Axial/Coronal) Tri-Slab MIPs.
    - Processes Clinical Metadata (Age, Sex, Smoking, Percent).
    - Handles Baseline vs. Follow-up logic for parametric inference.
    - Applies spatial-only augmentations.
    """

    # Fixed statistics from EDA for Robust Scaling
    # These ensure consistency between Train/Val/Test without data leakage
    STATS = {
        "Age_mean": 67.58,
        "Age_std": 6.63,
        "Percent_mean": 76.91,
        "Percent_std": 19.20,
    }

    def __init__(self, mode="train", transform=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms (optional).
            load_cached_data (bool): Whether to load processed images from cache.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.dicom_processor = DicomProcessor()

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.df = self._prepare_train_data(self.df)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.df = self._prepare_train_data(self.df)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            self.df = self._prepare_test_data(self.df)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Setup Transforms
        if transform is None:
            if mode == "train":
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
                    [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
                )
        else:
            self.transform = transform

    def _prepare_train_data(self, df):
        """
        Prepares training/validation data by identifying baseline rows.
        For each patient, the visit closest to Week 0 is treated as the baseline.
        """
        # Ensure data is sorted
        df = df.sort_values(["Patient", "Weeks"])

        # Identify baseline for each patient (row with min absolute weeks)
        df["abs_weeks"] = df["Weeks"].abs()
        baseline_df = df.loc[df.groupby("Patient")["abs_weeks"].idxmin()].copy()

        # Select relevant baseline columns
        baseline_cols = [
            "Patient",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
        ]
        baseline_df = baseline_df[baseline_cols]

        # Rename to Baseline_ prefix
        rename_map = {c: f"Baseline_{c}" for c in baseline_cols if c != "Patient"}
        baseline_df = baseline_df.rename(columns=rename_map)

        # Merge back to original dataframe
        df = df.merge(baseline_df, on="Patient", how="left")

        # Calculate time delta
        df["dt"] = df["Weeks"] - df["Baseline_Weeks"]

        return df

    def _prepare_test_data(self, df):
        """
        Prepares test data. Metadata/test.csv already has Baseline_ columns.
        """
        # Calculate time delta (Predict_Week - Baseline_Week)
        df["dt"] = df["Predict_Week"] - df["Baseline_Week"]
        return df

    def _process_tabular_features(self, row):
        """
        Encodes and scales tabular features.
        Returns a float32 numpy array.
        """
        # 1. Numerical Scaling (Standardization)
        age_norm = (row["Baseline_Age"] - self.STATS["Age_mean"]) / self.STATS[
            "Age_std"
        ]
        percent_norm = (
            row["Baseline_Percent"] - self.STATS["Percent_mean"]
        ) / self.STATS["Percent_std"]

        # 2. Categorical Encoding (One-Hot)
        # Sex: Male (0), Female (1) -> [IsMale, IsFemale]
        sex = row["Baseline_Sex"]
        sex_vec = [1, 0] if sex == "Male" else [0, 1]

        # Smoking: Ex-smoker, Never smoked, Currently smokes
        # Map to 3 dim one-hot
        smoke = row["Baseline_SmokingStatus"]
        if smoke == "Ex-smoker":
            smoke_vec = [1, 0, 0]
        elif smoke == "Never smoked":
            smoke_vec = [0, 1, 0]
        else:  # Currently smokes
            smoke_vec = [0, 0, 1]

        # Concatenate: [Age, Percent, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2]
        # Total dims: 1 + 1 + 2 + 3 = 7
        features = np.array(
            [age_norm, percent_norm] + sex_vec + smoke_vec, dtype=np.float32
        )
        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Dual View)
        # Use dicom_dir from row if available, else construct it
        if "dicom_dir" in row:
            dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])
        else:
            # Fallback logic if dicom_dir missing
            folder = "train" if self.mode != "test" else "test"
            dicom_dir = os.path.join(Config.INPUT_DIR, folder, patient_id)

        # Generate or Load Tri-Slab MIPs
        # Returns numpy arrays (H, W, 3) in uint8 [0, 255]
        img_axial, img_coronal = self.dicom_processor.generate_dual_view_mips(
            patient_id, dicom_dir, load_cached_data=self.load_cached_data
        )

        # 2. Apply Augmentations
        # Albumentations expects dict
        if self.transform:
            # We apply independent spatial augmentations to views as they are orthogonal
            # and we want to encourage robustness.
            res_ax = self.transform(image=img_axial)
            img_axial_t = res_ax["image"]

            res_cor = self.transform(image=img_coronal)
            img_coronal_t = res_cor["image"]
        else:
            # Fallback to simple tensor conversion
            img_axial_t = torch.from_numpy(img_axial.transpose(2, 0, 1)).float() / 255.0
            img_coronal_t = (
                torch.from_numpy(img_coronal.transpose(2, 0, 1)).float() / 255.0
            )

        # 3. Process Tabular Data
        tabular_vec = self._process_tabular_features(row)

        # 4. Prepare Metadata & Targets
        # Baseline FVC is the anchor for the parametric model
        baseline_fvc = float(row["Baseline_FVC"])
        dt = float(row["dt"])

        # Target FVC (Ground Truth)
        # For test set, this is a dummy value (2000)
        target_fvc = float(row["FVC"]) if "FVC" in row else 0.0

        return {
            "image_axial": img_axial_t,  # (3, 224, 224)
            "image_coronal": img_coronal_t,  # (3, 224, 224)
            "tabular": torch.tensor(tabular_vec, dtype=torch.float32),  # (7,)
            "dt": torch.tensor(dt, dtype=torch.float32),  # Scalar
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),  # Scalar
            "target": torch.tensor(target_fvc, dtype=torch.float32),  # Scalar
            "patient_id": patient_id,
        }
