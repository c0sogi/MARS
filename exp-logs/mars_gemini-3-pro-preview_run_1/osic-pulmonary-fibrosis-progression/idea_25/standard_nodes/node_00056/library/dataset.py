import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.image_processing import process_patient


class LungDataset(Dataset):
    """
    Dataset for the Identity-Aware Symmetric Dual-Axis Network.
    Loads Axial and Coronal Tri-Slab MIPs, processes tabular metadata,
    and aligns patient history relative to the baseline visit.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.mode = mode
        self.root_dir = Config.INPUT_ROOT

        # 1. Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Preprocess Tabular Data & Calculate Baselines
        self._prepare_tabular_data()

        # 3. Define Transforms
        # Spatial-only augmentation for training (No brightness/contrast changes)
        if transform is None:
            if mode == "train":
                self.transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.05,
                            scale_limit=0.1,
                            rotate_limit=10,
                            p=0.5,
                            border_mode=cv2.BORDER_CONSTANT,
                        ),
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

    def _prepare_tabular_data(self):
        """
        Calculates baseline features for train/val sets and fits scalers.
        """
        # A. Handle Baseline Logic for Train/Val
        # The model predicts FVC based on Baseline Info + Time Delta.
        # We must identify the baseline visit for each patient in the history.
        if self.mode in ["train", "val"]:
            # Identify the row closest to Week 0 for each patient
            self.df["abs_weeks"] = self.df["Weeks"].abs()

            # Group by patient and take the first row (min abs_weeks)
            baseline_df = (
                self.df.sort_values("abs_weeks")
                .groupby("Patient")
                .first()
                .reset_index()
            )

            # Columns to merge back as baseline context
            baseline_cols = [
                "Patient",
                "Weeks",
                "FVC",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
            ]
            baseline_subset = baseline_df[baseline_cols].copy()

            # Rename columns to Baseline_...
            rename_map = {c: f"Baseline_{c}" for c in baseline_cols if c != "Patient"}
            # Special handle for Weeks -> Baseline_Week
            rename_map["Weeks"] = "Baseline_Week"
            baseline_subset = baseline_subset.rename(columns=rename_map)

            # Merge baseline info onto every visit row
            self.df = pd.merge(self.df, baseline_subset, on="Patient", how="left")

            # Calculate Time Delta (Current Week - Baseline Week)
            self.df["time_delta"] = self.df["Weeks"] - self.df["Baseline_Week"]

        elif self.mode == "test":
            # Test CSV already contains Baseline_FVC, Baseline_Percent, etc.
            # And Predict_Week, Baseline_Week
            self.df["time_delta"] = self.df["Predict_Week"] - self.df["Baseline_Week"]

        # B. Fit Scalers on Training Data
        # We load the training set specifically to compute global stats for consistency
        train_df = pd.read_csv(Config.TRAIN_CSV)
        train_df["abs_weeks"] = train_df["Weeks"].abs()
        # Get unique patients (baselines) to avoid bias from frequent visitors
        train_unique = (
            train_df.sort_values("abs_weeks").groupby("Patient").first().reset_index()
        )

        self.age_mean = train_unique["Age"].mean()
        self.age_std = train_unique["Age"].std()
        self.pct_mean = train_unique["Percent"].mean()
        self.pct_std = train_unique["Percent"].std()

        # C. Categorical Maps
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        if Config.DEBUG:
            return min(len(self.df), 32)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial & Coronal)
        # process_patient handles caching to working/idea_25/cache
        dicom_dir = row["dicom_dir"]
        ax_img, cor_img = process_patient(patient_id, dicom_dir, load_cached_data=True)

        # 2. Apply Transforms
        # Apply transforms independently to each view
        if self.transform:
            res_ax = self.transform(image=ax_img)
            ax_tensor = res_ax["image"]

            res_cor = self.transform(image=cor_img)
            cor_tensor = res_cor["image"]
        else:
            # Fallback to simple normalization
            ax_tensor = torch.from_numpy(ax_img.transpose(2, 0, 1)).float() / 255.0
            cor_tensor = torch.from_numpy(cor_img.transpose(2, 0, 1)).float() / 255.0

        # 3. Prepare Tabular Vector
        # We use the BASELINE features for the model input
        age_norm = (row["Baseline_Age"] - self.age_mean) / self.age_std
        pct_norm = (row["Baseline_Percent"] - self.pct_mean) / self.pct_std

        sex_val = self.sex_map.get(row["Baseline_Sex"], 0)
        smoke_val = self.smoke_map.get(row["Baseline_SmokingStatus"], 0)

        # One-hot encode smoking (3 classes)
        smoke_ohe = [0.0, 0.0, 0.0]
        smoke_ohe[smoke_val] = 1.0

        # Construct Feature Vector
        # [Age, Sex, Smoke_0, Smoke_1, Smoke_2, Percent]
        # Size: 1 + 1 + 3 + 1 = 6
        tab_features = [age_norm, float(sex_val)] + smoke_ohe + [pct_norm]
        tab_tensor = torch.tensor(tab_features, dtype=torch.float32)

        # 4. Prepare Anchors and Targets
        time_delta = torch.tensor([row["time_delta"]], dtype=torch.float32)
        baseline_fvc = torch.tensor([row["Baseline_FVC"]], dtype=torch.float32)

        data = {
            "axial_img": ax_tensor,
            "coronal_img": cor_tensor,
            "tabular": tab_tensor,
            "time_delta": time_delta,
            "baseline_fvc": baseline_fvc,
            "patient_id": patient_id,
        }

        # Add target if available
        if self.mode != "test":
            target = torch.tensor([row["FVC"]], dtype=torch.float32)
            data["target"] = target

        return data
