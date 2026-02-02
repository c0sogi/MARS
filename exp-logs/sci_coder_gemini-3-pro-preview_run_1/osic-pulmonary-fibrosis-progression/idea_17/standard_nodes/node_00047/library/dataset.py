import os
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
    Dataset class for the Full-Fidelity Concatenated Dual-Axis Network.
    Handles loading of metadata, on-the-fly (cached) generation of Tri-Slab images,
    and preparation of tabular/skip-connection features.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Custom augmentation pipeline.
        """
        self.mode = mode

        # 1. Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            if Config.DEBUG:
                self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)
            self.setup_baselines()
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            if Config.DEBUG:
                self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE)
            self.setup_baselines()
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            # Test set is already formatted with baseline info in columns
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Define Augmentations
        # Spatial only, no intensity changes as per strategy
        if transform:
            self.aug = transform
        elif mode == "train":
            self.aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

        # 3. Tabular Preprocessing Constants
        # Derived from EDA to ensure consistency across splits
        self.age_mean = 67.58
        self.age_std = 6.62
        self.pct_mean = 76.91
        self.pct_std = 19.20

        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def setup_baselines(self):
        """
        Identifies the baseline visit (closest to Week 0) for each patient in the training/val set.
        Creates a lookup dictionary to retrieve baseline features for any visit.
        """
        self.baseline_lookup = {}
        patients = self.df["Patient"].unique()

        for p in patients:
            p_data = self.df[self.df["Patient"] == p].copy()
            # Find row closest to week 0 (scan time)
            p_data["abs_week_diff"] = p_data["Weeks"].abs()
            baseline_row = p_data.sort_values("abs_week_diff").iloc[0]

            self.baseline_lookup[p] = {
                "Baseline_Week": baseline_row["Weeks"],
                "Baseline_FVC": baseline_row["FVC"],
                "Baseline_Percent": baseline_row["Percent"],
                "Baseline_Age": baseline_row["Age"],
                "Baseline_Sex": baseline_row["Sex"],
                "Baseline_SmokingStatus": baseline_row["SmokingStatus"],
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # ==========================================
        # 1. Retrieve Clinical Data (Baseline & Current)
        # ==========================================
        if self.mode in ["train", "val"]:
            baseline = self.baseline_lookup[patient_id]

            # Inputs (Baseline features)
            base_age = baseline["Baseline_Age"]
            base_sex = baseline["Baseline_Sex"]
            base_smoke = baseline["Baseline_SmokingStatus"]
            base_pct = baseline["Baseline_Percent"]
            base_fvc = baseline["Baseline_FVC"]
            base_week = baseline["Baseline_Week"]

            # Target Context
            current_week = row["Weeks"]
            target_fvc = row["FVC"]
            patient_week_id = f"{patient_id}_{current_week}"

        else:  # Test mode
            # Metadata/test.csv already has baseline info merged
            base_age = row["Baseline_Age"]
            base_sex = row["Baseline_Sex"]
            base_smoke = row["Baseline_SmokingStatus"]
            base_pct = row["Baseline_Percent"]
            base_fvc = row["Baseline_FVC"]
            base_week = row["Baseline_Week"]

            current_week = row["Predict_Week"]
            target_fvc = 0.0  # Dummy
            patient_week_id = row["Patient_Week"]

        # ==========================================
        # 2. Load & Augment Images
        # ==========================================
        dicom_dir = row["dicom_dir"]

        # Load from cache or process
        axial_img, coronal_img = process_patient(
            patient_id, dicom_dir, load_cached_data=True
        )

        # Apply Augmentations
        # Note: Albumentations works on numpy arrays
        aug_axial = self.aug(image=axial_img)["image"]
        aug_coronal = self.aug(image=coronal_img)["image"]

        # ==========================================
        # 3. Process Tabular Features
        # ==========================================
        # Normalize Numerical
        age_norm = (base_age - self.age_mean) / self.age_std
        pct_norm = (base_pct - self.pct_mean) / self.pct_std

        # One-Hot Encode Categorical
        sex_vec = [0, 0]
        if base_sex in self.sex_map:
            sex_vec[self.sex_map[base_sex]] = 1

        smoke_vec = [0, 0, 0]
        if base_smoke in self.smoke_map:
            smoke_vec[self.smoke_map[base_smoke]] = 1

        # Construct Tabular Embedding Vector (Input to MLP)
        # [Age, Percent, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2]
        tabular_feats = np.array(
            [age_norm, pct_norm] + sex_vec + smoke_vec, dtype=np.float32
        )

        # Construct Skip Connection Vector (Input to Head)
        # [Baseline_FVC_Scaled, Percent_Scaled]
        # Scaling FVC by 1000 to keep it in similar range to activations
        fvc_scaled = base_fvc / 1000.0
        skip_feats = np.array([fvc_scaled, pct_norm], dtype=np.float32)

        # Meta info for calculating prediction from slope
        time_delta = current_week - base_week

        return {
            "axial": aug_axial,  # (3, 224, 224)
            "coronal": aug_coronal,  # (3, 224, 224)
            "tabular": torch.from_numpy(tabular_feats),  # (7,)
            "skip": torch.from_numpy(skip_feats),  # (2,)
            "meta": torch.tensor(
                [base_fvc, time_delta], dtype=torch.float32
            ),  # [Base_FVC, Delta_Week]
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "patient_week": patient_week_id,
        }
