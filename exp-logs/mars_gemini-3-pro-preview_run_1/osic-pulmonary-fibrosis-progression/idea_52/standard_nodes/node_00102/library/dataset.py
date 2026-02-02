import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.image_processing import generate_orthogonal_tri_slabs


class LungDataset(Dataset):
    """
    PyTorch Dataset for the NSL-HN model.

    Handles loading of metadata, aligning patient history to baseline features,
    fetching cached Tri-Slab images, and applying augmentations.
    """

    def __init__(self, mode="train", split="train"):
        """
        Args:
            mode (str): 'train' or 'inference'. Controls augmentation application.
            split (str): 'train', 'val', or 'test'. Controls which metadata file to load.
        """
        self.mode = mode
        self.split = split

        # 1. Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            self._prepare_train_val_data()
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
            self._prepare_train_val_data()
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_METADATA_PATH)
            self._prepare_test_data()
        else:
            raise ValueError(f"Unknown split: {split}")

        # 2. Debug Mode: Slice dataset if enabled
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SIZE].copy()
            print(f"DEBUG MODE: Reduced {split} dataset to {len(self.df)} samples.")

        # 3. Define Augmentations
        # Note: Input images are float32 in [0, 1]. We set max_pixel_value=1.0 for Normalize.
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    # Spatial shifts and rotations allowed; strictly no intensity/contrast changes
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=10,
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1.0,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1.0,
                    ),
                    ToTensorV2(),
                ]
            )

        # 4. Feature Encoding Maps
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoking_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def _prepare_train_val_data(self):
        """
        Prepares training/validation data by identifying baseline features for each patient.
        The model must predict current FVC based on *Baseline* features + Time Delta.
        """
        # Identify baseline row for each patient (row with minimum Weeks)
        baseline_df = (
            self.df.sort_values("Weeks").groupby("Patient").first().reset_index()
        )

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

        # Rename columns to indicate they are baseline features
        baseline_df.columns = [
            "Patient",
            "Base_FVC",
            "Base_Percent",
            "Base_Age",
            "Base_Sex",
            "Base_Smoking",
            "Base_Week",
        ]

        # Merge baseline info back onto the full visit history
        self.df = pd.merge(self.df, baseline_df, on="Patient", how="left")

        # Calculate Delta Week (Time elapsed since baseline)
        self.df["Delta_Week"] = self.df["Weeks"] - self.df["Base_Week"]

    def _prepare_test_data(self):
        """
        Prepares test data. The test metadata already contains 'Baseline_' columns.
        """
        # Map existing columns to internal standard names
        self.df["Base_FVC"] = self.df["Baseline_FVC"]
        self.df["Base_Percent"] = self.df["Baseline_Percent"]
        self.df["Base_Age"] = self.df["Baseline_Age"]
        self.df["Base_Sex"] = self.df["Baseline_Sex"]
        self.df["Base_Smoking"] = self.df["Baseline_SmokingStatus"]

        # Calculate Delta Week (Predict_Week - Baseline_Week)
        self.df["Delta_Week"] = self.df["Predict_Week"] - self.df["Baseline_Week"]

        # Ensure FVC target exists (dummy 0 for test set)
        if "FVC" not in self.df.columns:
            self.df["FVC"] = 0.0

    def _get_tabular_features(self, row):
        """
        Constructs the normalized tabular feature vector.
        Vector: [Age_norm, Sex_enc, Smoke_0, Smoke_1, Smoke_2, Percent_norm]
        """
        # 1. Age: Normalize (approx range 50-90 -> -1 to 1.5)
        age_norm = (row["Base_Age"] - 65.0) / 15.0

        # 2. Percent: Normalize (approx range 30-150 -> 0.3 to 1.5)
        percent_norm = row["Base_Percent"] / 100.0

        # 3. Sex: Binary
        sex_val = self.sex_map.get(row["Base_Sex"], 0)

        # 4. Smoking: One-Hot (3 classes)
        smoke_idx = self.smoking_map.get(
            row["Base_Smoking"], 1
        )  # Default to Never smoked
        smoke_vec = [0, 0, 0]
        smoke_vec[smoke_idx] = 1

        # Assemble vector (dim = 6)
        features = np.array(
            [age_norm, sex_val] + smoke_vec + [percent_norm], dtype=np.float32
        )
        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images (Axial and Coronal Tri-Slabs)
        # dicom_dir is relative path e.g., "train/ID..."
        full_dicom_path = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        # Load from cache or generate (returns float32 [0,1] HWC)
        axial_img, coronal_img = generate_orthogonal_tri_slabs(
            full_dicom_path, load_cached_data=True
        )

        # 2. Apply Transforms
        # Independent augmentation for orthogonal views to maximize regularization
        if self.transform:
            res_ax = self.transform(image=axial_img)
            axial_tensor = res_ax["image"]

            res_cor = self.transform(image=coronal_img)
            coronal_tensor = res_cor["image"]
        else:
            # Fallback manual conversion (HWC -> CHW)
            axial_tensor = torch.from_numpy(axial_img.transpose(2, 0, 1))
            coronal_tensor = torch.from_numpy(coronal_img.transpose(2, 0, 1))

        # 3. Tabular Features
        tab_vec = self._get_tabular_features(row)
        tab_tensor = torch.from_numpy(tab_vec)

        # 4. Targets and Meta-parameters
        target_fvc = torch.tensor(row["FVC"], dtype=torch.float32)
        base_fvc = torch.tensor(row["Base_FVC"], dtype=torch.float32)
        delta_week = torch.tensor(row["Delta_Week"], dtype=torch.float32)

        # Identifier for submission (Patient_Week)
        # For train/val, we construct it; for test, it exists.
        pw_id = (
            row["Patient_Week"]
            if "Patient_Week" in row
            else f"{row['Patient']}_{row['Weeks']}"
        )

        return {
            "axial": axial_tensor,  # (3, 224, 224)
            "coronal": coronal_tensor,  # (3, 224, 224)
            "tabular": tab_tensor,  # (6,)
            "fvc_target": target_fvc,  # Scalar
            "base_fvc": base_fvc,  # Scalar (used for projection)
            "delta_week": delta_week,  # Scalar (used for projection)
            "patient_week_id": pw_id,  # String
        }
