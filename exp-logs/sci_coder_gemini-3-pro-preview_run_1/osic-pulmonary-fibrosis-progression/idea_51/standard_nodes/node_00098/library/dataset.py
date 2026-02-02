import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.image_processing import process_patient


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Prediction.

    Handles:
    1. Loading and standardizing metadata (Train/Val/Test).
    2. Extracting Baseline Clinical Priors.
    3. Loading/Caching CT Scan Tri-Slabs (Axial/Coronal).
    4. Applying Spatial Augmentations.
    5. Encoding Tabular Features.
    """

    def __init__(self, df, mode="train", cache_dir=Config.CACHE_DIR):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe (train, val, or test).
            mode (str): 'train', 'val', or 'test'. Controls augmentation and data prep.
            cache_dir (str): Directory to store/load cached numpy image arrays.
        """
        self.mode = mode
        self.cache_dir = cache_dir

        # Mappings for categorical features
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Prepare the dataframe with standardized columns
        self.df = self._prepare_data(df.copy())

        # Define Augmentations (Spatial Only)
        if self.mode == "train":
            self.transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transforms = A.Compose([ToTensorV2()])

    def _prepare_data(self, df):
        """
        Standardizes the dataframe so that every row contains:
        - Baseline clinical info (Age, Sex, etc.)
        - Baseline FVC
        - Time Delta (Week - BaselineWeek)
        - Target FVC
        """
        # Columns we want to standardize
        # meta_age, meta_sex, meta_smoking, meta_percent, meta_baseline_fvc, meta_dt, target

        if "Predict_Week" in df.columns:
            # === TEST SET CASE ===
            # Columns in test.csv (from metadata):
            # Patient_Week, FVC (dummy), Confidence, Patient, Predict_Week,
            # Baseline_Week, Baseline_FVC, Baseline_Percent, Baseline_Age, ...

            df = df.rename(
                columns={
                    "Baseline_Age": "meta_age",
                    "Baseline_Sex": "meta_sex",
                    "Baseline_SmokingStatus": "meta_smoking",
                    "Baseline_Percent": "meta_percent",
                    "Baseline_FVC": "meta_baseline_fvc",
                }
            )

            df["meta_dt"] = df["Predict_Week"] - df["Baseline_Week"]
            df["target"] = 0.0  # Dummy target for inference

        else:
            # === TRAIN/VAL SET CASE ===
            # Columns: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus, dicom_dir
            # We need to find the baseline row for each patient to populate meta_* columns

            # 1. Identify baseline rows (closest to Week 0)
            df["abs_week"] = df["Weeks"].abs()
            # Sort by patient and abs_week so the first record is the closest to baseline
            df_sorted = df.sort_values(["Patient", "abs_week"])
            baseline_df = df_sorted.drop_duplicates(subset=["Patient"], keep="first")

            # 2. Select and rename baseline columns
            baseline_cols = {
                "Patient": "Patient",
                "Age": "meta_age",
                "Sex": "meta_sex",
                "SmokingStatus": "meta_smoking",
                "Percent": "meta_percent",
                "FVC": "meta_baseline_fvc",
                "Weeks": "baseline_week_ref",
            }
            baseline_info = baseline_df[list(baseline_cols.keys())].rename(
                columns=baseline_cols
            )

            # 3. Merge baseline info back to the main dataframe
            df = pd.merge(df, baseline_info, on="Patient", how="left")

            # 4. Calculate Delta and Target
            df["meta_dt"] = df["Weeks"] - df["baseline_week_ref"]
            df["target"] = df["FVC"]

        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial & Coronal)
        # Construct full path to DICOM directory
        dicom_path = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])

        # Load or compute images (returns dict {'axial': np.array, 'coronal': np.array})
        # Arrays are float32 [0, 1], shape (224, 224, 3)
        images = process_patient(
            patient_id=patient_id,
            dicom_dir=dicom_path,
            cache_dir=self.cache_dir,
            load_cached=True,
        )

        img_ax = images["axial"]
        img_cor = images["coronal"]

        # 2. Apply Augmentations
        # We apply transforms independently.
        # Note: Albumentations expects HWC. ToTensorV2 converts to CHW.
        if self.transforms:
            res_ax = self.transforms(image=img_ax)
            img_ax_t = res_ax["image"]

            res_cor = self.transforms(image=img_cor)
            img_cor_t = res_cor["image"]

        # 3. Process Tabular Features
        # Encoding
        sex_enc = self.sex_map.get(row["meta_sex"], 0)
        smoke_enc = self.smoke_map.get(
            row["meta_smoking"], 1
        )  # Default to Never if missing

        # Normalization (Approximate scaling for NN stability)
        # Age: (x - 30) / 70 -> Maps 30-100 to 0-1
        age_norm = (float(row["meta_age"]) - 30.0) / 70.0
        # Percent: x / 100 -> Maps 0-100% to 0-1
        percent_norm = float(row["meta_percent"]) / 100.0

        tabular_vec = torch.tensor(
            [age_norm, float(sex_enc), float(smoke_enc), percent_norm],
            dtype=torch.float32,
        )

        # 4. Prepare Scalars
        time_delta = torch.tensor([float(row["meta_dt"])], dtype=torch.float32)
        baseline_fvc = torch.tensor(
            [float(row["meta_baseline_fvc"])], dtype=torch.float32
        )
        target = torch.tensor([float(row["target"])], dtype=torch.float32)

        return {
            "img_axial": img_ax_t,  # (3, 224, 224)
            "img_coronal": img_cor_t,  # (3, 224, 224)
            "tabular": tabular_vec,  # (4,)
            "time_delta": time_delta,  # (1,)
            "baseline_fvc": baseline_fvc,  # (1,)
            "target": target,  # (1,)
            "patient_id": patient_id,  # string, useful for debugging/submission
        }
