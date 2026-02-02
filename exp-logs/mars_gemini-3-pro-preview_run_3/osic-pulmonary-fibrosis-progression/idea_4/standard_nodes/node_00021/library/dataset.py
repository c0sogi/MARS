import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from library.config import Config
from library.image_processing import process_patient


class LungFVCDataset(Dataset):
    """
    PyTorch Dataset for Lung FVC Prediction.

    This dataset prepares inputs for the Dual-Path Transformer-Fused Network:
    1. Context Path: 3 Adaptive CT Slices + Demographic Metadata (Age, Sex, Smoking).
    2. Linear Path: Baseline FVC + Relative Weeks (Skip connection).
    """

    def __init__(self, df, mode="train", debug=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing Patient, Weeks, FVC, etc.
            mode (str): 'train', 'val', or 'test'.
            debug (bool): If True, restricts dataset size for debugging.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode

        if debug:
            print(f"DEBUG MODE: Limiting dataset to {Config.DEBUG_SIZE} samples.")
            self.df = self.df.iloc[: Config.DEBUG_SIZE]

        # Standardization Statistics (Fixed from EDA to ensure consistency)
        self.stats = {
            "fvc_mean": Config.TARGET_MEAN,
            "fvc_std": Config.TARGET_STD,
            "weeks_mean": 31.3751,
            "weeks_std": 23.4602,
            "age_mean": 67.5825,
            "age_std": 6.6259,
        }

        # Categorical Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Compute Baseline FVC for every patient in the dataframe.
        # The Baseline FVC is the measurement closest to Week 0 (CT Scan time).
        # This is crucial for the linear skip connection.
        self.baseline_lookup = self._compute_baselines(self.df)

    def _compute_baselines(self, df):
        """
        Identifies the FVC value closest to Week 0 for each patient.
        """
        lookup = {}
        # We need to find the row with min(abs(Weeks)) for each patient.
        # Create a temporary copy to sort without affecting the main df
        temp_df = df.copy()

        # If 'FVC' is missing (e.g. in pure submission file), we can't compute baseline from it.
        # We assume the passed dataframe has valid FVCs for the baseline rows (like test.csv).
        if "FVC" not in temp_df.columns:
            return {}

        temp_df["abs_weeks"] = temp_df["Weeks"].abs()

        # Sort by Patient and distance from Week 0
        temp_df = temp_df.sort_values(["Patient", "abs_weeks"])

        # Drop duplicates to keep the closest measurement
        baselines = temp_df.drop_duplicates("Patient", keep="first")

        for _, row in baselines.iterrows():
            lookup[row["Patient"]] = row["FVC"]

        return lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # ===========================
        # 1. Image Data (Path 1)
        # ===========================
        # Construct relative path. Metadata usually contains 'image_path'.
        # Fallback to standard directory structure if missing.
        rel_path = row.get("image_path")
        if pd.isna(rel_path):
            folder = "test" if self.mode == "test" else "train"
            rel_path = f"{folder}/{patient_id}"

        # Load cached images.
        # Returns np.array of shape (3, IMG_SIZE, IMG_SIZE)
        images_array = process_patient(patient_id, rel_path, load_cached_data=True)
        images = torch.tensor(images_array, dtype=torch.float32)

        # ===========================
        # 2. Meta Tokens (Path 1)
        # ===========================
        # Age (Standardized)
        age_std = (row["Age"] - self.stats["age_mean"]) / self.stats["age_std"]

        # Categoricals (Indices)
        sex_idx = self.sex_map.get(row["Sex"], 0)  # Default to 0 (Male) if unknown
        smoke_idx = self.smoke_map.get(
            row["SmokingStatus"], 1
        )  # Default to 1 (Never smoked) if unknown

        # ===========================
        # 3. Linear Features (Path 2)
        # ===========================
        # Baseline FVC (Standardized)
        # Retrieve the baseline FVC for this patient from lookup.
        # If not found (e.g. new patient in streaming), fallback to current FVC.
        base_fvc = self.baseline_lookup.get(patient_id, row.get("FVC", 0))
        base_fvc_std = (base_fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]

        # Weeks (Standardized)
        weeks_std = (row["Weeks"] - self.stats["weeks_mean"]) / self.stats["weeks_std"]

        linear_features = torch.tensor([base_fvc_std, weeks_std], dtype=torch.float32)

        # ===========================
        # 4. Target
        # ===========================
        if self.mode != "test" and "FVC" in row:
            target_val = row["FVC"]
            target_std = (target_val - self.stats["fvc_mean"]) / self.stats["fvc_std"]
            target = torch.tensor(target_std, dtype=torch.float32)
        else:
            # For test set, target is unknown
            target = torch.tensor(0.0, dtype=torch.float32)

        return {
            "images": images,  # Tensor (3, 256, 256)
            "meta_age": torch.tensor(age_std, dtype=torch.float32),
            "meta_sex": torch.tensor(sex_idx, dtype=torch.long),
            "meta_smoke": torch.tensor(smoke_idx, dtype=torch.long),
            "linear_features": linear_features,  # Tensor (2,)
            "target": target,  # Scalar Tensor
            "patient_id": patient_id,
            "weeks": row["Weeks"],
        }
