import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.image_processing import cache_images


class LungDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", transform=None, load_cached_data=True
    ):
        """
        PyTorch Dataset for Lung Function Decline Prediction.

        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentations to apply to images.
            load_cached_data (bool): If True, attempts to load pre-processed .npy files.
                                     If False or files missing, triggers processing.
        """
        self.mode = mode
        self.transform = transform
        self.cache_dir = Config.CACHE_DIR

        # 1. Ensure Data Availability (Caching Mechanism)
        # We trigger the caching process here. The cache_images function handles
        # checking for existing files and processing in parallel if needed.
        if mode in ["train", "val", "test"]:
            # Only print for the main process to avoid clutter if multiple workers were initializing
            # (though Dataset init usually happens in main process)
            print(
                f"[{mode.upper()}] Checking/Generating image cache at {self.cache_dir}..."
            )
            cache_images(metadata_path, load_cached_data=load_cached_data)

        # 2. Load and Preprocess Metadata
        self.df = pd.read_csv(metadata_path)
        self.df = self._preprocess_metadata(self.df, mode)

        # 3. Encoders (Hardcoded for simplicity and consistency across splits)
        self.sex_map = {"Male": 0.0, "Female": 1.0}
        self.smoking_map = {
            "Ex-smoker": 0.0,
            "Never smoked": 0.5,
            "Currently smokes": 1.0,
        }

    def _preprocess_metadata(self, df, mode):
        """
        Prepares the dataframe by extracting baseline values and calculating time deltas.
        """
        data = df.copy()

        if mode in ["train", "val"]:
            # Sort to ensure we can pick the earliest visit as baseline
            data = data.sort_values(["Patient", "Weeks"])

            # Group by Patient to extract baseline characteristics (first row per patient)
            # We rename these columns to 'Baseline_X'
            baseline_df = data.groupby("Patient").first().reset_index()
            cols = ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus", "Weeks"]
            baseline_df = baseline_df[cols]
            baseline_df.columns = [
                "Patient",
                "Baseline_FVC",
                "Baseline_Percent",
                "Baseline_Age",
                "Baseline_Sex",
                "Baseline_SmokingStatus",
                "Baseline_Week",
            ]

            # Merge baseline info back to the original dataframe
            data = pd.merge(data, baseline_df, on="Patient", how="left")

            # Calculate time delta: Current Week - Baseline Week
            data["Week_Delta"] = data["Weeks"] - data["Baseline_Week"]

        elif mode == "test":
            # Test metadata already contains Baseline columns and Predict_Week
            # We just need to calculate the delta
            data["Week_Delta"] = data["Predict_Week"] - data["Baseline_Week"]

        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # -----------------------------------------------------------
        # 1. Load Dual-View Images
        # -----------------------------------------------------------
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # Load .npy files (uint8 0-255)
        try:
            img_axial = np.load(axial_path)
            img_coronal = np.load(coronal_path)
        except Exception:
            # Fallback for safety (should be handled by cache_images)
            img_axial = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
            )
            img_coronal = np.zeros(
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
            )

        # -----------------------------------------------------------
        # 2. Apply Augmentations
        # -----------------------------------------------------------
        if self.transform:
            # Apply transforms independently to each view
            aug_axial = self.transform(image=img_axial)["image"]
            aug_coronal = self.transform(image=img_coronal)["image"]
        else:
            # Default transform (Normalize + ToTensor)
            default_t = get_transforms(mode="val")
            aug_axial = default_t(image=img_axial)["image"]
            aug_coronal = default_t(image=img_coronal)["image"]

        # -----------------------------------------------------------
        # 3. Prepare Tabular Features
        # -----------------------------------------------------------
        # Extract raw values
        age = float(row["Baseline_Age"])
        percent = float(row["Baseline_Percent"])
        base_fvc = float(row["Baseline_FVC"])
        sex = row["Baseline_Sex"]
        smoking = row["Baseline_SmokingStatus"]

        # Normalize Numerical Features (Approximate Min-Max scaling)
        # Age: ~30-90 -> 0-1
        age_norm = (age - 30.0) / 60.0
        # Percent: ~50-150 -> 0-1
        percent_norm = (percent - 50.0) / 100.0
        # Base FVC: ~1000-6000 -> 0-1
        base_fvc_norm = (base_fvc - 1000.0) / 5000.0

        # Encode Categorical Features
        sex_norm = self.sex_map.get(sex, 0.0)
        smoking_norm = self.smoking_map.get(smoking, 0.0)

        # Construct Tabular Vector for MLP
        # [Age, Sex, Smoking, Percent, Baseline_FVC]
        tabular_vector = torch.tensor(
            [age_norm, sex_norm, smoking_norm, percent_norm, base_fvc_norm],
            dtype=torch.float32,
        )

        # -----------------------------------------------------------
        # 4. Prepare Outputs
        # -----------------------------------------------------------
        week_delta = float(row["Week_Delta"])

        result = {
            "image_axial": aug_axial,
            "image_coronal": aug_coronal,
            "tabular": tabular_vector,
            "week_delta": torch.tensor(week_delta, dtype=torch.float32),
            "baseline_fvc": torch.tensor(base_fvc, dtype=torch.float32),
        }

        # Add target for training/validation
        if self.mode != "test":
            target_fvc = float(row["FVC"])
            result["target"] = torch.tensor(target_fvc, dtype=torch.float32)

        return result


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specific mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # Spatial augmentations only
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # Normalize to ImageNet stats
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
