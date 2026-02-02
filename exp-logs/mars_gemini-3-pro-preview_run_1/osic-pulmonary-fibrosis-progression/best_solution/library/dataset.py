import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.image_processing import generate_dual_views


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Decline Prediction.

    Features:
    - Loads Dual-Axis Tri-Slab MIPs (Axial + Coronal).
    - Manages caching via library.image_processing.
    - Pre-calculates patient baselines for trajectory modeling.
    - applies spatial augmentations while preserving HU intensity.
    """

    def __init__(self, mode="train", cache_dir="./working/idea_5", load_cached=True):
        self.mode = mode
        self.root_dir = "./input"
        self.cache_dir = cache_dir
        self.load_cached = load_cached

        # Ensure cache directory exists (though library function also handles it)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load appropriate metadata
        if mode == "train":
            self.df = pd.read_csv("./metadata/train.csv")
        elif mode == "val":
            self.df = pd.read_csv("./metadata/val.csv")
        elif mode == "test":
            self.df = pd.read_csv("./metadata/test.csv")
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

        # Pre-calculate baseline info for train/val sets
        # The model predicts FVC = Baseline_FVC + alpha * t
        # So we need to identify the baseline parameters for every row.
        if mode in ["train", "val"]:
            self.patient_baselines = {}
            patients = self.df["Patient"].unique()
            for p in patients:
                p_data = self.df[self.df["Patient"] == p]
                # Identify baseline row: The visit closest to Week 0
                idx_min = p_data["Weeks"].abs().idxmin()
                baseline_row = p_data.loc[idx_min]

                self.patient_baselines[p] = {
                    "FVC": float(baseline_row["FVC"]),
                    "Weeks": int(baseline_row["Weeks"]),
                    "Percent": float(baseline_row["Percent"]),
                    "Age": float(baseline_row["Age"]),
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                }

        # Define Augmentations
        # Note: Images are loaded as float [0,1], converted to uint8 [0,255] for Albumentations
        # Normalize will scale them back using ImageNet stats.
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=10,
                        p=0.5,
                        border_mode=0,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image Data (Dual-Axis Tri-Slab MIPs)
        dicom_rel_path = row["dicom_dir"]
        dicom_full_path = os.path.join(self.root_dir, dicom_rel_path)

        # Returns shape (2, 224, 224, 3) -> [Axial, Coronal]
        # Values are float32 in [0, 1]
        dual_view_images = generate_dual_views(
            dicom_full_path, patient_id, load_cached_data=self.load_cached
        )

        # Convert to uint8 for Albumentations
        images_uint8 = (dual_view_images * 255).astype(np.uint8)

        # Apply transforms independently to each view
        # We process them separately to allow independent spatial variations if needed,
        # though here we just want to get them into tensor format.
        view_axial = self.transform(image=images_uint8[0])["image"]
        view_coronal = self.transform(image=images_uint8[1])["image"]

        # Stack back to (2, 3, 224, 224)
        image_tensor = torch.stack([view_axial, view_coronal], dim=0)

        # 2. Extract Clinical Features & Targets
        if self.mode in ["train", "val"]:
            # Retrieve pre-calculated baseline info
            base_info = self.patient_baselines[patient_id]

            age = base_info["Age"]
            sex = base_info["Sex"]
            smoke = base_info["SmokingStatus"]
            base_percent = base_info["Percent"]
            base_fvc = base_info["FVC"]
            base_week = base_info["Weeks"]

            current_fvc = float(row["FVC"])
            current_week = int(row["Weeks"])
            patient_week_id = f"{patient_id}_{current_week}"

        else:  # Test mode
            # Metadata already contains baseline info merged
            age = float(row["Baseline_Age"])
            sex = row["Baseline_Sex"]
            smoke = row["Baseline_SmokingStatus"]
            base_percent = float(row["Baseline_Percent"])
            base_fvc = float(row["Baseline_FVC"])
            base_week = int(row["Baseline_Week"])

            current_fvc = 0.0  # Dummy
            current_week = int(row["Predict_Week"])
            patient_week_id = str(row["Patient_Week"])

        # 3. Normalize Tabular Features
        # Statistics derived from EDA: Age Mean~67.6 Std~6.6, Percent Mean~76.9 Std~19.2
        feat_age = (age - 67.58) / 6.62
        feat_percent = (base_percent - 76.91) / 19.19

        # Sex: Male=0, Female=1
        feat_sex = 0.0 if sex == "Male" else 1.0

        # Smoking: One-Hot Encoding [Ex, Never, Current]
        feat_smoke = [0.0, 0.0, 0.0]
        if smoke == "Ex-smoker":
            feat_smoke[0] = 1.0
        elif smoke == "Never smoked":
            feat_smoke[1] = 1.0
        elif smoke == "Currently smokes":
            feat_smoke[2] = 1.0

        # Construct Tabular Vector: Size 6
        tabular_vector = np.array(
            [feat_age, feat_percent, feat_sex] + feat_smoke, dtype=np.float32
        )

        # Calculate Time Delta (t)
        time_delta = float(current_week - base_week)

        return {
            "image": image_tensor,  # (2, 3, 224, 224)
            "tabular": torch.tensor(tabular_vector, dtype=torch.float32),
            "time": torch.tensor(time_delta, dtype=torch.float32),
            "fvc": torch.tensor(current_fvc, dtype=torch.float32),
            "baseline_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "patient_week": patient_week_id,
        }
