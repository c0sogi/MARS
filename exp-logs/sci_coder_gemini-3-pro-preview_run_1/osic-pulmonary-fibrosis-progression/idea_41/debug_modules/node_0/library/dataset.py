import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from library.config import Config
from library.data_utils import process_patient


class LungDataset(Dataset):
    """
    PyTorch Dataset for the H2-DAN model.
    Handles loading of CT scans (Axial/Coronal Tri-Slabs) and Tabular Clinical Data.
    """

    def __init__(self, mode="train", transform=None, scaler=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Optional image transformations.
            scaler (sklearn object): Pre-fitted scaler/encoder pipeline. If None, fits on data.
        """
        self.mode = mode
        self.transform = transform
        self.scaler = scaler

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.df = self._prepare_training_data(self.df)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.df = self._prepare_training_data(self.df)
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            # Test data already has Baseline_ columns from metadata step
            # We just need to ensure Delta_Week is calculated
            self.df["Delta_Week"] = self.df["Predict_Week"] - self.df["Baseline_Week"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Debugging: Subset data if configured
        if Config.DEBUG:
            print(
                f"[DEBUG] Subsetting {mode} dataset to {Config.DEBUG_SAMPLE_SIZE} samples."
            )
            self.df = self.df.head(Config.DEBUG_SAMPLE_SIZE).copy()

        # Prepare Tabular Processors
        self._prepare_tabular_features()

    def _prepare_training_data(self, df):
        """
        For train/val sets, identifies the baseline row (Week approx 0) for each patient
        and broadcasts baseline features to all rows.
        """
        # Ensure data is sorted by patient and week
        df = df.sort_values(["Patient", "Weeks"])

        # We need to find the baseline row for each patient.
        # Ideally, this is where Weeks == 0. If not present, take the row with min absolute weeks.
        # However, for this dataset, the CT scan is associated with the first measurement.
        # We'll assume the first record per patient (sorted by Weeks) represents the baseline state
        # closest to the CT scan acquisition.

        baseline_df = df.drop_duplicates(subset=["Patient"], keep="first").copy()

        # Rename columns to Baseline_
        baseline_cols = {
            "Weeks": "Baseline_Week",
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Age": "Baseline_Age",
            "Sex": "Baseline_Sex",
            "SmokingStatus": "Baseline_SmokingStatus",
        }
        baseline_df = baseline_df[["Patient"] + list(baseline_cols.keys())]
        baseline_df = baseline_df.rename(columns=baseline_cols)

        # Merge back to original dataframe
        df = pd.merge(df, baseline_df, on="Patient", how="left")

        # Calculate Delta Week (Time since baseline)
        df["Delta_Week"] = df["Weeks"] - df["Baseline_Week"]

        return df

    def _prepare_tabular_features(self):
        """
        Configures the scaler/encoder for tabular data.
        Deep Features: Age, Percent, Sex, SmokingStatus
        """
        # Define feature groups
        self.num_features = ["Baseline_Age", "Baseline_Percent"]
        self.cat_features = ["Baseline_Sex", "Baseline_SmokingStatus"]

        # If scaler is not provided (i.e., this is the training set), fit it.
        if self.scaler is None:
            # Numerical Pipeline: Impute -> Scale
            num_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical Pipeline: Impute -> OneHot
            cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            self.scaler = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, self.num_features),
                    ("cat", cat_transformer, self.cat_features),
                ]
            )

            # Fit on the current dataframe
            self.scaler.fit(self.df)

        # Pre-transform the deep features for efficiency
        # We store them as a numpy array aligned with the dataframe
        self.deep_features_array = self.scaler.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial and Coronal)
        # dicom_dir is relative path in metadata, construct full path
        dicom_dir = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])

        axial_img, coronal_img = process_patient(
            patient_id, dicom_dir, Config.CACHE_DIR, load_cached=Config.LOAD_CACHED_DATA
        )

        # 2. Apply Augmentations (Spatial Only)
        if self.transform:
            # Apply same transform to both or independent?
            # Independent augmentation is generally robust for dual-stream networks
            # provided the transforms are spatial and realistic.
            aug_axial = self.transform(image=axial_img)["image"]
            aug_coronal = self.transform(image=coronal_img)["image"]
        else:
            # Convert to tensor directly
            to_tensor = ToTensorV2()
            aug_axial = to_tensor(image=axial_img)["image"]
            aug_coronal = to_tensor(image=coronal_img)["image"]

        # 3. Prepare Tabular Data

        # A. Deep Features (Normalized for MLP)
        deep_tab = torch.tensor(self.deep_features_array[idx], dtype=torch.float32)

        # B. Raw Features (For Skip Connection: Baseline FVC, Baseline Percent)
        # Note: We scale Percent by 0.01 to keep it in a similar range to FVC/1000 if needed,
        # but here we keep raw as requested by the idea, though normalizing FVC roughly is good practice.
        # The idea says "Raw Tabular Features (original scalars)".
        # We will provide them as is; the model head can learn the scaling parameter alpha.
        raw_tab = torch.tensor(
            [row["Baseline_FVC"], row["Baseline_Percent"]], dtype=torch.float32
        )

        # 4. Prepare Meta/Targets
        delta_week = torch.tensor(row["Delta_Week"], dtype=torch.float32)

        sample = {
            "axial": aug_axial,  # (3, 240, 240)
            "coronal": aug_coronal,  # (3, 240, 240)
            "deep_tab": deep_tab,  # (D,)
            "raw_tab": raw_tab,  # (2,) [FVC, Percent]
            "delta_week": delta_week,  # (1,)
            "patient_id": patient_id,
        }

        # Add target if available (Train/Val)
        if "FVC" in row and self.mode != "test":
            sample["target"] = torch.tensor(row["FVC"], dtype=torch.float32)

        # For submission file generation
        if "Patient_Week" in row:
            sample["patient_week_id"] = row["Patient_Week"]

        return sample


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.
    Strictly spatial augmentations, no intensity changes.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=0,  # Constant padding (black)
                ),
                # CoarseDropout can simulate artifacts or occlusion, helpful for robustness
                A.CoarseDropout(
                    max_holes=4,
                    max_height=20,
                    max_width=20,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Just Normalize and Convert
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders(batch_size=None, num_workers=None):
    """
    Factory function to create train and validation dataloaders.
    Handles the fitting of the scaler on training data and passing it to validation.
    """
    bs = batch_size if batch_size is not None else Config.BATCH_SIZE
    nw = num_workers if num_workers is not None else Config.NUM_WORKERS

    # 1. Create Training Dataset (Fits scaler)
    train_ds = LungDataset(mode="train", transform=get_transforms("train"))

    # 2. Create Validation Dataset (Uses train scaler)
    val_ds = LungDataset(
        mode="val", transform=get_transforms("val"), scaler=train_ds.scaler
    )

    # 3. Create Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds.scaler
