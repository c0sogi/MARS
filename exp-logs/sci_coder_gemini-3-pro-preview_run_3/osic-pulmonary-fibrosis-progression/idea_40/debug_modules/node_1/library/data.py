import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class LungDataProcessor:
    """
    Handles data preprocessing, scaling, and feature engineering for CIDS-Net.
    """

    def __init__(self):
        self.age_scaler = StandardScaler()
        self.base_fvc_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.sex_encoder = LabelEncoder()
        self.smoke_encoder = LabelEncoder()

        self.fitted = False
        self.target_mean = None
        self.target_std = None

    def fit(self, train_df):
        """
        Fits scalers on the training data.
        """
        df = train_df.copy()

        # Identify baseline FVC for each patient (visit closest to Week 0)
        df["abs_weeks"] = df["Weeks"].abs()
        baseline_df = (
            df.sort_values(["Patient", "abs_weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC"]].rename(
            columns={"FVC": "Baseline_FVC"}
        )

        # Merge baseline FVC back to fit the scaler on the correct distribution
        df = df.merge(baseline_df, on="Patient", how="left")

        # Fit encoders
        # Explicitly fit on all known domain categories to handle rare/unseen labels
        self.sex_encoder.fit(["Male", "Female"])
        self.smoke_encoder.fit(["Ex-smoker", "Never smoked", "Currently smokes"])

        # Fit scalers
        self.age_scaler.fit(df[["Age"]])
        self.base_fvc_scaler.fit(df[["Baseline_FVC"]])
        self.target_scaler.fit(df[["FVC"]])

        self.target_mean = self.target_scaler.mean_[0]
        self.target_std = self.target_scaler.scale_[0]

        self.fitted = True
        return df

    def transform(self, df, is_test=False):
        """
        Transforms the dataframe into features.
        """
        if not self.fitted:
            raise RuntimeError("Processor must be fitted before transform.")

        df = df.copy()

        # 1. Handle Baseline FVC derivation
        if is_test:
            # For test set, the provided clinical data IS the baseline.
            # If Baseline_FVC is not explicitly present, we assume FVC column holds it.
            if "Baseline_FVC" not in df.columns:
                df["Baseline_FVC"] = df["FVC"]
        else:
            # For train/val, we derive Baseline_FVC from the history if not present
            if "Baseline_FVC" not in df.columns:
                df["abs_weeks"] = df["Weeks"].abs()
                baseline_df = (
                    df.sort_values(["Patient", "abs_weeks"])
                    .groupby("Patient")
                    .first()
                    .reset_index()
                )
                baseline_map = baseline_df.set_index("Patient")["FVC"].to_dict()
                df["Baseline_FVC"] = df["Patient"].map(baseline_map)

        # 2. Encode Categoricals
        df["Sex_Code"] = self.sex_encoder.transform(df["Sex"])
        df["Smoke_Code"] = self.smoke_encoder.transform(df["SmokingStatus"])

        # 3. Scale Numerical Features
        df["Age_Scaled"] = self.age_scaler.transform(df[["Age"]])
        df["Base_FVC_Scaled"] = self.base_fvc_scaler.transform(df[["Baseline_FVC"]])

        # 4. Relative Time (Scaled)
        df["Time_Scaled"] = df["Weeks"] * Config.TIME_SCALE

        # 5. Target Scaling
        if "FVC" in df.columns and not is_test:
            df["FVC_Scaled"] = self.target_scaler.transform(df[["FVC"]])
        else:
            df["FVC_Scaled"] = 0.0  # Dummy for test

        return df


class LungDataset(Dataset):
    def __init__(self, df, processor, cache_dir=Config.CACHE_DIR, mode="train"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cache_dir = cache_dir
        self.mode = mode

        # Pre-compute tabular features: [BaseFVC, RelTime, Age, Sex, Smoke]
        self.features = np.stack(
            [
                self.df["Base_FVC_Scaled"].values,
                self.df["Time_Scaled"].values,
                self.df["Age_Scaled"].values,
                self.df["Sex_Code"].values,
                self.df["Smoke_Code"].values,
            ],
            axis=1,
        ).astype(np.float32)

        self.targets = self.df["FVC_Scaled"].values.astype(np.float32)
        self.patient_ids = self.df["Patient"].values
        self.image_paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        rel_path = self.image_paths[idx]

        # Load Image (with caching)
        image = self.load_processed_image(patient_id, rel_path, load_cached_data=True)

        features = self.features[idx]
        target = self.targets[idx]

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    def load_processed_image(self, patient_id, rel_path, load_cached_data=True):
        """
        Loads processed image from cache or processes from scratch.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img_tensor = self.process_dicom(full_path)

        # 3. Save to cache
        try:
            np.save(cache_path, img_tensor)
        except Exception:
            pass  # Ignore save errors

        return img_tensor

    def process_dicom(self, dir_path):
        """
        Reads DICOMs, applies Lung Window, selects 3 slices (Anchor + Boundaries), and resizes.
        """
        files = glob.glob(os.path.join(dir_path, "*.dcm"))
        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read and sort by Z-position
        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                # Try ImagePositionPatient Z, else InstanceNumber
                try:
                    z = float(dcm.ImagePositionPatient[2])
                except:
                    z = float(dcm.InstanceNumber)
                slices.append((z, dcm))
            except:
                continue

        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        slices.sort(key=lambda x: x[0])

        processed_candidates = []
        areas = []

        # Pre-process all slices to find lung area
        for _, dcm in slices:
            try:
                img = dcm.pixel_array.astype(np.float32)

                # Convert to HU
                intercept = getattr(dcm, "RescaleIntercept", 0)
                slope = getattr(dcm, "RescaleSlope", 1)
                img = img * slope + intercept

                # Calculate Lung Area (approx pixels between -1000 and -400 HU)
                area = np.sum((img > -1000) & (img < -400))
                areas.append(area)

                # Apply Lung Window
                level = Config.WINDOW_LEVEL
                width = Config.WINDOW_WIDTH
                lower = level - width / 2
                upper = level + width / 2

                img = np.clip(img, lower, upper)
                img = (img - lower) / (upper - lower)  # Normalize [0, 1]

                # Resize
                img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
                processed_candidates.append(img)
            except:
                continue

        if not processed_candidates:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Slice Selection Logic
        num_slices = len(processed_candidates)
        areas = np.array(areas)

        if num_slices < 3:
            # Duplicate if insufficient slices
            indices = [0, 0, 0]
            if num_slices == 2:
                indices = [0, 0, 1]
        else:
            max_area = np.max(areas)
            if max_area == 0:
                indices = [0, num_slices // 2, num_slices - 1]
            else:
                # ROI: Slices with > 50% of max lung area
                roi_indices = np.where(areas > 0.5 * max_area)[0]
                if len(roi_indices) == 0:
                    indices = [0, num_slices // 2, num_slices - 1]
                else:
                    start = roi_indices[0]
                    end = roi_indices[-1]
                    anchor = roi_indices[len(roi_indices) // 2]
                    indices = [start, anchor, end]

        final_slices = [processed_candidates[i] for i in indices]
        img_tensor = np.stack(final_slices, axis=0)  # (3, H, W)

        return img_tensor.astype(np.float32)


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    test_csv_path=Config.TEST_CSV,
    sample_sub_path=Config.SAMPLE_SUBMISSION,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):

    # 1. Load DataFrames
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_meta_df = pd.read_csv(test_csv_path)
    sample_sub_df = pd.read_csv(sample_sub_path)

    # Debugging: Limit data size
    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.head(Config.MAX_TRAIN_SAMPLES)
        val_df = val_df.head(Config.MAX_TRAIN_SAMPLES)

    # 2. Fit Processor
    processor = LungDataProcessor()
    train_df_processed = processor.fit(train_df)

    # 3. Process Validation
    val_df_processed = processor.transform(val_df)

    # 4. Prepare Test Data (Merge Sample Submission with Baseline Metadata)
    # Parse Patient and Weeks from sample_submission
    sub_data = []
    for pw in sample_sub_df["Patient_Week"]:
        parts = pw.split("_")
        week = int(parts[-1])
        patient = "_".join(parts[:-1])
        sub_data.append({"Patient_Week": pw, "Patient": patient, "Weeks": week})

    sub_expanded = pd.DataFrame(sub_data)

    # Merge with clinical metadata (Age, Sex, Baseline FVC, image_path)
    # test_meta_df contains the baseline info. We drop 'Weeks' from it to avoid conflict with target 'Weeks'.
    test_meta_clean = test_meta_df.drop(columns=["Weeks"])
    test_final = sub_expanded.merge(test_meta_clean, on="Patient", how="left")

    # Transform Test Data
    test_df_processed = processor.transform(test_final, is_test=True)

    # 5. Create Datasets
    train_dataset = LungDataset(train_df_processed, processor, mode="train")
    val_dataset = LungDataset(val_df_processed, processor, mode="val")
    test_dataset = LungDataset(test_df_processed, processor, mode="test")

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, processor
