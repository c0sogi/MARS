import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung FVC Prediction.
    Loads cached 3D image tensors and processed tabular data.
    """

    def __init__(
        self,
        df,
        image_cache_dir,
        mode="train",
        transform=None,
        scalers=None,
        feature_cols=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_cache_dir = image_cache_dir
        self.mode = mode
        self.transform = transform
        self.scalers = scalers
        self.feature_cols = feature_cols

        # Pre-compute feature vectors
        if self.scalers and self.feature_cols:
            self.features = self._prepare_features()

    def _prepare_features(self):
        """
        Extracts and normalizes tabular features.
        """
        # Numerical Features: Baseline_FVC, Baseline_Percent, Age
        # Note: Relative_Time is handled separately as it doesn't use StandardScaler (scaled by 0.01)
        num_cols = ["Baseline_FVC", "Baseline_Percent", "Age"]
        X_num = self.df[num_cols].values.astype(np.float32)

        if self.scalers and "num" in self.scalers:
            X_num = self.scalers["num"].transform(X_num)

        # Categorical Features: Sex_Code, Smoking_Code
        X_cat = self.df[["Sex_Code", "Smoking_Code"]].values.astype(np.float32)

        # Time Feature: Relative_Time (already scaled by 0.01 in dataframe prep)
        X_time = self.df[["Relative_Time"]].values.astype(np.float32)

        # Concatenate: [Num(3), Cat(2), Time(1)] -> Dim 6
        return np.hstack([X_num, X_cat, X_time])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image from Cache
        cache_path = os.path.join(self.image_cache_dir, f"{patient_id}.npy")
        if os.path.exists(cache_path):
            try:
                # Load (H, W, C)
                imgs = np.load(cache_path)
            except Exception:
                # Fallback
                imgs = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_SLICES),
                    dtype=np.float32,
                )
        else:
            imgs = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_SLICES), dtype=np.float32
            )

        # Convert to Channel First: (C, H, W)
        imgs = imgs.transpose(2, 0, 1)
        image_tensor = torch.tensor(imgs, dtype=torch.float32)

        # 2. Load Tabular Features
        if hasattr(self, "features"):
            clinical_features = self.features[idx]
        else:
            clinical_features = np.zeros(Config.CLINICAL_INPUT_DIM, dtype=np.float32)

        clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32)

        # 3. Load Target
        if self.mode != "submission":
            raw_target = row["FVC"]
            # Normalize target
            if self.scalers and "target" in self.scalers:
                # reshape for scaler: (1, 1)
                target_val = self.scalers["target"].transform(np.array([[raw_target]]))[
                    0
                ][0]
            else:
                target_val = raw_target
            return (image_tensor, clinical_tensor), torch.tensor(
                target_val, dtype=torch.float32
            )
        else:
            # For submission, return dummy target
            return (image_tensor, clinical_tensor), torch.tensor(
                0.0, dtype=torch.float32
            )


def load_scan(path):
    """Loads all DICOM files from a directory and sorts them."""
    slices = [pydicom.dcmread(p) for p in glob.glob(os.path.join(path, "*.dcm"))]
    # Sort by ImagePositionPatient Z or InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass
    return slices


def get_pixels_hu(slices):
    """Converts DICOM slices to Hounsfield Units."""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    for n, s in enumerate(slices):
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1
        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)
        image[n] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def process_patient(patient_id, image_dir, cache_dir):
    """
    Reads DICOMs, selects slices, resizes, and saves to cache.
    """
    save_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # If already exists, skip
    if os.path.exists(save_path):
        return

    patient_dir = os.path.join(image_dir, patient_id)
    if not os.path.exists(patient_dir):
        # Create dummy if directory missing
        dummy = np.zeros(
            (Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_SLICES), dtype=np.float32
        )
        np.save(save_path, dummy)
        return

    try:
        slices = load_scan(patient_dir)
        if not slices:
            raise ValueError("No slices found")

        image = get_pixels_hu(slices)  # (D, H, W)

        # Resize slices to Config.IMG_SIZE
        resized_images = []
        for i in range(image.shape[0]):
            img = cv2.resize(
                image[i],
                (Config.IMG_SIZE, Config.IMG_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            resized_images.append(img)
        image = np.stack(resized_images)  # (D, 260, 260)

        # Content-Adaptive Slice Selection
        # Calculate Lung Area (HU between -1000 and -400)
        lung_mask = (image > -1000) & (image < -400)
        lung_area = lung_mask.sum(axis=(1, 2))

        if lung_area.max() == 0:
            idx_anchor = len(slices) // 2
        else:
            idx_anchor = np.argmax(lung_area)

        max_area = lung_area[idx_anchor]
        threshold = max_area * Config.SLICE_THRESHOLD

        # Find boundaries
        idx_upper = idx_anchor
        for i in range(idx_anchor, -1, -1):
            if lung_area[i] < threshold:
                idx_upper = i
                break

        idx_lower = idx_anchor
        for i in range(idx_anchor, len(slices)):
            if lung_area[i] < threshold:
                idx_lower = i
                break

        # Select 3 slices: [Upper Boundary, Anchor, Lower Boundary]
        # We sort them to maintain spatial order
        selected_indices = sorted(list(set([idx_upper, idx_anchor, idx_lower])))

        if len(selected_indices) == 1:
            final_slices = [image[selected_indices[0]]] * 3
        elif len(selected_indices) == 2:
            final_slices = [
                image[selected_indices[0]],
                image[selected_indices[1]],
                image[selected_indices[1]],
            ]
        else:
            final_slices = [image[idx_upper], image[idx_anchor], image[idx_lower]]

        final_volume = np.stack(final_slices, axis=-1)  # (260, 260, 3)

        # Normalization
        # Clip to [-1000, 400]
        final_volume = np.clip(final_volume, -1000, 400)
        # Scale to [0, 1]
        final_volume = (final_volume + 1000) / 1400.0

        np.save(save_path, final_volume.astype(np.float32))

    except Exception as e:
        # Save dummy on error
        dummy = np.zeros(
            (Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_SLICES), dtype=np.float32
        )
        np.save(save_path, dummy)


def preprocess_data(load_cached_data=True):
    """
    Iterates over all patients and creates cached processed images.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load all patient IDs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    all_patients = pd.concat(
        [train_df["Patient"], val_df["Patient"], test_df["Patient"]]
    ).unique()

    for patient in all_patients:
        # Check if cache exists
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient}.npy")

        # If caching is enabled and file exists, skip
        if load_cached_data and os.path.exists(cache_path):
            continue

        # Determine source directory
        if os.path.exists(os.path.join(Config.INPUT_DIR, "train", patient)):
            img_dir = os.path.join(Config.INPUT_DIR, "train")
        elif os.path.exists(os.path.join(Config.INPUT_DIR, "test", patient)):
            img_dir = os.path.join(Config.INPUT_DIR, "test")
        else:
            continue

        process_patient(patient, img_dir, Config.CACHE_DIR)


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Submission.
    """
    # 1. Ensure Images are Processed
    preprocess_data(load_cached_data)

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # 3. Feature Engineering Helper
    def prepare_tabular(df, is_submission=False):
        df_out = df.copy()

        if is_submission:
            # Parse Patient and Week from Patient_Week
            df_out["Patient"] = df_out["Patient_Week"].apply(lambda x: x.split("_")[0])
            df_out["Weeks"] = df_out["Patient_Week"].apply(
                lambda x: int(x.split("_")[1])
            )

            # Merge Baseline Metadata from test_meta_df
            base_df = test_meta_df.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Week",
                }
            )
            df_out = df_out.merge(base_df, on="Patient", how="left")

        else:
            # Calculate Baseline per Patient (Visit closest to Week 0)
            def get_baseline(group):
                group["abs_weeks"] = group["Weeks"].abs()
                base = group.sort_values("abs_weeks").iloc[0]
                return pd.Series(
                    {
                        "Baseline_FVC": base["FVC"],
                        "Baseline_Percent": base["Percent"],
                        "Baseline_Week": base["Weeks"],
                    }
                )

            baselines = df_out.groupby("Patient").apply(get_baseline).reset_index()
            df_out = df_out.merge(baselines, on="Patient", how="left")

        # Calculate Relative Time (Scaled by 0.01)
        df_out["Relative_Time"] = (df_out["Weeks"] - df_out["Baseline_Week"]) * 0.01

        return df_out

    train_processed = prepare_tabular(train_df)
    val_processed = prepare_tabular(val_df)
    submission_processed = prepare_tabular(sample_sub, is_submission=True)

    # 4. Encoding & Scaling
    # Encode Categoricals (Sex, Smoking)
    le_sex = LabelEncoder()
    le_smoke = LabelEncoder()

    # Fit on all available data
    all_sex = pd.concat(
        [train_processed["Sex"], val_processed["Sex"], submission_processed["Sex"]]
    )
    all_smoke = pd.concat(
        [
            train_processed["SmokingStatus"],
            val_processed["SmokingStatus"],
            submission_processed["SmokingStatus"],
        ]
    )

    le_sex.fit(all_sex.astype(str))
    le_smoke.fit(all_smoke.astype(str))

    for d in [train_processed, val_processed, submission_processed]:
        d["Sex_Code"] = le_sex.transform(d["Sex"].astype(str))
        d["Smoking_Code"] = le_smoke.transform(d["SmokingStatus"].astype(str))

    # Fit Scalers on TRAIN only
    scaler_num = StandardScaler()
    scaler_target = StandardScaler()

    num_cols = ["Baseline_FVC", "Baseline_Percent", "Age"]
    scaler_num.fit(train_processed[num_cols])
    scaler_target.fit(train_processed[["FVC"]])

    scalers = {"num": scaler_num, "target": scaler_target}

    # 5. Create Datasets
    # Feature columns: Baseline_FVC, Baseline_Percent, Age, Sex_Code, Smoking_Code, Relative_Time
    # Note: Order must match Config.CLINICAL_INPUT_DIM logic

    train_dataset = LungDataset(
        train_processed,
        Config.CACHE_DIR,
        mode="train",
        scalers=scalers,
        feature_cols=True,
    )

    val_dataset = LungDataset(
        val_processed, Config.CACHE_DIR, mode="val", scalers=scalers, feature_cols=True
    )

    test_dataset = LungDataset(
        submission_processed,
        Config.CACHE_DIR,
        mode="submission",
        scalers=scalers,
        feature_cols=True,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scalers
