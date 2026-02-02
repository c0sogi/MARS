import os
import cv2
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler

from library.config import Config

# ====================================================
# Helper Functions & Classes
# ====================================================


def get_img(path):
    """
    Loads a DICOM file, converts to Hounsfield Units (HU),
    applies lung windowing, and normalizes to [0, 255].
    """
    try:
        d = pydicom.dcmread(path)
        img = d.pixel_array.astype(np.float32)

        # Convert to HU
        intercept = getattr(d, "RescaleIntercept", 0)
        slope = getattr(d, "RescaleSlope", 1)
        img = img * slope + intercept

        # Lung Windowing
        # Level: -650, Window: 1500 => Min: -1400, Max: 100
        lung_win_min = -1400
        lung_win_max = 100
        img = np.clip(img, lung_win_min, lung_win_max)

        # Normalize to [0, 255]
        img = (img - lung_win_min) / (lung_win_max - lung_win_min)
        img = (img * 255).astype(np.uint8)

        return (
            img,
            d.SliceLocation if hasattr(d, "SliceLocation") else 0,
            d.InstanceNumber if hasattr(d, "InstanceNumber") else 0,
        )

    except Exception as e:
        # Return blank image if read fails
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8), 0, 0


class ContentAdaptiveSliceSelector:
    """
    Selects 3 slices (Apical, Middle, Basal) based on lung area and position.
    """

    def __init__(self, patient_dir):
        self.patient_dir = patient_dir

    def select_slices(self):
        files = glob.glob(os.path.join(self.patient_dir, "*.dcm"))
        if not files:
            return [np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)] * 3

        data = []
        for f in files:
            img, loc, inst = get_img(f)
            # Calculate lung area (simple heuristic: pixels < 150 in [0,255] scale roughly corresponds to air/lung)
            # In our windowing [-1400, 100], air is 0. Tissue is 255.
            # Lung parenchyma is roughly middle-dark.
            # Let's use a threshold.
            # Normalized 0 is -1400 HU. Normalized 255 is 100 HU.
            # -600 HU is approx 136.
            # So lung is roughly < 140.
            area = np.sum(img < 140)
            data.append({"img": img, "loc": loc, "inst": inst, "area": area})

        # Sort by instance number (or location) to establish Z-axis
        data.sort(key=lambda x: x["inst"])

        # Filter slices with valid lung area (e.g., > 5% of max area) to avoid neck/abdomen
        if not data:
            return [np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.uint8)] * 3

        max_area = max(d["area"] for d in data)
        valid_slices = [d for d in data if d["area"] > 0.05 * max_area]

        if len(valid_slices) < 3:
            # Fallback to original list if filtering removes too many
            valid_slices = data

        n = len(valid_slices)
        # Select Apical (top 30%), Middle (50%), Basal (80%)
        # Note: DICOM ordering can be top-down or bottom-up.
        # We assume sorted order represents physical traversal.
        indices = [int(n * 0.3), int(n * 0.5), int(n * 0.8)]

        # Clamp indices
        indices = [min(max(0, i), n - 1) for i in indices]

        selected_imgs = [valid_slices[i]["img"] for i in indices]

        # Resize
        resized_imgs = []
        for img in selected_imgs:
            img_r = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )
            resized_imgs.append(img_r)

        return resized_imgs


def process_patient_image(patient_id, dataset_type="train", load_cached_data=True):
    """
    Processes DICOMs for a patient, selecting 3 slices and creating a 3-channel image.
    Handles caching.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except:
            pass  # Fallback to re-processing if load fails

    # Determine path
    if dataset_type == "train" or dataset_type == "val":
        patient_dir = os.path.join(Config.TRAIN_DCM_DIR, patient_id)
    else:
        patient_dir = os.path.join(Config.TEST_DCM_DIR, patient_id)

    selector = ContentAdaptiveSliceSelector(patient_dir)
    slices = selector.select_slices()

    # Stack to (H, W, 3)
    img_3c = np.stack(slices, axis=-1)

    # Save to cache
    np.save(cache_path, img_3c)

    return img_3c


# ====================================================
# Dataset Class
# ====================================================


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        mode="train",
        transform=None,
        scaler_fvc=None,
        scaler_age=None,
        scaler_base_fvc=None,
    ):
        self.df = df.copy()
        self.mode = mode
        self.transform = transform

        # Scalers
        self.scaler_fvc = scaler_fvc
        self.scaler_age = scaler_age
        self.scaler_base_fvc = scaler_base_fvc

        # Precompute standardized features
        self._process_tabular_features()

    def _process_tabular_features(self):
        # 1. Age Standardization
        if self.scaler_age:
            self.df["Age_Scaled"] = self.scaler_age.transform(self.df[["Age"]])
        else:
            self.df["Age_Scaled"] = self.df["Age"]  # Should not happen in correct flow

        # 2. Baseline FVC Standardization
        if self.scaler_base_fvc:
            self.df["Base_FVC_Scaled"] = self.scaler_base_fvc.transform(
                self.df[["Baseline_FVC"]]
            )
        else:
            self.df["Base_FVC_Scaled"] = self.df["Baseline_FVC"]

        # 3. Relative Time Scaling (Weeks - Baseline_Week) * 0.01
        # Note: Baseline_Week is already computed in get_dataloaders and merged into df
        self.df["Relative_Time"] = (self.df["Weeks"] - self.df["Baseline_Week"]) * 0.01

        # 4. Target Standardization (Training only)
        if self.mode != "test" and self.scaler_fvc:
            self.df["FVC_Scaled"] = self.scaler_fvc.transform(self.df[["FVC"]])

        # 5. Categorical Encoding
        # Sex: Male=0, Female=1
        self.df["Sex_Code"] = self.df["Sex"].map({"Male": 0, "Female": 1}).fillna(0)
        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        self.df["Smoke_Code"] = (
            self.df["SmokingStatus"]
            .map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})
            .fillna(0)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # Determine dataset type for image loading path
        # Note: Val set comes from train folder
        d_type = "test" if self.mode == "test" else "train"
        image = process_patient_image(
            patient_id, dataset_type=d_type, load_cached_data=True
        )

        # 2. Augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default to tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        # Normalize image to [0, 1] if not handled by albumentations Normalize
        # (Albumentations Normalize usually handles subtraction of mean and div by std)
        # Here we assume the transform pipeline includes Normalize.
        # If not, the input is uint8 [0, 255] or float.

        # 3. Tabular Vector
        # Features: [Base_FVC_Scaled, Age_Scaled, Sex_Code, Smoke_Code, Relative_Time]
        tab_vec = np.array(
            [
                row["Base_FVC_Scaled"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoke_Code"],
                row["Relative_Time"],
            ],
            dtype=np.float32,
        )

        # 4. Target
        if self.mode != "test":
            target = np.array([row["FVC_Scaled"]], dtype=np.float32)
            # We return raw FVC as well for metric calculation
            raw_target = np.array([row["FVC"]], dtype=np.float32)
        else:
            target = np.array([0.0], dtype=np.float32)
            raw_target = np.array([0.0], dtype=np.float32)

        return {
            "image": image,
            "tabular": torch.from_numpy(tab_vec),
            "target": torch.from_numpy(target),
            "raw_target": torch.from_numpy(raw_target),
            "patient_week": f"{patient_id}_{row['Weeks']}",
        }


# ====================================================
# Data Loading & Preparation
# ====================================================


def get_dataloaders(debug=False):
    """
    Prepares DataLoaders for Train, Val, and Test sets.
    Handles feature engineering and scaler fitting.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    if debug:
        # Sample patients for debugging
        patients = train_df["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        train_df = train_df[train_df["Patient"].isin(patients)].reset_index(drop=True)
        val_df = val_df.iloc[:20].reset_index(drop=True)

    # 2. Feature Engineering: Identify Baseline for Train/Val
    # For Train/Val, we have full history. Baseline is the row with min Weeks.

    def add_baseline_info(df):
        # Group by patient and find the row with minimum Weeks
        baseline_df = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
        )

        # Merge back
        df = df.merge(baseline_df, on="Patient", how="left")
        return df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # 3. Prepare Test DataFrame
    # Test logic: merge sample_submission with test_meta
    # sample_sub has Patient_Week -> split to Patient, Weeks
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Weeks"] = (
        sample_sub["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
    )

    # test_meta_df contains the baseline info (one row per patient)
    # We rename FVC -> Baseline_FVC, Weeks -> Baseline_Week
    test_base = test_meta_df[
        ["Patient", "FVC", "Weeks", "Age", "Sex", "SmokingStatus"]
    ].copy()
    test_base = test_base.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge
    test_df = sample_sub.merge(test_base, on="Patient", how="left")

    # 4. Fit Scalers (on Train only)
    scaler_fvc = StandardScaler()
    scaler_age = StandardScaler()
    scaler_base_fvc = StandardScaler()

    scaler_fvc.fit(train_df[["FVC"]])
    scaler_age.fit(train_df[["Age"]])
    scaler_base_fvc.fit(train_df[["Baseline_FVC"]])

    # 5. Define Transforms
    # ImageNet stats for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    # 6. Create Datasets
    train_dataset = OSICDataset(
        train_df,
        mode="train",
        transform=train_transform,
        scaler_fvc=scaler_fvc,
        scaler_age=scaler_age,
        scaler_base_fvc=scaler_base_fvc,
    )

    val_dataset = OSICDataset(
        val_df,
        mode="val",
        transform=val_transform,
        scaler_fvc=scaler_fvc,
        scaler_age=scaler_age,
        scaler_base_fvc=scaler_base_fvc,
    )

    test_dataset = OSICDataset(
        test_df,
        mode="test",
        transform=val_transform,
        scaler_fvc=scaler_fvc,
        scaler_age=scaler_age,
        scaler_base_fvc=scaler_base_fvc,
    )

    # 7. Create DataLoaders
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

    # Return scalers as well for inverse transform
    scalers = {"fvc_mean": scaler_fvc.mean_[0], "fvc_std": scaler_fvc.scale_[0]}

    return train_loader, val_loader, test_loader, scalers
