import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        mode="train",
        transform=None,
        cache_dir=None,
        load_cached=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.load_cached = load_cached

        # Pre-compute patient paths
        self.patient_paths = {
            p: os.path.join(img_dir, p) for p in self.df["Patient"].unique()
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image Loading & Processing
        image = self._get_image(patient_id)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # 2. Tabular Features
        # Features: [Baseline FVC, Baseline Percent, t_rel, Age, Sex, Smoking]
        tabular = row[
            [
                "base_FVC_scaled",
                "base_Percent_scaled",
                "t_rel",
                "Age_scaled",
                "Sex_encoded",
                "Smoking_encoded",
            ]
        ].values.astype(np.float32)
        tabular = torch.from_numpy(tabular)

        # 3. Target & Metadata
        patient_week = (
            row["Patient_Week"]
            if "Patient_Week" in row
            else f"{patient_id}_{row['Weeks']}"
        )

        if self.mode != "test":
            # Return scaled target for loss calculation
            target = torch.tensor([row["FVC_scaled"]], dtype=torch.float32)
            return {
                "image": image,
                "tabular": tabular,
                "target": target,
                "patient_week": patient_week,
                "FVC_raw": row["FVC"],  # For metric calculation
            }
        else:
            return {"image": image, "tabular": tabular, "patient_week": patient_week}

    def _get_image(self, patient_id):
        # Check Cache
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        if self.load_cached and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except:
                pass

        # Process from scratch
        patient_dir = self.patient_paths.get(patient_id)
        if not patient_dir or not os.path.exists(patient_dir):
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # Read DICOMs
        files = [f for f in os.listdir(patient_dir) if f.endswith(".dcm")]
        if not files:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        dcm_files = []
        for f in files:
            try:
                dcm = pydicom.dcmread(os.path.join(patient_dir, f))
                dcm_files.append(dcm)
            except:
                continue

        # Sort by InstanceNumber (Z-position)
        dcm_files.sort(
            key=lambda x: float(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
        )

        if not dcm_files:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        processed_slices = []
        slice_areas = []

        for dcm in dcm_files:
            img = self._process_dicom_slice(dcm)
            processed_slices.append(img)

            # Calculate lung area (simple thresholding on normalized image)
            # Lung window is approx 0.2 to 0.7 in [0,1] space
            mask = (img > 0.1) & (img < 0.6)
            slice_areas.append(np.sum(mask))

        # Content-Adaptive Selection
        max_area_idx = np.argmax(slice_areas)
        max_area = slice_areas[max_area_idx]

        # Select boundaries with > 50% max area
        threshold = 0.5 * max_area
        valid_indices = [i for i, area in enumerate(slice_areas) if area > threshold]

        if not valid_indices:
            valid_indices = [max_area_idx]

        # Select Top, Anchor, Bottom
        idx1 = valid_indices[0]
        idx2 = max_area_idx
        idx3 = valid_indices[-1]

        selection_indices = sorted([idx1, idx2, idx3])
        final_slices = [processed_slices[i] for i in selection_indices]

        # Stack and convert to uint8
        img_3ch = np.stack(final_slices, axis=-1)
        img_3ch = (img_3ch * 255).astype(np.uint8)

        # Save to cache
        if self.cache_dir:
            np.save(cache_path, img_3ch)

        return img_3ch

    def _process_dicom_slice(self, dcm):
        try:
            pixel_array = dcm.pixel_array.astype(np.float32)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            slope = getattr(dcm, "RescaleSlope", 1)
            img = pixel_array * slope + intercept

            # Lung Windowing (W:1500, L:-600) -> [-1350, 150]
            img = (img + 1350) / 1500
            img = np.clip(img, 0, 1)

            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
            return img
        except:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
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


def preprocess_tabular(df, train_stats=None):
    # Encode Categoricals
    # Sex: Male=0, Female=1
    # Smoking: Never=0, Ex=1, Current=2 (Scaled to 0, 0.5, 1.0)

    if "Sex" in df.columns:
        df["Sex_encoded"] = df["Sex"].map({"Male": 0, "Female": 1}).astype(float)

    if "SmokingStatus" in df.columns:
        df["Smoking_encoded"] = (
            df["SmokingStatus"]
            .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
            .astype(float)
        )
        df["Smoking_encoded"] = df["Smoking_encoded"] / 2.0

    if train_stats is None:
        train_stats = {
            "FVC_mean": df["FVC"].mean(),
            "FVC_std": df["FVC"].std(),
            "Percent_mean": df["Percent"].mean(),
            "Percent_std": df["Percent"].std(),
            "Age_mean": df["Age"].mean(),
            "Age_std": df["Age"].std(),
        }

    # Standardize Features using Training Stats
    df["base_FVC_scaled"] = (df["base_FVC"] - train_stats["FVC_mean"]) / train_stats[
        "FVC_std"
    ]
    df["base_Percent_scaled"] = (
        df["base_Percent"] - train_stats["Percent_mean"]
    ) / train_stats["Percent_std"]
    df["Age_scaled"] = (df["Age"] - train_stats["Age_mean"]) / train_stats["Age_std"]

    if "FVC" in df.columns:
        df["FVC_scaled"] = (df["FVC"] - train_stats["FVC_mean"]) / train_stats[
            "FVC_std"
        ]

    return df, train_stats


def add_baseline_features(df, baseline_source_df=None):
    if baseline_source_df is None:
        # For Train/Val: Find baseline (min absolute weeks)
        df["abs_weeks"] = df["Weeks"].abs()
        df_sorted = df.sort_values(["Patient", "abs_weeks"])
        baseline_df = df_sorted.groupby("Patient").first().reset_index()

        baseline_df = baseline_df[["Patient", "FVC", "Percent"]]
        baseline_df.columns = ["Patient", "base_FVC", "base_Percent"]

        df = pd.merge(df, baseline_df, on="Patient", how="left")
        df["t_rel"] = df["Weeks"] * Config.TIME_SCALE
        df = df.drop(columns=["abs_weeks"])
    else:
        # For Test: Merge baseline from source
        baseline_df = baseline_source_df[
            ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        ]
        baseline_df.columns = [
            "Patient",
            "base_FVC",
            "base_Percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ]

        # Parse Patient and Weeks from Patient_Week
        df["Patient"] = df["Patient_Week"].apply(lambda x: x.split("_")[0])
        df["Weeks"] = df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

        df = pd.merge(df, baseline_df, on="Patient", how="left")
        df["t_rel"] = df["Weeks"] * Config.TIME_SCALE

    return df


def get_dataloaders(debug=False):
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    if debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:20]
        sub_df = sub_df.iloc[:20]

    # Feature Engineering
    train_df = add_baseline_features(train_df)
    val_df = add_baseline_features(val_df)

    # Compute stats on train, apply to all
    train_df, stats = preprocess_tabular(train_df, train_stats=None)
    val_df, _ = preprocess_tabular(val_df, train_stats=stats)

    # Prepare Test
    test_expanded_df = add_baseline_features(sub_df.copy(), baseline_source_df=test_df)
    test_expanded_df, _ = preprocess_tabular(test_expanded_df, train_stats=stats)

    # Datasets
    train_ds = OSICDataset(
        train_df,
        Config.TRAIN_IMG_DIR,
        mode="train",
        transform=get_transforms("train"),
        cache_dir=Config.CACHE_DIR,
    )
    val_ds = OSICDataset(
        val_df,
        Config.TRAIN_IMG_DIR,
        mode="val",
        transform=get_transforms("val"),
        cache_dir=Config.CACHE_DIR,
    )
    test_ds = OSICDataset(
        test_expanded_df,
        Config.TEST_IMG_DIR,
        mode="test",
        transform=get_transforms("test"),
        cache_dir=Config.CACHE_DIR,
    )

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, stats
