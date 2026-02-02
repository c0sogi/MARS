import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    CACHE_DIR,
    IMG_SIZE,
    NUM_SLICES,
    TARGET_MEAN,
    TARGET_STD,
    TIME_SCALE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import seed_everything


class TabularProcessor:
    """
    Handles preprocessing of tabular features:
    1. Identifies Baseline FVC and Week for each patient.
    2. Z-score standardizes continuous features (Age, Baseline FVC).
    3. One-hot encodes categorical features (Sex, SmokingStatus).
    4. Computes and scales relative time.
    """

    def __init__(self):
        self.age_mean = 0
        self.age_std = 1
        self.base_fvc_mean = 0
        self.base_fvc_std = 1
        self.fitted = False

    def fit(self, df):
        """Computes statistics from the training dataframe."""
        # Extract unique patient data for stats to avoid bias from frequent visitors
        unique_df = df.drop_duplicates(subset=["Patient"])

        # Calculate Baseline FVC stats
        # Note: We need to determine baseline FVC for the unique_df first
        patient_baselines = self._get_baselines(df)
        baselines = unique_df["Patient"].map(lambda x: patient_baselines[x]["FVC"])

        self.base_fvc_mean = baselines.mean()
        self.base_fvc_std = baselines.std()

        self.age_mean = unique_df["Age"].mean()
        self.age_std = unique_df["Age"].std()
        self.fitted = True

    def _get_baselines(self, df):
        """
        Determines the baseline FVC and Week for each patient.
        Assumes the baseline is the measurement with the minimum 'Weeks' value.
        """
        baselines = {}
        for patient, group in df.groupby("Patient"):
            # Find row with min weeks
            base_row = group.loc[group["Weeks"].idxmin()]
            baselines[patient] = {"FVC": base_row["FVC"], "Weeks": base_row["Weeks"]}
        return baselines

    def transform(self, df, mode="train"):
        """
        Transforms the dataframe into feature vectors.
        Returns:
            tabular_feats: np.array of shape (N, D)
            relative_time: np.array of shape (N, 1)
            targets: np.array of shape (N, 1) (if mode != 'test')
        """
        if not self.fitted:
            raise RuntimeError("TabularProcessor must be fitted before transform.")

        patient_baselines = self._get_baselines(df)

        features = []
        times = []
        targets = []

        # Mappings
        sex_map = {"Male": 0, "Female": 1}
        smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        for idx, row in df.iterrows():
            pid = row["Patient"]
            base_data = patient_baselines[pid]

            # 1. Continuous Features (Standardized)
            age_norm = (row["Age"] - self.age_mean) / self.age_std
            base_fvc_norm = (base_data["FVC"] - self.base_fvc_mean) / self.base_fvc_std

            # 2. Categorical Features (One-Hot approximation via list)
            # Sex: Male(1,0), Female(0,1)
            sex_vec = [0, 0]
            if row["Sex"] in sex_map:
                sex_vec[sex_map[row["Sex"]]] = 1

            # Smoking: Ex(1,0,0), Never(0,1,0), Current(0,0,1)
            smoke_vec = [0, 0, 0]
            if row["SmokingStatus"] in smoke_map:
                smoke_vec[smoke_map[row["SmokingStatus"]]] = 1

            # Combine Tabular: [BaseFVC, Age, Sex_0, Sex_1, Smoke_0, Smoke_1, Smoke_2]
            feat_vec = [base_fvc_norm, age_norm] + sex_vec + smoke_vec
            features.append(feat_vec)

            # 3. Relative Time (Scaled, not Z-scored)
            rel_week = row["Weeks"] - base_data["Weeks"]
            times.append(rel_week * TIME_SCALE)

            # 4. Target (Z-scored FVC)
            if mode != "test":
                target_norm = (row["FVC"] - TARGET_MEAN) / TARGET_STD
                targets.append(target_norm)

        features = np.array(features, dtype=np.float32)
        times = np.array(times, dtype=np.float32).reshape(-1, 1)

        if mode != "test":
            targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
            return features, times, targets

        return features, times, None


def load_scan(path):
    """Loads all DICOM files from a directory and sorts them."""
    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except:
                continue

    # Sort by ImagePositionPatient Z or InstanceNumber
    slices.sort(
        key=lambda x: (
            float(x.ImagePositionPatient[2])
            if hasattr(x, "ImagePositionPatient")
            else float(x.InstanceNumber)
        )
    )
    return slices


def get_pixels_hu(slices):
    """Converts DICOM slices to Hounsfield Units (HU)."""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def process_patient_images(patient_id, image_dir):
    """
    Reads DICOMs, selects 3 slices (Content-Adaptive), resizes, and normalizes.
    Returns: np.array of shape (IMG_SIZE, IMG_SIZE, 3)
    """
    full_path = os.path.join(INPUT_DIR, image_dir)
    if not os.path.exists(full_path):
        # Fallback for missing directories (should not happen based on metadata check)
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    slices = load_scan(full_path)
    if not slices:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    image = get_pixels_hu(slices)

    # --- Content-Adaptive Slice Selection ---
    # 1. Calculate Lung Area per slice (Threshold < -500 HU)
    # Lung is typically -900 to -500. Air is -1000.
    lung_areas = []
    for i in range(image.shape[0]):
        # Simple heuristic: count pixels < -500
        area = np.sum(image[i] < -500)
        lung_areas.append(area)

    lung_areas = np.array(lung_areas)

    if len(lung_areas) == 0 or np.max(lung_areas) == 0:
        # Fallback: take middle and boundaries
        indices = [0, len(slices) // 2, len(slices) - 1]
    else:
        # Find Anchor (Max Area)
        idx_max = np.argmax(lung_areas)
        max_area = lung_areas[idx_max]
        target_area = max_area * 0.5

        # Find Apical (Upper, index < idx_max)
        # We look for the slice with area closest to target_area
        candidates_up = np.arange(idx_max)
        if len(candidates_up) > 0:
            diffs = np.abs(lung_areas[candidates_up] - target_area)
            idx_apical = candidates_up[np.argmin(diffs)]
        else:
            idx_apical = 0

        # Find Basal (Lower, index > idx_max)
        candidates_down = np.arange(idx_max + 1, len(slices))
        if len(candidates_down) > 0:
            diffs = np.abs(lung_areas[candidates_down] - target_area)
            idx_basal = candidates_down[np.argmin(diffs)]
        else:
            idx_basal = len(slices) - 1

        indices = [idx_apical, idx_max, idx_basal]
        # Sort indices to maintain anatomical order (Top to Bottom)
        indices.sort()

    # --- Resize and Normalize ---
    selected_slices = []
    for idx in indices:
        img = image[idx]

        # Lung Windowing: Level -600, Width 1500
        # Range: [-1350, 150]
        L, W = -600, 1500
        lower, upper = L - W // 2, L + W // 2

        img = np.clip(img, lower, upper)

        # Normalize to [0, 1]
        img = (img - lower) / (upper - lower)

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        selected_slices.append(img)

    # Stack to (H, W, 3)
    final_img = np.stack(selected_slices, axis=-1)
    return final_img.astype(np.float32)


def cache_images(df, load_cached_data=True):
    """
    Iterates through patients in dataframe, processes images, and saves to cache.
    """
    unique_patients = df[["Patient", "image_path"]].drop_duplicates()

    print(f"Checking image cache for {len(unique_patients)} patients...")

    for _, row in tqdm(
        unique_patients.iterrows(), total=len(unique_patients), disable=True
    ):
        patient_id = row["Patient"]
        rel_path = row["image_path"]
        save_path = os.path.join(CACHE_DIR, f"{patient_id}.npy")

        if load_cached_data and os.path.exists(save_path):
            continue

        # Process and save
        try:
            img_data = process_patient_images(patient_id, rel_path)
            np.save(save_path, img_data)
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            # Save a zero array to prevent crash during training
            np.save(save_path, np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))


class LungDataset(Dataset):
    def __init__(self, df, tabular_features, times, targets=None, mode="train"):
        self.df = df
        self.tabular_features = tabular_features
        self.times = times
        self.targets = targets
        self.mode = mode
        self.patient_ids = df["Patient"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Load Image
        img_path = os.path.join(CACHE_DIR, f"{patient_id}.npy")
        if os.path.exists(img_path):
            image = np.load(img_path)
        else:
            # Fallback (should not happen if cache is run)
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        # Transpose to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))

        # Get Features
        tab = self.tabular_features[idx]
        t = self.times[idx]

        data = {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tab, dtype=torch.float32),
            "time": torch.tensor(t, dtype=torch.float32),
        }

        if self.mode != "test":
            target = self.targets[idx]
            data["target"] = torch.tensor(target, dtype=torch.float32)

        return data


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Main entry point to get DataLoaders.
    1. Loads metadata.
    2. Preprocesses tabular data.
    3. Caches images.
    4. Creates Datasets and Loaders.
    """
    seed_everything(SEED)

    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    if debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:20]
        # Keep test full usually, or subset if really needed

    # Combine for caching check (unique patients)
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    cache_images(all_df, load_cached_data=load_cached_data)

    # Tabular Processing
    processor = TabularProcessor()
    processor.fit(train_df)

    # Transform
    train_feats, train_times, train_targets = processor.transform(
        train_df, mode="train"
    )
    val_feats, val_times, val_targets = processor.transform(val_df, mode="val")
    test_feats, test_times, _ = processor.transform(test_df, mode="test")

    # Create Datasets
    train_dataset = LungDataset(
        train_df, train_feats, train_times, train_targets, mode="train"
    )
    val_dataset = LungDataset(val_df, val_feats, val_times, val_targets, mode="val")
    test_dataset = LungDataset(
        test_df, test_feats, test_times, targets=None, mode="test"
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
