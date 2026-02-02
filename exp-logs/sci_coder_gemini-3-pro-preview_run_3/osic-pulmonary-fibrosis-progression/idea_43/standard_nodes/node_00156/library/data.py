import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


def get_ct_scans(dcm_dir):
    """
    Loads all DICOM files from a directory, sorts them by slice location.
    """
    files = []
    if not os.path.exists(dcm_dir):
        return []

    for f in os.listdir(dcm_dir):
        if f.endswith(".dcm"):
            files.append(os.path.join(dcm_dir, f))

    datasets = []
    for f in files:
        try:
            d = pydicom.dcmread(f)
            datasets.append(d)
        except:
            continue

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        datasets.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            datasets.sort(key=lambda x: int(x.InstanceNumber))
        except AttributeError:
            pass  # Keep original order if sorting fails

    return datasets


def get_windowed_img(ds, window_level, window_width):
    """
    Converts DICOM to HU and applies windowing. Returns normalized image [0, 1].
    """
    try:
        img = ds.pixel_array.astype(np.float32)
        intercept = getattr(ds, "RescaleIntercept", 0)
        slope = getattr(ds, "RescaleSlope", 1)
        img = slope * img + intercept
    except:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Apply windowing
    img_min = window_level - window_width // 2
    img_max = window_level + window_width // 2
    img = np.clip(img, img_min, img_max)

    # Normalize to [0, 1]
    img = (img - img_min) / (img_max - img_min)

    return img


def resize_img(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def process_patient_scans(patient_id, dcm_dir, cache_path):
    """
    Selects 3 slices (Anchor + 2 boundaries) and saves as .npy.
    """
    scans = get_ct_scans(dcm_dir)
    if not scans:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    areas = []
    images = []

    for s in scans:
        # Get raw HU for area calc
        try:
            raw = s.pixel_array.astype(np.float32) * getattr(
                s, "RescaleSlope", 1
            ) + getattr(s, "RescaleIntercept", 0)
        except:
            raw = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE))

        # Lung tissue is approx -1000 to -400 HU
        mask = (raw > -1000) & (raw < -400)
        area = mask.sum()
        areas.append(area)

        # Get processed image for storage
        img = get_windowed_img(s, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH)
        img = resize_img(img, Config.IMG_SIZE)
        images.append(img)

    areas = np.array(areas)

    if len(areas) == 0:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # 1. Anchor Slice: Max Lung Area
    idx_max = np.argmax(areas)
    max_area = areas[idx_max]

    # 2. Boundary Slices: Closest to 50% of max area above and below
    target_area = max_area * 0.5

    # Search below (superior)
    idx_top = 0
    min_diff = float("inf")
    for i in range(idx_max):
        diff = abs(areas[i] - target_area)
        if diff < min_diff:
            min_diff = diff
            idx_top = i

    # Search above (inferior)
    idx_bottom = len(areas) - 1
    min_diff = float("inf")
    for i in range(idx_max + 1, len(areas)):
        diff = abs(areas[i] - target_area)
        if diff < min_diff:
            min_diff = diff
            idx_bottom = i

    # If logic fails or too few slices, duplicate anchor or use available
    if len(images) < 3:
        selected_imgs = [images[idx_max]] * 3
    else:
        # Use indices found
        indices = [idx_top, idx_max, idx_bottom]
        selected_imgs = [images[i] for i in indices]

    # Stack: (3, H, W)
    final_tensor = np.stack(selected_imgs, axis=0).astype(np.float32)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        np.save(cache_path, final_tensor)
    except Exception as e:
        print(f"Failed to save cache for {patient_id}: {e}")

    return final_tensor


class LungDataset(Dataset):
    def __init__(
        self,
        df,
        dicom_root_dir,
        cache_dir,
        tabular_scalers=None,
        is_train=True,
        load_cached_data=True,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.dicom_root_dir = dicom_root_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.load_cached_data = load_cached_data

        # --- Tabular Preprocessing ---
        # Identify Baseline FVC for each patient (First measurement in history)
        sorted_df = self.df.sort_values(["Patient", "Weeks"])
        baseline_df = sorted_df.groupby("Patient").first().reset_index()
        self.baseline_lookup = baseline_df.set_index("Patient")[["FVC"]].to_dict()[
            "FVC"
        ]

        # Encoders & Scalers
        self.scalers = tabular_scalers if tabular_scalers else {}

        if "age_mean" not in self.scalers:
            self.scalers["age_mean"] = baseline_df["Age"].mean()
            self.scalers["age_std"] = baseline_df["Age"].std()
            self.scalers["fvc_mean"] = baseline_df["FVC"].mean()
            self.scalers["fvc_std"] = baseline_df["FVC"].std()

        # Pre-compute tabular vectors
        self.tabular_features = []
        self.targets = []
        self.patient_ids = []
        self.weeks = []
        self.image_paths = []

        for idx, row in self.df.iterrows():
            pid = row["Patient"]
            week = row["Weeks"]

            # Features
            base_fvc = self.baseline_lookup.get(pid, row["FVC"])

            # Normalize Age
            age = (row["Age"] - self.scalers["age_mean"]) / self.scalers["age_std"]

            # Normalize Baseline FVC
            base_fvc_norm = (base_fvc - self.scalers["fvc_mean"]) / self.scalers[
                "fvc_std"
            ]

            # Relative Time (Scaled)
            time_scaled = week * Config.TIME_SCALE_FACTOR

            # Sex (Male=0, Female=1)
            sex = 0 if row["Sex"] == "Male" else 1

            # SmokingStatus (Ordinal)
            ss = row["SmokingStatus"]
            if ss == "Never smoked":
                smoke = 0
            elif ss == "Ex-smoker":
                smoke = 1
            else:
                smoke = 2

            # Vector: [Baseline_FVC, Time, Age, Sex, Smoking]
            vec = np.array(
                [base_fvc_norm, time_scaled, age, sex, smoke], dtype=np.float32
            )
            self.tabular_features.append(vec)

            self.patient_ids.append(pid)
            self.weeks.append(week)
            self.image_paths.append(row["image_path"])

            if self.is_train:
                self.targets.append(row["FVC"])
            else:
                self.targets.append(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # --- Image Loading ---
        cache_path = os.path.join(self.cache_dir, f"{pid}.npy")

        img_tensor = None
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                img_tensor = np.load(cache_path)

                # Handle stale cache: Transpose (H, W, C) -> (C, H, W) if needed (Cite debug_lesson_11)
                if img_tensor.ndim == 3 and img_tensor.shape == (
                    Config.IMG_SIZE,
                    Config.IMG_SIZE,
                    Config.NUM_SLICES,
                ):
                    img_tensor = img_tensor.transpose(2, 0, 1)

                # Enforce float32 precision (Cite debug_lesson_2)
                img_tensor = img_tensor.astype(np.float32)

                # Validate final shape against current config
                if img_tensor.shape != (
                    Config.NUM_SLICES,
                    Config.IMG_SIZE,
                    Config.IMG_SIZE,
                ):
                    img_tensor = None  # Force regeneration
            except:
                img_tensor = None

        if img_tensor is None:
            # Process from scratch
            rel_path = self.image_paths[idx]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            img_tensor = process_patient_scans(pid, full_path, cache_path)

        # --- Return ---
        tab_vec = self.tabular_features[idx]
        target = self.targets[idx]

        return {
            "image": torch.tensor(img_tensor, dtype=torch.float32),
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": f"{pid}_{self.weeks[idx]}",
        }


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
    max_train_samples=Config.MAX_TRAIN_SAMPLES,
    max_val_samples=Config.MAX_VAL_SAMPLES,
):

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Limit samples for debugging
    if max_train_samples:
        train_df = train_df.head(max_train_samples)
    if max_val_samples:
        val_df = val_df.head(max_val_samples)

    # Calculate Scalers on Training Data
    # Age is static per patient
    unique_train_df = (
        train_df.sort_values(["Patient", "Weeks"])
        .groupby("Patient")
        .first()
        .reset_index()
    )

    # FVC is dynamic, so we use the full distribution (Cite solution_lesson_node_00155)
    scalers = {
        "age_mean": unique_train_df["Age"].mean(),
        "age_std": unique_train_df["Age"].std(),
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
    }

    # Train/Val Datasets
    train_ds = LungDataset(
        train_df, Config.INPUT_DIR, Config.CACHE_DIR, scalers, True, load_cached_data
    )
    val_ds = LungDataset(
        val_df, Config.INPUT_DIR, Config.CACHE_DIR, scalers, True, load_cached_data
    )

    # Test Dataset Expansion
    # Expand test set to match sample_submission (predict all weeks)
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge with test metadata
    # Rename FVC to Baseline_FVC for clarity, though we need 'FVC' column for Dataset logic
    # Rename Weeks to Baseline_Weeks to prevent collision with sub_df['Weeks']
    test_meta = test_df.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Weeks"}
    )
    test_expanded = sub_df.merge(test_meta, on="Patient", how="left")

    # Ensure 'FVC' column exists for dataset baseline lookup (it will be treated as baseline)
    test_expanded["FVC"] = test_expanded["Baseline_FVC"]

    test_ds = LungDataset(
        test_expanded,
        Config.INPUT_DIR,
        Config.CACHE_DIR,
        scalers,
        False,
        load_cached_data,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
