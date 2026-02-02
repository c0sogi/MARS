import os
import glob
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Attempt to import pydicom for DICOM handling
try:
    import pydicom
except ImportError:
    pydicom = None


def load_image(path, size=(224, 224)):
    """
    Loads a DICOM image, applies lung windowing, and resizes.
    Falls back to cv2 or returns a blank image if pydicom is missing.
    """
    if pydicom is None:
        # Fallback for environments without pydicom
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(size, dtype=np.float32)
        img = cv2.resize(img, size)
        # Simple min-max normalization since we lack HU information
        denom = np.max(img) - np.min(img)
        if denom == 0:
            return np.zeros(size, dtype=np.float32)
        img = (img - np.min(img)) / denom
        return img.astype(np.float32)

    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)

        # Rescale to Hounsfield Units (HU)
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            slope = float(dcm.RescaleSlope)
            intercept = float(dcm.RescaleIntercept)
            img = img * slope + intercept

        # Apply Lung Window: Width 1500, Level -600
        # Range: [-1350, 150]
        window_center = -600
        window_width = 1500
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        # Normalize to [0, 1]
        img = (img - img_min) / (img_max - img_min)

        # Resize
        img = cv2.resize(img, size)
        return img

    except Exception as e:
        # Return blank image on corruption
        return np.zeros(size, dtype=np.float32)


def get_lung_area(img):
    """
    Approximates lung area by counting pixels in a specific intensity range.
    Assumes image is normalized to [0, 1] based on lung window [-1350, 150].
    Lung air (-1000 HU to -400 HU) maps roughly to [0.2, 0.7].
    """
    mask = (img > 0.15) & (img < 0.75)
    return np.sum(mask)


def process_patient_images(patient_id, image_dir, load_cached_data=True):
    """
    Selects 3 adaptive slices (Apical, Middle, Basal) using the Max Area heuristic.
    Caches the processed tensor as a .npy file.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Load from cache if available
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Corrupted cache, recompute

    # 2. Process from scratch
    files = glob.glob(os.path.join(image_dir, "*.dcm"))

    # Handle missing files
    if not files:
        dummy = np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )
        if load_cached_data:
            np.save(cache_path, dummy)
        return dummy

    # Sort files by Z-position (ImagePositionPatient[2]) or filename
    file_tuples = []
    for f in files:
        try:
            if pydicom:
                dcm = pydicom.dcmread(f, stop_before_pixels=True)
                pos = float(dcm.ImagePositionPatient[2])
                file_tuples.append((pos, f))
            else:
                # Fallback sort by numeric filename
                num = int(os.path.basename(f).split(".")[0])
                file_tuples.append((num, f))
        except Exception:
            file_tuples.append((0, f))

    file_tuples.sort(key=lambda x: x[0])
    sorted_files = [t[1] for t in file_tuples]

    # Load all images to compute areas for heuristic
    # (Optimization: could downsample here, but full res is safer for accuracy)
    images = []
    areas = []
    for f in sorted_files:
        img = load_image(f, size=(Config.IMG_SIZE, Config.IMG_SIZE))
        images.append(img)
        areas.append(get_lung_area(img))

    # Adaptive Slice Selection
    if not areas:
        selected_indices = [0, 0, 0]
    else:
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        threshold = max_area * Config.SLICE_THRESHOLD

        # Find Apical (scan upwards/downwards from max depending on sort order)
        # We scan towards index 0
        apical_idx = 0
        for i in range(max_idx, -1, -1):
            if areas[i] < threshold:
                apical_idx = i
                break

        # Find Basal (scan towards end)
        basal_idx = len(images) - 1
        for i in range(max_idx, len(images)):
            if areas[i] < threshold:
                basal_idx = i
                break

        selected_indices = [apical_idx, max_idx, basal_idx]

    # Stack images: (3, H, W)
    final_images = np.stack([images[i] for i in selected_indices], axis=0)

    # Save to cache
    np.save(cache_path, final_images)

    return final_images


class LungDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, load_cached_data=True):
        self.df = df.copy()
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Ensure Baseline_FVC exists
        # For training/val, calculating baseline from history
        if "Baseline_FVC" not in self.df.columns:
            # We assume the baseline is the measurement with the minimum 'Weeks' value
            baseline_df = (
                self.df.sort_values("Weeks").groupby("Patient").first().reset_index()
            )
            baseline_df = baseline_df[["Patient", "FVC"]].rename(
                columns={"FVC": "Baseline_FVC"}
            )
            self.df = self.df.merge(baseline_df, on="Patient", how="left")

        # Categorical Encoders
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image (Cached)
        full_image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])
        imgs = process_patient_images(patient_id, full_image_dir, self.load_cached_data)
        imgs = torch.tensor(imgs, dtype=torch.float32)

        # 2. Tabular Features
        # Weeks: Scaled by 100 for numerical stability
        weeks = float(row["Weeks"]) / 100.0

        # Baseline FVC: Z-score standardized
        base_fvc = (float(row["Baseline_FVC"]) - Config.FVC_MEAN) / Config.FVC_STD

        # Age: Scaled by 100
        age = float(row["Age"]) / 100.0

        # Categoricals
        sex = self.sex_map.get(row["Sex"], 0)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        # 3. Target
        target = 0.0
        if "FVC" in row and self.mode != "submission":
            target = (float(row["FVC"]) - Config.FVC_MEAN) / Config.FVC_STD

        return {
            "image": imgs,
            "weeks": torch.tensor([weeks], dtype=torch.float32),
            "baseline_fvc": torch.tensor([base_fvc], dtype=torch.float32),
            "age": torch.tensor([age], dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.long),
            "smoke": torch.tensor(smoke, dtype=torch.long),
            "target": torch.tensor([target], dtype=torch.float32),
            "patient_id": patient_id,
        }


def get_dataloaders(train_df, val_df):
    """
    Constructs PyTorch DataLoaders for training and validation.
    """
    train_ds = LungDataset(train_df, mode="train", load_cached_data=True)
    val_ds = LungDataset(val_df, mode="val", load_cached_data=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_submission_df(test_df, sample_sub_df):
    """
    Expands the test set metadata to match the submission format (Patient_Week).
    """
    submission_rows = []
    test_lookup = test_df.set_index("Patient")

    for _, row in sample_sub_df.iterrows():
        pw = row["Patient_Week"]
        # Format is PatientID_WeekNum
        parts = pw.rsplit("_", 1)
        patient_id = parts[0]
        target_week = int(parts[1])

        if patient_id in test_lookup.index:
            base_data = test_lookup.loc[patient_id]

            # Calculate relative weeks for the model
            # Model Input 'Weeks' = Target Week - Baseline Week
            baseline_week = base_data["Weeks"]
            relative_weeks = target_week - baseline_week

            new_row = base_data.to_dict()
            new_row["Patient"] = patient_id
            new_row["Weeks"] = relative_weeks
            new_row["Baseline_FVC"] = base_data[
                "FVC"
            ]  # In test set, FVC column is the baseline
            new_row["Patient_Week"] = pw

            submission_rows.append(new_row)

    return pd.DataFrame(submission_rows)
