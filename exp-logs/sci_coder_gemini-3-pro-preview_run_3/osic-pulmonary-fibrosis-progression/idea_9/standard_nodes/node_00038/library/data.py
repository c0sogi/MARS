import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# Constants for Tabular Normalization (from EDA)
AGE_MEAN = 67.58
AGE_STD = 6.62


def get_img_transforms(phase="train"):
    """
    Constructs the Albumentations transform pipeline based on the phase.
    """
    params = Config.get_transforms(phase)
    img_size = params["size"]

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=params.get("horizontal_flip_prob", 0.5)),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=params.get("rotate_limit", 15),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    p=params.get("brightness_contrast_prob", 0.2)
                ),
                A.Normalize(
                    mean=(0.485,), std=(0.229,)
                ),  # ImageNet-like stats for grayscale
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Normalize(mean=(0.485,), std=(0.229,)), ToTensorV2()])


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by instance number,
    and converts to Hounsfield Units.
    """
    if not os.path.exists(path):
        return []

    slices = [
        pydicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.endswith(".dcm")
    ]
    if not slices:
        return []

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel data to Hounsfield Units.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    for n in range(len(slices)):
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return image


def process_patient_images(patient_id, image_dir):
    """
    Content-Adaptive Slice Selection and Preprocessing.
    1. Load scan and convert to HU.
    2. Estimate lung area per slice using thresholding.
    3. Select 3 slices: Apical, Middle (Max Area), Basal.
    4. Resize to Config.IMG_SIZE.
    5. Return stacked array (3, H, W).
    """
    full_path = os.path.join(Config.INPUT_DIR, image_dir)
    slices = load_scan(full_path)

    if not slices:
        # Fallback for missing data: return zeros
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    try:
        image = get_pixels_hu(slices)

        # Lung Window for area estimation: -1000 to -400 HU
        # Simple thresholding to find lung area
        lung_mask = (image > -1000) & (image < -400)
        lung_area = lung_mask.sum(axis=(1, 2))

        # Sort indices by lung area
        valid_indices = np.where(lung_area > 0)[0]

        if len(valid_indices) < 3:
            # Not enough valid slices, just take evenly spaced ones from the whole set
            indices = np.linspace(0, len(slices) - 1, Config.NUM_SLICES, dtype=int)
        else:
            # Heuristic Selection
            # 1. Middle: Max area
            max_area_idx = np.argmax(lung_area)

            # 2. Define "lung region" as slices with > 50% of max area
            max_area = lung_area[max_area_idx]
            roi_indices = np.where(lung_area > 0.5 * max_area)[0]

            if len(roi_indices) < 3:
                indices = np.linspace(
                    roi_indices[0], roi_indices[-1], Config.NUM_SLICES, dtype=int
                )
            else:
                # Select Top (Apical), Middle (Max), Bottom (Basal)
                # We want them physically distributed, so we take min/max of the ROI and the max area idx
                apical = roi_indices[0]  # Top of lungs
                basal = roi_indices[-1]  # Bottom of lungs

                # Ensure they are distinct and sorted
                indices = sorted(list(set([apical, max_area_idx, basal])))

                # If we still don't have 3 (e.g. max is at the top), fill in
                if len(indices) < 3:
                    needed = 3 - len(indices)
                    extra = np.linspace(
                        roi_indices[0], roi_indices[-1], needed + 2, dtype=int
                    )[1:-1]
                    indices = sorted(list(set(indices) | set(extra)))

                # Force strictly 3
                if len(indices) > 3:
                    # Pick first, middle, last
                    indices = [indices[0], indices[len(indices) // 2], indices[-1]]
                elif len(indices) < 3:
                    # Duplicate middle if desperate
                    while len(indices) < 3:
                        indices.append(indices[-1])

        # Extract selected slices
        selected_imgs = []
        for idx in indices:
            img = image[idx]

            # Normalize to standard window [-1000, 400] roughly for medical viewing
            # Then map to 0-255 for standard CNN processing
            img = np.clip(img, -1000, 400)
            img = (img - (-1000)) / (400 - (-1000))  # 0 to 1
            img = (img * 255).astype(np.uint8)

            # Resize
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
            selected_imgs.append(img)

        return np.stack(selected_imgs)  # (3, H, W)

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )


def get_cached_image(patient_id, image_path, load_cached_data=True):
    """
    Retrieves image from cache or processes it from scratch.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to reprocessing

    # Process
    img_data = process_patient_images(patient_id, image_path)

    # Save to cache
    np.save(cache_path, img_data)

    return img_data


class OSICDataset(Dataset):
    def __init__(self, df, phase="train", load_cached_data=True):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.load_cached_data = load_cached_data
        self.transforms = get_img_transforms(phase)

        # Pre-compute baseline lookup for relative time calculation
        # We need the baseline week and baseline FVC for every patient
        # The metadata df usually contains all visits.
        # For test set, it only contains baseline.

        # Group by patient to find baseline (Week 0 or closest to it, usually first row in sorted)
        # However, the provided metadata/train.csv is a subset.
        # We need to ensure we have the correct baseline reference.
        # In this competition, 'Weeks' is relative to baseline CT. So Baseline Week is effectively 0 for the CT.
        # But the FVC measurement might be at a different week.
        # The prompt says: "Weeks: the relative number of weeks pre/post the baseline CT".
        # So Week=0 is the CT scan time.
        # The "Baseline FVC" is the first FVC measurement provided.

        # For the purpose of this dataset class, we will treat the input df as containing
        # the target visits we want to predict.
        # We need to attach the "Baseline FVC" and "Baseline Week" (which is the week of that first FVC)
        # to each row.

        # In train.csv, we have full history. We can find the first visit per patient.
        self.patient_meta = {}
        unique_patients = self.df["Patient"].unique()

        for pid in unique_patients:
            p_data = self.df[self.df["Patient"] == pid].sort_values("Weeks")
            # Baseline is the first available measurement
            base_fvc = p_data.iloc[0]["FVC"]
            base_week = p_data.iloc[0]["Weeks"]
            base_age = p_data.iloc[0]["Age"]
            base_sex = p_data.iloc[0]["Sex"]
            base_smoke = p_data.iloc[0]["SmokingStatus"]
            base_path = p_data.iloc[0]["image_path"]

            self.patient_meta[pid] = {
                "base_fvc": base_fvc,
                "base_week": base_week,
                "age": base_age,
                "sex": base_sex,
                "smoke": base_smoke,
                "image_path": base_path,
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]
        meta = self.patient_meta[pid]

        # 1. Image Data
        # Shape: (3, H, W) -> Transpose to (H, W, 3) for Albumentations
        img_raw = get_cached_image(pid, meta["image_path"], self.load_cached_data)
        img_raw = np.transpose(img_raw, (1, 2, 0))

        if self.transforms:
            augmented = self.transforms(image=img_raw)
            img_tensor = augmented["image"]  # (3, H, W)
        else:
            img_tensor = torch.tensor(img_raw).permute(2, 0, 1)

        # 2. Tabular Data
        # Normalize Baseline FVC
        base_fvc_norm = (meta["base_fvc"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Normalize Age
        age_norm = (meta["age"] - AGE_MEAN) / AGE_STD

        # Encode Sex (Male=0, Female=1)
        sex_enc = 0.0 if meta["sex"] == "Male" else 1.0

        # Encode Smoking (Never=0, Ex=1, Current=2)
        smoke_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
        smoke_enc = smoke_map.get(meta["smoke"], 0.0)

        # Stack tabular features
        tabular = torch.tensor(
            [base_fvc_norm, age_norm, sex_enc, smoke_enc], dtype=torch.float32
        )

        # 3. Time Engineering
        # Relative time from the baseline measurement
        # t_rel = (Current_Week - Baseline_Week) / Scale
        current_week = row["Weeks"]
        t_rel = (current_week - meta["base_week"]) / Config.TIME_SCALE
        t_rel = torch.tensor([t_rel], dtype=torch.float32)

        # 4. Target
        # Standardized FVC
        target_val = row["FVC"]
        target_norm = (target_val - Config.TARGET_MEAN) / Config.TARGET_STD
        target = torch.tensor([target_norm], dtype=torch.float32)

        # Return dictionary or tuple? Tuple is standard for simple loaders
        # Returns: image, tabular, t_rel, target
        return img_tensor, tabular, t_rel, target, pid, current_week


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Create Datasets
    train_ds = OSICDataset(train_df, phase="train", load_cached_data=True)
    val_ds = OSICDataset(val_df, phase="val", load_cached_data=True)
    test_ds = OSICDataset(test_df, phase="test", load_cached_data=True)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
