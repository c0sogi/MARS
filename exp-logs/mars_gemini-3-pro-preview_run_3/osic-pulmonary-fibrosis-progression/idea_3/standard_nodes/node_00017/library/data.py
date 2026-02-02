import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from library.config import Config

# Attempt to import pydicom, set flag for fallback mechanism
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_dicom_image(path):
    """
    Reads a DICOM file and returns the pixel array converted to Hounsfield Units (HU).
    Includes a robust fallback for environments where pydicom is missing or files are raw.
    """
    # 1. Try standard pydicom read
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array.astype(np.float32)

            # Convert to HU
            intercept = getattr(dcm, "RescaleIntercept", -1024)
            slope = getattr(dcm, "RescaleSlope", 1)
            image = image * slope + intercept
            return image
        except Exception:
            pass  # Fall through to fallback

    # 2. Fallback: Raw binary read assuming standard 512x512 CT
    # This handles cases where pydicom is missing or fails, but files are valid raw DICOMs
    img_dim = 512
    expected_pixels = img_dim * img_dim
    expected_bytes = expected_pixels * 2  # uint16/int16 = 2 bytes

    try:
        file_size = os.path.getsize(path)
        if file_size >= expected_bytes:
            with open(path, "rb") as f:
                # DICOM pixel data is usually at the end. Seek to the last N bytes.
                f.seek(-expected_bytes, 2)
                buffer = f.read(expected_bytes)
                image = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
                image = image.reshape((img_dim, img_dim))

                # Apply standard CT intercept if unknown (-1024 is air/offset)
                image = image + (-1024)
                return image
    except Exception:
        pass

    # 3. Last Resort: Return empty air slice
    return np.zeros((512, 512), dtype=np.float32) - 1000.0


def process_patient_images(patient_id, image_dir, cache_dir, load_cached_data=True):
    """
    Selects 3 adaptive slices (Apical, Middle, Basal) for a patient.
    Caches the processed 3x256x256 tensor to disk.
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Check for Channel Last format (H, W, C) -> (256, 256, 3)
            if data.shape == (Config.img_size, Config.img_size, Config.num_slices):
                data = np.transpose(data, (2, 0, 1))

            # Verify shape matches expected (C, H, W)
            if data.shape == (Config.num_slices, Config.img_size, Config.img_size):
                return data.astype(np.float32)
        except Exception:
            pass  # Cache corrupted, recompute

    # Locate patient directory
    # image_dir is expected to be the parent folder (e.g. input/train)
    patient_dir = os.path.join(image_dir, patient_id)
    if not os.path.exists(patient_dir):
        # Return zeros if data missing
        return np.zeros(
            (Config.num_slices, Config.img_size, Config.img_size), dtype=np.float32
        )

    # Get all DICOM files
    files = glob.glob(os.path.join(patient_dir, "*.dcm"))
    # Sort numerically by instance number (filename)
    try:
        files = sorted(
            files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
    except ValueError:
        files = sorted(files)  # Fallback to string sort

    if not files:
        return np.zeros(
            (Config.num_slices, Config.img_size, Config.img_size), dtype=np.float32
        )

    # Scan all slices to calculate lung area (HU between -1000 and -400)
    slice_infos = []
    for i, fpath in enumerate(files):
        img = read_dicom_image(fpath)
        # Lung window check
        lung_pixels = np.sum((img > -1000) & (img < -400))
        slice_infos.append({"index": i, "path": fpath, "lung_pixels": lung_pixels})

    # Sort by index to ensure spatial order
    slice_infos.sort(key=lambda x: x["index"])

    # --- Content-Adaptive Selection Heuristic ---
    if not slice_infos:
        return np.zeros(
            (Config.num_slices, Config.img_size, Config.img_size), dtype=np.float32
        )

    # 1. Find Anchor (Slice with Max Lung Area)
    max_lung_slice = max(slice_infos, key=lambda x: x["lung_pixels"])
    max_val = max_lung_slice["lung_pixels"]
    anchor_idx = max_lung_slice["index"]

    # 2. Find Bounds (50% of max area)
    threshold = max_val * 0.5

    # Scan Up (towards head/lower index)
    upper_idx = 0
    for i in range(anchor_idx, -1, -1):
        if slice_infos[i]["lung_pixels"] < threshold:
            upper_idx = i
            break

    # Scan Down (towards feet/higher index)
    lower_idx = len(slice_infos) - 1
    for i in range(anchor_idx, len(slice_infos)):
        if slice_infos[i]["lung_pixels"] < threshold:
            lower_idx = i
            break

    selected_indices = [upper_idx, anchor_idx, lower_idx]

    # Handle collisions (if patient has very few slices)
    if len(slice_infos) >= 3:
        if selected_indices[0] == selected_indices[1]:
            selected_indices[0] = max(0, selected_indices[1] - 1)
        if selected_indices[2] == selected_indices[1]:
            selected_indices[2] = min(len(slice_infos) - 1, selected_indices[1] + 1)

    # Load, Resize, Normalize selected slices
    processed_slices = []
    for idx in selected_indices:
        info = slice_infos[idx]
        img = read_dicom_image(info["path"])

        # Resize
        img = cv2.resize(img, (Config.img_size, Config.img_size))

        # Normalize: Clip to broad lung window [-1000, 400] then scale to [0, 1]
        min_hu, max_hu = -1000.0, 400.0
        img = np.clip(img, min_hu, max_hu)
        img = (img - min_hu) / (max_hu - min_hu)

        processed_slices.append(img)

    # Stack -> (3, H, W)
    tensor_img = np.stack(processed_slices, axis=0).astype(np.float32)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, tensor_img)

    return tensor_img


class LungDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cache_dir = cache_dir

        # Normalization Stats (Derived from Training Set EDA)
        self.age_mean = 67.58
        self.age_std = 6.63
        self.weeks_mean = 31.38
        self.weeks_std = 23.46
        self.base_fvc_mean = 2654.65
        self.base_fvc_std = 801.70

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Augmentations for training (Cite Lesson 15 - preventing overfitting on frozen features)
        self.transforms = (
            A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                ]
            )
            if mode == "train"
            else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- Image Loading ---
        # Construct parent directory for the patient
        # row['image_path'] is relative, e.g., "train/ID..."
        full_path = os.path.join(Config.input_dir, row["image_path"])
        parent_dir = os.path.dirname(full_path)  # e.g., input/train

        img_tensor = process_patient_images(
            patient_id, parent_dir, self.cache_dir, load_cached_data=True
        )

        # Apply Augmentations if training
        if self.transforms:
            # Transpose (C, H, W) -> (H, W, C) for albumentations
            img_np = np.transpose(img_tensor, (1, 2, 0))
            augmented = self.transforms(image=img_np)["image"]
            # Transpose back (H, W, C) -> (C, H, W)
            img_tensor = np.transpose(augmented, (2, 0, 1))

        # --- Tabular Loading ---
        # Normalize Inputs
        age = (row["Age"] - self.age_mean) / self.age_std
        weeks = (row["Weeks"] - self.weeks_mean) / self.weeks_std

        # Baseline FVC (Must be present in DF)
        base_fvc = row.get("Baseline_FVC", 0)
        base_fvc = (base_fvc - self.base_fvc_mean) / self.base_fvc_std

        sex = self.sex_map.get(row["Sex"], 0)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        tabular = np.array([age, sex, smoke, weeks, base_fvc], dtype=np.float32)

        # --- Target Loading ---
        if self.mode != "test":
            target_raw = row["FVC"]
            # Z-score standardization for target
            target = (target_raw - Config.target_mean) / Config.target_std
            return (
                torch.tensor(img_tensor, dtype=torch.float32),
                torch.tensor(tabular, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
        else:
            return torch.tensor(img_tensor, dtype=torch.float32), torch.tensor(
                tabular, dtype=torch.float32
            )


def add_baseline_fvc(df):
    """
    Identifies the baseline visit (min Weeks) for each patient and adds 'Baseline_FVC' column.
    """
    # Sort by Patient and Weeks to find the first visit
    baseline_df = (
        df.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
    )

    # Create map
    baseline_map = dict(zip(baseline_df["Patient"], baseline_df["FVC"]))

    # Apply map
    df["Baseline_FVC"] = df["Patient"].map(baseline_map)
    return df


def get_dataloaders(batch_size=Config.batch_size, num_workers=Config.num_workers):
    """
    Creates training and validation dataloaders.
    """
    # Load metadata
    train_df = pd.read_csv(Config.train_csv_path)
    val_df = pd.read_csv(Config.val_csv_path)

    # Add Baseline FVC feature
    train_df = add_baseline_fvc(train_df)
    val_df = add_baseline_fvc(val_df)

    # Instantiate Datasets
    train_ds = LungDataset(train_df, mode="train", cache_dir=Config.cache_dir)
    val_ds = LungDataset(val_df, mode="val", cache_dir=Config.cache_dir)

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def prepare_inference_data():
    """
    Prepares the test loader for submission generation.
    Expands the test set to cover all Patient_Weeks requested in sample_submission.csv.
    """
    # Load base test metadata (contains Baseline info)
    test_df = pd.read_csv(Config.test_csv_path)

    # Load submission template
    sub_df = pd.read_csv(os.path.join(Config.input_dir, "sample_submission.csv"))

    # Parse Patient and Weeks from 'Patient_Week' column
    # Example: ID00419637202311204720264_18
    def parse_id(s):
        p, w = s.rsplit("_", 1)
        return p, int(w)

    sub_df["Patient"], sub_df["Weeks"] = zip(*sub_df["Patient_Week"].map(parse_id))

    # Prepare metadata for merge
    # Rename FVC in test.csv to Baseline_FVC, as test.csv only contains the initial visit
    meta_cols = ["Patient", "Age", "Sex", "SmokingStatus", "FVC", "image_path"]
    test_meta = test_df[meta_cols].copy()
    test_meta = test_meta.rename(columns={"FVC": "Baseline_FVC"})

    # Merge metadata onto the submission template
    inference_df = pd.merge(sub_df, test_meta, on="Patient", how="left")

    # Create Dataset and Loader
    test_ds = LungDataset(inference_df, mode="test", cache_dir=Config.cache_dir)

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader, sub_df
