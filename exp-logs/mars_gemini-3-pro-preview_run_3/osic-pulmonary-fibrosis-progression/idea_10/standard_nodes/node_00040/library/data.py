import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from library.config import Config
from library.utils import seed_everything

# -------------------------------------------------------------------------
# Image Processing Helper Functions
# -------------------------------------------------------------------------


def get_img_path(patient_id, dataset_type="train"):
    """
    Constructs the path to the DICOM directory for a given patient.
    """
    if dataset_type == "train" or dataset_type == "val":
        return os.path.join(Config.TRAIN_DICOM_DIR, patient_id)
    else:
        return os.path.join(Config.TEST_DICOM_DIR, patient_id)


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber (Z-position).
    """
    if not os.path.exists(path):
        return []

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s), stop_before_pixels=False)
                slices.append(ds)
            except Exception:
                continue

    # Sort by InstanceNumber if available, else by ImagePositionPatient Z
    slices.sort(
        key=lambda x: int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
    )

    # Secondary sort by ImagePositionPatient if InstanceNumber is unreliable
    if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient"):
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    """
    image = np.stack([s.pixel_array.astype(np.float32) for s in scans])

    # Convert to HU
    for i, s in enumerate(scans):
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.float32)

        image[i] += np.float32(intercept)

    return image


def select_smart_slices(image_hu):
    """
    Implements the Content-Adaptive Heuristic to select 3 slices:
    1. Anchor (Middle): Max lung area.
    2. Apical: Upper lung (approx 50% of max area).
    3. Basal: Lower lung (approx 50% of max area).
    """
    n_slices = len(image_hu)
    if n_slices == 0:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    if n_slices < 3:
        # Padding if fewer than 3 slices
        selected = [image_hu[0] for _ in range(3)]
        return np.array(selected)

    # Calculate approximate lung area for each slice
    # Lung window is typically -1000 to -400 HU.
    # We threshold to count pixels in this range.
    areas = []
    for i in range(n_slices):
        slice_img = image_hu[i]
        # Simple thresholding for air/lung tissue
        mask = (slice_img > -1000) & (slice_img < -400)
        areas.append(np.sum(mask))

    areas = np.array(areas)

    # Anchor: Slice with maximum lung area
    idx_anchor = np.argmax(areas)
    max_area = areas[idx_anchor]

    # Threshold for boundary slices
    threshold = 0.5 * max_area

    # Find Apical (above anchor)
    idx_apical = 0
    for i in range(idx_anchor, -1, -1):
        if areas[i] < threshold:
            idx_apical = i
            break

    # Find Basal (below anchor)
    idx_basal = n_slices - 1
    for i in range(idx_anchor, n_slices):
        if areas[i] < threshold:
            idx_basal = i
            break

    # Safety check: ensure indices are distinct if possible, or at least sorted
    indices = sorted([idx_apical, idx_anchor, idx_basal])

    # Extract slices
    selected_slices = image_hu[indices]

    return selected_slices


def process_scan(scans):
    """
    Full processing pipeline for a single patient's scan list.
    Returns a (H, W, 3) numpy array normalized to 0-1 range.
    """
    # Get HU
    image_hu = get_pixels_hu(scans)

    # Select Slices
    selected_slices = select_smart_slices(image_hu)

    # Resize and Normalize (Lung Windowing)
    # Lung Window: W=1500, L=-600 -> Range [-1350, 150]
    processed_slices = []
    for i in range(3):
        slice_img = selected_slices[i]

        # Apply Lung Window
        window_center = -600
        window_width = 1500
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        slice_img = np.clip(slice_img, img_min, img_max)

        # Normalize to 0-1
        slice_img = (slice_img - img_min) / (img_max - img_min)

        # Resize
        slice_img = cv2.resize(slice_img, (Config.IMG_SIZE, Config.IMG_SIZE))
        processed_slices.append(slice_img)

    # Stack to (H, W, 3) for Albumentations/CNN
    image_stacked = np.stack(processed_slices, axis=-1)

    return image_stacked.astype(np.float32)


def cache_images(patient_ids, dataset_type="train", load_cached_data=True):
    """
    Iterates through patient IDs, processes their CT scans, and caches the result.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    missing_patients = []

    print(
        f"Checking/Processing images for {len(patient_ids)} patients ({dataset_type})..."
    )

    for pid in tqdm(patient_ids):
        cache_path = os.path.join(Config.CACHE_DIR, f"{pid}.npy")

        if load_cached_data and os.path.exists(cache_path):
            continue

        # Process from scratch
        dicom_dir = get_img_path(pid, dataset_type)
        if not os.path.exists(dicom_dir):
            # If directory doesn't exist (should be rare given metadata check), create a blank placeholder
            # to avoid crashing, but log it.
            # In this competition, we expect data to be present.
            # Create a black image
            img_data = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        else:
            scans = load_scan(dicom_dir)
            if not scans:
                img_data = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                )
            else:
                try:
                    img_data = process_scan(scans)
                except Exception as e:
                    print(f"Error processing {pid}: {e}")
                    img_data = np.zeros(
                        (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                    )

        np.save(cache_path, img_data)


# -------------------------------------------------------------------------
# Tabular Data Processing
# -------------------------------------------------------------------------


def preprocess_tabular_data(train_df, val_df, test_df, sample_sub_df=None):
    """
    Preprocesses tabular data:
    1. Extracts Baseline FVC/Weeks for each patient.
    2. Expands Test set using sample_submission.csv (if provided).
    3. Computes Relative Time.
    4. Encodes Categoricals.
    5. Standardizes Features and Target.
    """

    # 1. Helper to extract baseline (Visit with min absolute weeks, usually 0)
    def get_baseline(df):
        # Sort by patient and absolute weeks to find the closest to 0
        df_temp = df.copy()
        df_temp["Abs_Weeks"] = df_temp["Weeks"].abs()
        df_temp = df_temp.sort_values(["Patient", "Abs_Weeks"])
        # Drop duplicates to keep first (best) baseline per patient
        baseline = df_temp.drop_duplicates(subset=["Patient"], keep="first")
        return baseline[
            ["Patient", "FVC", "Weeks", "Age", "Sex", "SmokingStatus"]
        ].rename(columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"})

    # Extract baselines from the respective datasets
    # Note: For train/val, we use their own history to find baseline.
    # For test, the provided test.csv IS the baseline info.
    train_baseline = get_baseline(train_df)
    val_baseline = get_baseline(val_df)
    test_baseline = test_df.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )[["Patient", "Baseline_FVC", "Baseline_Week", "Age", "Sex", "SmokingStatus"]]

    # 2. Merge Baseline info back to original dataframes
    train_df = train_df.merge(
        train_baseline[["Patient", "Baseline_FVC", "Baseline_Week"]],
        on="Patient",
        how="left",
    )
    val_df = val_df.merge(
        val_baseline[["Patient", "Baseline_FVC", "Baseline_Week"]],
        on="Patient",
        how="left",
    )

    # 3. Handle Test Set Expansion
    # The test.csv only has 1 row per patient. We need to predict for all weeks in sample_submission.
    if sample_sub_df is not None:
        # Parse Patient and Week from Patient_Week column
        # Format: ID..._WeekNum
        sub_df = sample_sub_df.copy()
        sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

        # Merge baseline info onto the submission structure
        test_expanded = sub_df.merge(test_baseline, on="Patient", how="left")

        # Keep relevant columns
        # Note: 'FVC' in sub_df is just a placeholder/prediction target, we don't use it as input.
        test_df = test_expanded
    else:
        # If no sample sub provided, just prepare test_df as is (likely for debugging)
        test_df = test_df.merge(
            test_baseline[["Patient", "Baseline_FVC", "Baseline_Week"]],
            on="Patient",
            how="left",
        )

    # 4. Compute Relative Time (t_rel)
    for df in [train_df, val_df, test_df]:
        df["Rel_Weeks"] = df["Weeks"] - df["Baseline_Week"]
        # Scale Relative Time immediately (Config.TIME_SCALE = 0.01)
        df["Rel_Weeks_Scaled"] = df["Rel_Weeks"] * Config.TIME_SCALE

    # 5. Encode Categoricals
    # Sex: Male=1, Female=0
    # Smoking: Ex-smoker=1, Never smoked=0, Currently smokes=2 (Arbitrary mapping, will be standardized)
    le_sex = LabelEncoder()
    le_smoke = LabelEncoder()

    # Fit on all available data to cover all categories
    all_sex = pd.concat([train_df["Sex"], val_df["Sex"], test_df["Sex"]], axis=0)
    all_smoke = pd.concat(
        [train_df["SmokingStatus"], val_df["SmokingStatus"], test_df["SmokingStatus"]],
        axis=0,
    )

    le_sex.fit(all_sex)
    le_smoke.fit(all_smoke)

    for df in [train_df, val_df, test_df]:
        df["Sex_Code"] = le_sex.transform(df["Sex"])
        df["Smoking_Code"] = le_smoke.transform(df["SmokingStatus"])

    # 6. Standardization (Z-score)
    # Fit scalers ONLY on Training Data
    scaler_fvc = StandardScaler()
    scaler_age = StandardScaler()
    scaler_base_fvc = StandardScaler()
    scaler_sex = StandardScaler()
    scaler_smoke = StandardScaler()

    # Fit
    scaler_fvc.fit(train_df[["FVC"]])
    scaler_age.fit(train_df[["Age"]])
    scaler_base_fvc.fit(train_df[["Baseline_FVC"]])
    scaler_sex.fit(train_df[["Sex_Code"]])
    scaler_smoke.fit(train_df[["Smoking_Code"]])

    # Apply
    for df in [train_df, val_df, test_df]:
        # Target (FVC) - only exists in Train/Val (and Test if provided, but usually ignored for inference)
        if "FVC" in df.columns:
            # For Test set expanded from submission, FVC column exists but is dummy.
            # We only transform if it's real data (Train/Val).
            # However, to keep code simple, we transform if column exists.
            # For test inference, we don't use the transformed target column anyway.
            df["FVC_Scaled"] = scaler_fvc.transform(df[["FVC"]])

        df["Age_Scaled"] = scaler_age.transform(df[["Age"]])
        df["Baseline_FVC_Scaled"] = scaler_base_fvc.transform(df[["Baseline_FVC"]])
        df["Sex_Scaled"] = scaler_sex.transform(df[["Sex_Code"]])
        df["Smoking_Scaled"] = scaler_smoke.transform(df[["Smoking_Code"]])

    # Store statistics for inverse transform later
    stats = {"fvc_mean": scaler_fvc.mean_[0], "fvc_std": scaler_fvc.scale_[0]}

    return train_df, val_df, test_df, stats


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Extract columns to numpy for faster access
        self.patient_ids = df["Patient"].values
        self.rel_weeks = df["Rel_Weeks_Scaled"].values.astype(np.float32)

        # Static Features: [Baseline_FVC, Age, Sex, Smoking]
        self.static_features = df[
            ["Baseline_FVC_Scaled", "Age_Scaled", "Sex_Scaled", "Smoking_Scaled"]
        ].values.astype(np.float32)

        # Target
        if mode != "test":
            self.targets = df["FVC_Scaled"].values.astype(np.float32)

        # Keep track of original FVC for metric calculation if needed
        if "FVC" in df.columns:
            self.raw_fvc = df["FVC"].values

        # For submission file generation
        if "Patient_Week" in df.columns:
            self.patient_weeks = df["Patient_Week"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # Load Image from Cache
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")
        try:
            image = np.load(cache_path)
        except FileNotFoundError:
            # Fallback (should be handled by cache_images, but for safety)
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Apply Albumentations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Default to tensor conversion
            image = ToTensorV2()(image=image)["image"]

        # Get Features
        static = torch.tensor(self.static_features[idx], dtype=torch.float32)
        rel_time = torch.tensor([self.rel_weeks[idx]], dtype=torch.float32)

        # Return based on mode
        if self.mode == "test":
            return {
                "image": image,
                "static": static,
                "rel_time": rel_time,
                "patient_week": self.patient_weeks[idx],
            }
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return {
                "image": image,
                "static": static,
                "rel_time": rel_time,
                "target": target,
                "raw_fvc": self.raw_fvc[idx],
            }


# -------------------------------------------------------------------------
# Main Data Loading Function
# -------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    """
    Main entry point for data loading.
    1. Loads metadata.
    2. Caches images.
    3. Preprocesses tabular data.
    4. Creates DataLoaders.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.METADATA_TRAIN)
    val_df = pd.read_csv(Config.METADATA_VAL)
    test_df = pd.read_csv(Config.METADATA_TEST)
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))

    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        # Keep test small but valid
        test_patients = test_df["Patient"].unique()[:2]
        test_df = test_df[test_df["Patient"].isin(test_patients)]
        sample_sub = sample_sub[
            sample_sub["Patient_Week"].str.contains("|".join(test_patients))
        ]

    # 2. Cache Images
    # Identify all unique patients
    train_patients = train_df["Patient"].unique()
    val_patients = val_df["Patient"].unique()
    test_patients = test_df["Patient"].unique()

    cache_images(
        np.concatenate([train_patients, val_patients]),
        dataset_type="train",
        load_cached_data=load_cached_data,
    )
    cache_images(test_patients, dataset_type="test", load_cached_data=load_cached_data)

    # 3. Preprocess Tabular Data
    train_df, val_df, test_df, stats = preprocess_tabular_data(
        train_df, val_df, test_df, sample_sub
    )

    # 4. Define Transforms
    train_transforms = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CoarseDropout(
                max_holes=8,
                max_height=Config.IMG_SIZE // 10,
                max_width=Config.IMG_SIZE // 10,
                p=0.3,
            ),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
    )

    # 5. Create Datasets
    train_dataset = LungDataset(train_df, transforms=train_transforms, mode="train")
    val_dataset = LungDataset(val_df, transforms=val_transforms, mode="val")
    test_dataset = LungDataset(test_df, transforms=val_transforms, mode="test")

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader, stats
