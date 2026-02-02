import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from library.config import Config, seed_everything

# ==========================================
# 1. Helper Functions for Image Processing
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber,
    and returns a list of pydicom datasets.
    """
    if not os.path.exists(path):
        return []

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception:
                continue

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
    Converts a list of dicom slices to a numpy array of Hounsfield Units.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield Units (HU)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def generate_tri_slab(volume, view="axial"):
    """
    Generates a 3-channel image using Tri-Slab MIP logic.

    Args:
        volume: 3D numpy array (Depth, Height, Width)
        view: 'axial' or 'coronal'

    Returns:
        2D numpy array (H, W, 3) normalized to 0-255
    """
    # Permute volume based on view
    if view == "coronal":
        # Axial is (D, H, W). Coronal is (H, D, W) conceptually for slicing Y-axis
        # We want to slice along the Height dimension (axis 1)
        # So we move axis 1 to axis 0: (H, D, W)
        volume = np.transpose(volume, (1, 0, 2))

    # Now volume is (Slices, H_img, W_img)
    num_slices = volume.shape[0]

    if num_slices == 0:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    # Define slab boundaries with overlap
    # We want 3 slabs: 0-33%, 33-66%, 66-100% roughly, but with overlap.
    # Let's use fixed percentages that sum > 100% to ensure overlap.
    # E.g., 0.0-0.4, 0.3-0.7, 0.6-1.0

    p1_end = int(num_slices * 0.40)
    p2_start = int(num_slices * 0.30)
    p2_end = int(num_slices * 0.70)
    p3_start = int(num_slices * 0.60)

    # Ensure indices are valid
    p1_end = max(1, p1_end)
    p2_start = min(p2_start, num_slices - 2)
    p2_end = max(p2_start + 1, p2_end)
    p3_start = min(p3_start, num_slices - 1)

    slab1 = volume[0:p1_end, :, :]
    slab2 = volume[p2_start:p2_end, :, :]
    slab3 = volume[p3_start:, :, :]

    # Compute MIP (Maximum Intensity Projection)
    # Handle empty slabs just in case
    m1 = np.max(slab1, axis=0) if slab1.size > 0 else np.zeros_like(volume[0])
    m2 = np.max(slab2, axis=0) if slab2.size > 0 else np.zeros_like(volume[0])
    m3 = np.max(slab3, axis=0) if slab3.size > 0 else np.zeros_like(volume[0])

    # Stack to 3 channels
    img = np.stack([m1, m2, m3], axis=-1)

    # Normalize HU to 0-255 range
    # Standard lung window: W=1500, L=-600 -> [-1350, 150]
    # Or just min-max clipping for robustness
    min_hu = -1000
    max_hu = 400

    img = np.clip(img, min_hu, max_hu)
    img = (img - min_hu) / (max_hu - min_hu)
    img = (img * 255).astype(np.uint8)

    # Resize to target size
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return img


def process_patient_images(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Handles caching logic for patient images.
    Returns tuple (axial_img, coronal_img).
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    full_dicom_path = os.path.join(Config.INPUT_DIR, dicom_dir)
    slices = load_scan(full_dicom_path)

    if not slices:
        # Return black images if no dicoms found
        axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
    else:
        try:
            # Cite debug_lesson_2: Handle Missing Codec Dependencies Gracefully
            vol_hu = get_pixels_hu(slices)
            axial = generate_tri_slab(vol_hu, view="axial")
            coronal = generate_tri_slab(vol_hu, view="coronal")
        except (RuntimeError, Exception) as e:
            print(
                f"Warning: Failed to process images for {patient_id}. Using black placeholder. Error: {e}"
            )
            axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    # 3. Save to cache
    try:
        np.save(axial_path, axial)
        np.save(coronal_path, coronal)
    except Exception as e:
        print(f"Warning: Failed to save cache for {patient_id}: {e}")

    return axial, coronal


# ==========================================
# 2. Tabular Preprocessor
# ==========================================


class TabularPreprocessor:
    def __init__(self):
        self.numeric_features = ["Age", "Percent"]
        self.categorical_features = ["Sex", "SmokingStatus"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_features,
                ),
            ]
        )

    def fit(self, df):
        self.preprocessor.fit(df)

    def transform(self, df):
        return self.preprocessor.transform(df)

    def get_feature_dim(self):
        # Helper to check output dimension
        if hasattr(self.preprocessor, "transformers_"):
            # This is an estimation, actual dim is checked after transform
            pass
        return 0


# ==========================================
# 3. Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        tabular_preprocessor,
        mode="train",
        transform=None,
        load_cached_data=True,
    ):
        """
        Args:
            df: DataFrame containing patient info.
            tabular_preprocessor: Fitted TabularPreprocessor instance.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transform.
            load_cached_data: Boolean to enable/disable loading from cache.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Precompute tabular features
        self.tabular_features = tabular_preprocessor.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image Loading (with caching)
        dicom_dir = row["dicom_dir"]
        axial_img, coronal_img = process_patient_images(
            patient_id, dicom_dir, Config.CACHE_DIR, self.load_cached_data
        )

        # 2. Augmentations
        if self.transform:
            # Apply same transform to both or independent?
            # Usually independent is fine for robustness, but here they are different views of same patient.
            # Let's apply independent spatial augs.
            aug_axial = self.transform(image=axial_img)["image"]
            aug_coronal = self.transform(image=coronal_img)["image"]
        else:
            # Just normalize and to tensor
            default_t = A.Compose(
                [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
            )
            aug_axial = default_t(image=axial_img)["image"]
            aug_coronal = default_t(image=coronal_img)["image"]

        # 3. Tabular Data
        tab_vec = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # 4. Target
        # For training/val, we have FVC. For test, we might have dummy.
        # We also need Weeks for the physics-based head if we were using it,
        # but the idea description says "strictly exclude Week from input" for the head.
        # However, the model output is alpha, sigma_base, sigma_growth.
        # The LOSS function needs True FVC.
        # The inference logic uses Week.
        # So we pass Week as metadata, not feature.

        target = 0.0
        if "FVC" in row:
            target = float(row["FVC"])

        week = float(row["Weeks"]) if "Weeks" in row else float(row["Predict_Week"])
        baseline_week = 0.0
        baseline_fvc = 0.0

        # Handle Baseline info for Test vs Train
        # In train, baseline is Week 0 usually, but we treat every row as a sample.
        # The model predicts parameters, we need (Week - Baseline_Week) for loss calculation?
        # Idea 47 says: "Model predicts alpha, sigma_base, sigma_growth".
        # "FVC = Baseline_FVC + alpha * (Week - Baseline_Week)"
        # So we need Baseline_FVC and Baseline_Week available in the batch.

        if self.mode == "test":
            baseline_fvc = float(row["Baseline_FVC"])
            baseline_week = float(row["Baseline_Week"])
        else:
            # In training, we need to find the baseline for this patient.
            # But the dataframe provided in metadata/train.csv has all weeks.
            # We can assume the row with Weeks=0 is baseline, or pass it explicitly.
            # For simplicity in this dataset class, we might need to look it up or
            # assume the caller has prepared the DF.
            # However, usually we just pass the current FVC as target.
            # The model forward pass might need baseline info.
            # Let's try to extract baseline from the dataframe if possible,
            # or rely on the fact that we are training to predict FVC at 'Week'.
            # Wait, if the model predicts slope alpha, it needs delta_t.
            # Delta_t = Week - Baseline_Week.
            # For training, we can treat the *current* visit as target, and we need the baseline visit info as input.
            # The metadata/train.csv doesn't explicitly have "Baseline_FVC" column for every row.
            # We should probably merge baseline info into train df in get_dataloaders.

            # Check if columns exist (pre-merged)
            if "Baseline_FVC" in row:
                baseline_fvc = float(row["Baseline_FVC"])
                baseline_week = float(row["Baseline_Week"])
            else:
                # Fallback: if not merged, maybe we don't use baseline in input?
                # But the formula requires it.
                # We will assume get_dataloaders handles the merge.
                baseline_fvc = 0.0  # Should be handled in setup
                baseline_week = 0.0

        return {
            "image_axial": aug_axial,
            "image_coronal": aug_coronal,
            "tabular": tab_vec,
            "target": torch.tensor(target, dtype=torch.float32),
            "week": torch.tensor(week, dtype=torch.float32),
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "baseline_week": torch.tensor(baseline_week, dtype=torch.float32),
            "patient_id": patient_id,
        }


# ==========================================
# 4. Data Loaders Factory
# ==========================================


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Constructs and returns dataloaders for train, val, and test.
    Also handles the preparation of tabular preprocessor and baseline merging.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Baseline Info for Training/Val Data
    # The model relies on Baseline_FVC and Baseline_Week to compute predictions.
    # For train/val, we identify the baseline visit (Weeks approx 0 or min abs week)
    # and merge it back to all rows for that patient.

    def merge_baseline(df):
        # Find baseline rows: usually Weeks == 0, or closest to 0
        # We'll sort by absolute week and take first
        df["Abs_Week"] = df["Weeks"].abs()
        baselines = (
            df.sort_values(["Patient", "Abs_Week"])
            .groupby("Patient")
            .first()
            .reset_index()
        )

        # Select relevant baseline cols
        base_cols = [
            "Patient",
            "FVC",
            "Weeks",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
        ]
        baselines = baselines[base_cols]
        baselines = baselines.rename(
            columns={
                "FVC": "Baseline_FVC",
                "Weeks": "Baseline_Week",
                "Percent": "Baseline_Percent",
                "Age": "Baseline_Age",
                "Sex": "Baseline_Sex",
                "SmokingStatus": "Baseline_SmokingStatus",
            }
        )

        # Merge back
        # Note: We keep the original 'Age', 'Sex' etc as current features,
        # but the prompt implies using Baseline metadata for the model input usually.
        # The Idea description says "Raw metadata (Age, Sex...)".
        # Usually clinical models use baseline characteristics to predict future.
        # We will use the columns from the row itself as "Current" features,
        # but for the physics model we need Baseline FVC.
        merged = pd.merge(df, baselines, on="Patient", how="left")
        return merged

    train_df = merge_baseline(train_df)
    val_df = merge_baseline(val_df)

    # Test df already has Baseline_ columns from metadata generation

    # 3. Fit Tabular Preprocessor
    # We fit on Training data only (using Baseline features or Current features?)
    # The prompt says "Raw metadata (Age, Sex, Smoking, Percent)".
    # Since test set only provides Baseline metadata, we should probably train on Baseline metadata
    # to be consistent with inference time where we only have baseline.
    # Let's use the Baseline columns for feature extraction to ensure consistency.

    # Map the columns to standard names for the preprocessor
    # We create a temporary DF with standard names for fitting
    fit_df = train_df[
        ["Baseline_Age", "Baseline_Sex", "Baseline_SmokingStatus", "Baseline_Percent"]
    ].copy()
    fit_df.columns = ["Age", "Sex", "SmokingStatus", "Percent"]

    tab_preprocessor = TabularPreprocessor()
    tab_preprocessor.fit(fit_df)

    # Now we need to ensure the DFs passed to Dataset have these columns named correctly
    # OR we map them inside.
    # Let's rename Baseline columns to standard names in the DFs we pass to Dataset,
    # because at inference time (Test), we only have baseline info.
    # So for Train/Val, we should also use Baseline info as the input features.

    def prepare_features(df, is_test=False):
        # We want the input features to be the Baseline characteristics
        target_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
        source_cols = [
            "Baseline_Age",
            "Baseline_Sex",
            "Baseline_SmokingStatus",
            "Baseline_Percent",
        ]

        # Create a copy to avoid SettingWithCopy warnings
        df_out = df.copy()
        for src, tgt in zip(source_cols, target_cols):
            df_out[tgt] = df_out[src]
        return df_out

    train_df = prepare_features(train_df)
    val_df = prepare_features(val_df)
    test_df = prepare_features(test_df, is_test=True)

    # 4. Define Augmentations
    # Spatial only, no intensity changes.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
    )

    # 5. Create Datasets
    train_ds = OSICDataset(
        train_df,
        tab_preprocessor,
        mode="train",
        transform=train_transform,
        load_cached_data=load_cached_data,
    )
    val_ds = OSICDataset(
        val_df,
        tab_preprocessor,
        mode="val",
        transform=val_transform,
        load_cached_data=load_cached_data,
    )
    test_ds = OSICDataset(
        test_df,
        tab_preprocessor,
        mode="test",
        transform=val_transform,
        load_cached_data=load_cached_data,
    )

    # 6. Create Loaders
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

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
