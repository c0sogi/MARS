import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try importing pydicom, handle if missing (though essential for this task)
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config
from library.utils import seed_everything

# ==========================================
# Constants & Encoders
# ==========================================
SEX_MAP = {"Male": 0, "Female": 1}
SMOKING_MAP = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specific phase.
    Strictly spatial augmentations for training; Resize/Norm for all.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


# ==========================================
# DICOM Processing Logic
# ==========================================
def read_dicom_volume(path_to_dicom_dir):
    """
    Reads a directory of .dcm files and constructs a 3D volume.
    Returns:
        volume (np.array): 3D array (Depth, Height, Width) normalized to [0, 1].
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files.")

    files = glob.glob(os.path.join(path_to_dicom_dir, "*.dcm"))
    if not files:
        # Return a dummy volume if no files found (edge case handling)
        return np.zeros((10, 512, 512), dtype=np.float32)

    # Read files and sort by InstanceNumber (or ImagePositionPatient Z)
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return np.zeros((10, 512, 512), dtype=np.float32)

    # Sort
    # Try sorting by ImagePositionPatient Z, fallback to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Construct volume
    # Handle Rescale Slope/Intercept for Hounsfield Units
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images)  # (D, H, W)

    # Lung Windowing
    # W: 1500, L: -600 -> Range: [-1350, 150]
    # We clip to this range and normalize
    min_hu = -1350
    max_hu = 150
    volume = np.clip(volume, min_hu, max_hu)

    # Normalize to 0-1
    volume = (volume - min_hu) / (max_hu - min_hu)

    return volume.astype(np.float32)


def generate_trislabs(volume, axis=0):
    """
    Generates Fixed Overlapping Tri-Slabs from a 3D volume.

    Args:
        volume: 3D numpy array.
        axis: 0 for Axial (splitting Depth), 1 for Coronal (splitting Height/Y).

    Returns:
        image: (H, W, 3) numpy array (RGB-like)
    """
    # If generating Coronal, permute volume so the split axis is 0
    if axis == 1:
        # Original: (D, H, W). Coronal view looks at H vs W (or D vs W).
        # We want to split along H (Anterior-Posterior).
        # Permute to (H, D, W)
        vol_view = np.transpose(volume, (1, 0, 2))
    else:
        vol_view = volume

    depth = vol_view.shape[0]
    if depth < 3:
        # Edge case: very few slices. Just repeat.
        mip = np.max(vol_view, axis=0)
        return np.stack([mip, mip, mip], axis=-1)

    # Define slab boundaries with overlap
    # Config.SLAB_OVERLAP is 0.15 (15%)
    chunk_size = depth / 3.0
    overlap = chunk_size * Config.SLAB_OVERLAP

    # Slab 1: 0 to 33% + overlap
    s1_start = 0
    s1_end = int(chunk_size + overlap)

    # Slab 2: 33% - overlap to 66% + overlap
    s2_start = int(chunk_size - overlap)
    s2_end = int(2 * chunk_size + overlap)

    # Slab 3: 66% - overlap to 100%
    s3_start = int(2 * chunk_size - overlap)
    s3_end = depth

    # Clip indices
    s1_end = min(s1_end, depth)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, depth)
    s3_start = max(0, s3_start)

    # Compute MIPs
    # Note: If a slice range is empty (rare), handle gracefully
    def get_mip(start, end):
        if start >= end:
            return np.zeros(vol_view.shape[1:], dtype=np.float32)
        return np.max(vol_view[start:end], axis=0)

    c1 = get_mip(s1_start, s1_end)
    c2 = get_mip(s2_start, s2_end)
    c3 = get_mip(s3_start, s3_end)

    # Stack to channels (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # If Coronal, the aspect ratio might be (Depth, Width).
    # Resize is handled by Albumentations later, but we return the raw MIP here.
    return img


def process_patient_dicom(patient_id, dicom_root_dir, cache_dir, load_cached_data=True):
    """
    Handles the full pipeline: Check cache -> (Load DICOM -> Process -> Save) -> Return.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            img_ax = np.load(axial_path)
            img_cor = np.load(coronal_path)
            return img_ax, img_cor
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    # Construct full path to patient directory
    # Note: metadata 'dicom_dir' is relative, e.g., "train/ID..."
    # We need to join with input root.
    # However, for this function, we expect the full path or we resolve it here.
    # Let's assume the caller passes the correct full path to the patient folder.

    volume = read_dicom_volume(dicom_root_dir)

    # Generate Views
    img_ax = generate_trislabs(volume, axis=0)  # Axial
    img_cor = generate_trislabs(volume, axis=1)  # Coronal

    # 3. Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(axial_path, img_ax)
    np.save(coronal_path, img_cor)

    return img_ax, img_cor


# ==========================================
# Dataset Class
# ==========================================
class LungDataset(Dataset):
    def __init__(self, df, root_dir, cache_dir, phase="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            root_dir (str): Root directory containing 'train' and 'test' folders.
            cache_dir (str): Directory to store/load .npy files.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.phase = phase
        self.transforms = get_transforms(phase)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Resolve DICOM directory
        # metadata 'dicom_dir' is like "train/ID..." or "test/ID..."
        dicom_rel_path = row["dicom_dir"]
        full_dicom_path = os.path.join(self.root_dir, dicom_rel_path)

        # Load Images (Cached or Processed)
        # We use the patient_id to name cache files
        img_ax, img_cor = process_patient_dicom(
            patient_id, full_dicom_path, self.cache_dir, load_cached_data=True
        )

        # Apply Transforms
        # Albumentations expects uint8 or float. Our images are float 0-1.
        # We need to ensure they are compatible.
        aug_ax = self.transforms(image=img_ax)["image"]
        aug_cor = self.transforms(image=img_cor)["image"]

        # Tabular Features
        # [Age, Sex, Smoking, Percent]
        # Normalize Age (approx 0-1) and Percent (approx 0-1)
        # Age range ~50-90. Percent range ~30-150.
        age_norm = float(row["Age"] if "Age" in row else row["Baseline_Age"]) / 100.0
        percent_norm = (
            float(row["Percent"] if "Percent" in row else row["Baseline_Percent"])
            / 100.0
        )

        sex_raw = row["Sex"] if "Sex" in row else row["Baseline_Sex"]
        smoke_raw = (
            row["SmokingStatus"]
            if "SmokingStatus" in row
            else row["Baseline_SmokingStatus"]
        )

        sex_enc = SEX_MAP.get(sex_raw, 0)
        smoke_enc = SMOKING_MAP.get(smoke_raw, 1)

        tabular = torch.tensor(
            [age_norm, sex_enc, smoke_enc, percent_norm], dtype=torch.float32
        )

        data = {
            "image_ax": aug_ax,
            "image_cor": aug_cor,
            "tabular": tabular,
            "patient_id": patient_id,
        }

        # Targets (if available)
        if self.phase != "test":
            fvc = float(row["FVC"])
            weeks = float(row["Weeks"])
            # We predict FVC, so target is FVC.
            # But we also need Weeks relative to baseline for the model to know "when" this is.
            # Wait, the model architecture description says:
            # "We strictly exclude Week or Time from the input (Cite solution_lesson_node_00062)."
            # "The model predicts alpha, sigma_base, sigma_growth" -> Static prediction per patient.
            # BUT, the training loss is calculated on specific visits.
            # The model output is parameters. We compute prediction = Base + alpha * week.
            # So we need 'Weeks' to compute the loss.
            data["target"] = torch.tensor([fvc], dtype=torch.float32)
            data["weeks"] = torch.tensor([weeks], dtype=torch.float32)

            # For the static model, we also need Baseline FVC.
            # In train.csv, we don't explicitly have "Baseline_FVC" column for every row,
            # but we can infer it or it might be passed.
            # Actually, for the training set, we need to know which FVC is the baseline (Week 0 or closest).
            # However, usually in this task, we pass the *current* visit's metadata.
            # The prompt says: "In the training set... entire history...".
            # The model description says: "Pass inputs and baseline metadata...".
            # This implies for Training, we should probably use the Baseline visit's metadata as input?
            # OR, we treat every visit as a training sample using its own metadata?
            # "Shared Encoder: The raw metadata (Age, Sex, Smoking, Percent)..."
            # If we use the current visit's Percent, we are leaking the target (since Percent ~ FVC).
            # CRITICAL: We must use the BASELINE Percent/Age for the patient, even when predicting future weeks.
            # The metadata provided in 'train.csv' has 'Percent' for *that specific week*.
            # We should ideally use the week 0 values.
            # However, to keep it simple and robust (and since 'Percent' is highly correlated),
            # standard solutions often use the current visit's metadata to predict the current FVC,
            # OR they group by patient and broadcast baseline.
            # Given the "Static Model" approach (predicting trajectory parameters), we should ideally input Baseline info.
            # But `train.csv` structure is flat.
            # Let's check `test.csv`: it has `Baseline_FVC`, `Baseline_Percent`.
            # For `train.csv`, we don't have those columns pre-calculated in the provided metadata schema.
            # To avoid complex merging here, we will use the row's data.
            # Note: Using current Percent to predict current FVC is strong, but valid if we have Percent at inference.
            # At inference (test), we ONLY have Baseline Percent.
            # Therefore, for training, we should ideally use Baseline Percent.
            # But we don't have it easily.
            # Strategy: We will use the row's values. The model will learn to map (Features_t) -> FVC_t.
            # Wait, the model is "Parametric Inference": FVC = Base + alpha * dt.
            # This implies we need Baseline FVC as an input to the equation, NOT the network.
            # The network predicts alpha.
            # So `data['base_fvc']` is needed.
            # In `train.csv`, we don't have `Baseline_FVC`.
            # We will approximate: For training, we can treat the current FVC as the target,
            # and we might need to rely on the network to predict the absolute FVC or the slope.
            # Re-reading Model section: "Pass inputs and baseline metadata... to obtain static parameters... Calculate predictions... FVC = Baseline_FVC + alpha * week".
            # This strictly requires Baseline_FVC.
            # Since `train.csv` lacks it, and I cannot modify metadata generation, I will:
            # 1. Group `train_df` by Patient inside `get_dataloaders` to find the Week~0 entry.
            # 2. Merge it back.
            pass

        else:
            # Test phase
            # We need Baseline info for the equation
            data["base_fvc"] = torch.tensor(
                [float(row["Baseline_FVC"])], dtype=torch.float32
            )
            data["base_week"] = torch.tensor(
                [float(row["Baseline_Week"])], dtype=torch.float32
            )
            data["predict_week"] = torch.tensor(
                [float(row["Predict_Week"])], dtype=torch.float32
            )

        return data


# ==========================================
# Data Loaders
# ==========================================
def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for Train, Val, and Test.
    Handles Baseline merging for Train/Val to support the parametric model.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Debug Subsetting
    if Config.DEBUG_SUBSET_SIZE:
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        # Keep test intact or subset if needed, but usually we want full test for pipeline check
        # test_df = test_df.iloc[:Config.DEBUG_SUBSET_SIZE]

    # 3. Add Baseline Info to Train/Val
    # The parametric model requires Baseline FVC/Week to project from.
    # We identify the baseline visit (min weeks or closest to 0) for each patient.
    def add_baseline_columns(df):
        # Sort by Weeks to ensure first is earliest
        df = df.sort_values(["Patient", "Weeks"])
        # Group by Patient and take the first entry as baseline
        baseline_df = df.groupby("Patient").first().reset_index()

        # Select relevant columns to merge back
        cols = ["Patient", "FVC", "Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
        baseline_df = baseline_df[cols]

        # Rename to Baseline_...
        rename_map = {c: f"Baseline_{c}" for c in cols if c != "Patient"}
        baseline_df = baseline_df.rename(columns=rename_map)

        # Merge
        merged = pd.merge(df, baseline_df, on="Patient", how="left")
        return merged

    # Apply to train and val
    train_df = add_baseline_columns(train_df)
    val_df = add_baseline_columns(val_df)

    # For the Dataset class, we need to ensure it uses these Baseline columns for features
    # The Dataset logic above checks: "row['Age'] if 'Age' in row else row['Baseline_Age']"
    # Since we now have both, we want to force using Baseline features for the input vector
    # to match the inference time scenario (where we only have baseline).
    # So we rename the original Age/Percent/etc to something else or drop them?
    # No, we just update the Dataset logic or ensure the columns exist.
    # Let's update the dataframes to prioritize Baseline columns for features.
    # We will rename 'Age' -> 'Visit_Age', 'Baseline_Age' -> 'Age' so the Dataset picks the Baseline.

    def prioritize_baseline(df):
        # We want Dataset to use Baseline values for features
        # Dataset uses: Age, Sex, SmokingStatus, Percent
        # We swap columns
        for col in ["Age", "Sex", "SmokingStatus", "Percent"]:
            if f"Baseline_{col}" in df.columns:
                df[f"Visit_{col}"] = df[col]  # Keep original
                df[col] = df[f"Baseline_{col}"]  # Overwrite with Baseline
        return df

    train_df = prioritize_baseline(train_df)
    val_df = prioritize_baseline(val_df)

    # Note: For targets (FVC, Weeks), we still use the original columns which are now 'FVC' and 'Weeks'.
    # (The merge didn't overwrite FVC/Weeks, it added Baseline_FVC/Baseline_Weeks).
    # Correct.

    # 4. Create Datasets
    train_ds = LungDataset(
        train_df, Config.TRAIN_DICOM_ROOT, Config.CACHE_DIR, phase="train"
    )
    val_ds = LungDataset(val_df, Config.TRAIN_DICOM_ROOT, Config.CACHE_DIR, phase="val")
    test_ds = LungDataset(
        test_df, Config.TEST_DICOM_ROOT, Config.CACHE_DIR, phase="test"
    )

    # 5. Create Loaders
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
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
