import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from library.config import Config

# -------------------------------------------------------------------------
# Helper Functions: Image Processing & Caching
# -------------------------------------------------------------------------


def get_img_seq(patient_id, data_dir):
    """
    Locates the directory containing DICOM files for a given patient.
    """
    # The path structure is typically data_dir/PatientID/*.dcm
    # data_dir provided by Config is usually input/train or input/test
    # But the metadata dataframe contains 'image_path' which is relative.
    # We will rely on the dataframe's image_path if possible, but the dataset
    # might pass the root dir.

    # Construct path based on Config structure
    path = os.path.join(data_dir, patient_id)
    if not os.path.exists(path):
        # Fallback: try to find it in the parent directory if data_dir was specific
        # This handles cases where data_dir might be 'input/train' vs just 'input'
        pass
    return path


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber.
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

    # Sort by InstanceNumber (Z-position)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    """
    image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Convert to HU
    for i, s in enumerate(slices):
        intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
        slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.float32)

        image[i] += np.float32(intercept)

    return image


def process_adaptive_slices(patient_id, dataset_dir):
    """
    Selects 3 content-adaptive slices (Top, Max-Area, Bottom) and normalizes them.
    Returns a (H, W, 3) numpy array.
    """
    path = os.path.join(dataset_dir, patient_id)
    slices = load_scan(path)

    if not slices:
        # Fallback for missing data: return black image
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Get HU volume
    try:
        volume = get_pixels_hu(slices)
    except Exception:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Calculate Lung Area for each slice
    # Lung window approx: -1000 to -400 usually.
    # We threshold < -300 to capture lung tissue + air, excluding bone/tissue > -300
    # and exclude background air (usually -1000, but in HU it matches lung).
    # A simple threshold < -320 is often used for lung segmentation.
    # To avoid background noise, we can also check > -2000.

    areas = []
    for i in range(volume.shape[0]):
        # Simple count of pixels in lung range
        mask = (volume[i] > -1000) & (volume[i] < -320)
        areas.append(np.sum(mask))

    areas = np.array(areas)

    if np.max(areas) == 0:
        # No lung detected, pick middle slices
        mid = len(slices) // 2
        indices = [max(0, mid - 1), mid, min(len(slices) - 1, mid + 1)]
    else:
        max_area = np.max(areas)
        idx_max = np.argmax(areas)

        # Find valid range where area > 50% of max
        valid_indices = np.where(areas > 0.5 * max_area)[0]

        if len(valid_indices) > 0:
            idx_top = valid_indices[0]  # Top of lung (lowest index)
            idx_bottom = valid_indices[-1]  # Bottom of lung (highest index)
        else:
            idx_top = idx_max
            idx_bottom = idx_max

        indices = [idx_top, idx_max, idx_bottom]

    # Extract, Resize, Window, Normalize
    selected_slices = []
    for idx in indices:
        img = volume[idx]

        # Lung Window: L=-600, W=1500 -> [-1350, 150]
        # Clip to window
        img = np.clip(img, -1350, 150)

        # Normalize to [0, 1]
        img = (img - (-1350)) / (150 - (-1350))

        # Resize
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        selected_slices.append(img)

    # Stack to (H, W, 3)
    img_stack = np.stack(selected_slices, axis=-1)
    return img_stack.astype(np.float32)


def load_or_process_image(patient_id, dataset_dir, cache=True):
    """
    Caching wrapper for image processing.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    if cache and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Failed to load, reprocess

    # Process
    img = process_adaptive_slices(patient_id, dataset_dir)

    # Save
    if cache:
        np.save(cache_path, img)

    return img


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class FVCDataset(Dataset):
    def __init__(self, df, dataset_dir, mode="train", cache_images=True):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            dataset_dir (str): Directory containing patient DICOM folders (e.g., input/train).
            mode (str): 'train', 'val', or 'submission'.
            cache_images (bool): Whether to use disk caching for images.
        """
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.cache_images = cache_images

        # Pre-process One-Hot Encoding for Categorical Features
        # We need to ensure consistent columns.
        # Sex: Male, Female
        # SmokingStatus: Ex-smoker, Never smoked, Currently smokes

        # Create mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # Note: Test set images are in input/test, Train in input/train.
        # The dataset_dir passed to __init__ handles this distinction.
        img = load_or_process_image(
            patient_id, self.dataset_dir, cache=self.cache_images
        )

        # Transpose to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # 2. Clinical Features
        # Extract raw values
        base_fvc = row["Base_FVC"]
        percent = row["Percent"]  # Note: Using current percent, or baseline?
        # Prompt says "Stream A: Baseline FVC, Percent..."
        # Usually 'Percent' in the table is for the current measurement.
        # However, for consistency with 'Baseline FVC', we often use Baseline Percent.
        # But the row has 'Percent'. Let's use the row's Percent if available,
        # or Base_Percent if we are in submission mode where we only have baseline data.
        # In submission mode, the row comes from test.csv (baseline), so it is baseline percent.
        # In train mode, the row is a visit.
        # Let's stick to the prompt: "Stream A Input: Baseline FVC, Percent...".
        # I will use the Percent from the dataframe row (which is time-varying in train, static in test).

        age = row["Age"]

        # Relative Time
        if self.mode == "submission":
            # In submission, 'Weeks' is the target prediction week
            # 'Base_Week' is the week of the initial CT
            weeks = row["Weeks"]
            base_week = row["Base_Week"]
            rel_time = (weeks - base_week) * Config.TIME_SCALE
        else:
            # In train/val, 'Weeks' is the visit week
            weeks = row["Weeks"]
            base_week = row["Base_Week"]
            rel_time = (weeks - base_week) * Config.TIME_SCALE

        # Normalization (Z-score)
        base_fvc_norm = (base_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
        percent_norm = (percent - Config.PERCENT_MEAN) / Config.PERCENT_STD
        age_norm = (age - Config.AGE_MEAN) / Config.AGE_STD

        # Categorical (One-Hot)
        sex_val = self.sex_map.get(row["Sex"], 0)
        smoke_val = self.smoke_map.get(
            row["SmokingStatus"], 1
        )  # Default to Never if missing

        # Create One-Hot Vectors
        sex_oh = [0, 0]
        sex_oh[sex_val] = 1

        smoke_oh = [0, 0, 0]
        smoke_oh[smoke_val] = 1

        # Construct Feature Vector
        # [Base_FVC, Percent, Age, Rel_Time, Sex_M, Sex_F, Smoke_Ex, Smoke_Nev, Smoke_Cur]
        clin_features = (
            [base_fvc_norm, percent_norm, age_norm, rel_time] + sex_oh + smoke_oh
        )
        clin_tensor = torch.tensor(clin_features, dtype=torch.float32)

        # 3. Target
        if self.mode != "submission":
            target_fvc = row["FVC"]
            # Standardize target
            target_norm = (target_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
            target_tensor = torch.tensor([target_norm], dtype=torch.float32)
            return img_tensor, clin_tensor, target_tensor
        else:
            # Return dummy target for submission
            return img_tensor, clin_tensor, torch.tensor([0.0], dtype=torch.float32)


# -------------------------------------------------------------------------
# Data Loader Construction
# -------------------------------------------------------------------------


def get_baseline_dict(df):
    """
    Creates a dictionary of baseline information for each patient.
    Baseline is defined as the visit with the minimum 'Weeks' value.
    """
    baseline_dict = {}
    patients = df["Patient"].unique()

    for p in patients:
        p_data = df[df["Patient"] == p].sort_values("Weeks")
        baseline = p_data.iloc[0]
        baseline_dict[p] = {
            "Base_FVC": baseline["FVC"],
            "Base_Week": baseline["Weeks"],
            "Base_Percent": baseline["Percent"],
            "Base_Age": baseline["Age"],
        }
    return baseline_dict


def prepare_dataframe(df, baseline_dict=None):
    """
    Augments the dataframe with baseline columns.
    If baseline_dict is None, it calculates it from the df itself (train mode).
    If provided (test mode), it uses it.
    """
    if baseline_dict is None:
        baseline_dict = get_baseline_dict(df)

    # Map baseline values to the dataframe
    df["Base_FVC"] = df["Patient"].map(lambda x: baseline_dict[x]["Base_FVC"])
    df["Base_Week"] = df["Patient"].map(lambda x: baseline_dict[x]["Base_Week"])

    # Note: Age and Sex/Smoking are usually static per patient, so they exist in row.
    # But Base_FVC is specific to the first visit.
    return df


def get_dataloaders(debug=False):
    """
    Constructs DataLoaders for Train, Val, and Submission.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)

    # 2. Prepare Train/Val Dataframes (Calculate Baselines internally)
    # We combine them to calculate baselines to ensure robustness,
    # but strictly speaking, train baselines come from train, val from val.
    # Since val patients are disjoint from train, this is fine.
    train_df = prepare_dataframe(train_df)
    val_df = prepare_dataframe(val_df)

    # 3. Prepare Submission Dataframe
    # The submission file has Patient_Week (e.g., ID_12).
    # We need to explode this into Patient, Weeks, and attach baseline info from test_meta_df.

    # Parse Patient and Week
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    # Create baseline dict from test_meta_df (which only has baseline rows)
    test_baseline_dict = get_baseline_dict(test_meta_df)

    # Merge static features (Age, Sex, Smoking, Percent) from test_meta_df
    # We do this by merging sample_sub with test_meta_df on Patient
    # Note: test_meta_df has columns [Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus]
    # We rename FVC to Base_FVC for clarity before merge, or handle in prepare

    # Let's just map everything for simplicity and safety
    cols_to_map = ["Age", "Sex", "SmokingStatus", "Percent"]
    # Note: Percent in test.csv is the baseline percent.

    for col in cols_to_map:
        # Create a lookup
        lookup = test_meta_df.set_index("Patient")[col].to_dict()
        sample_sub[col] = sample_sub["Patient"].map(lookup)

    # Attach Base_FVC and Base_Week
    sample_sub = prepare_dataframe(sample_sub, test_baseline_dict)

    # 4. Create Datasets
    # Train/Val images are in input/train
    train_dataset = FVCDataset(train_df, Config.TRAIN_DIR, mode="train")
    val_dataset = FVCDataset(val_df, Config.TRAIN_DIR, mode="val")

    # Test images are in input/test
    sub_dataset = FVCDataset(sample_sub, Config.TEST_DIR, mode="submission")

    # 5. Create Loaders
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

    sub_loader = DataLoader(
        sub_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, sub_loader, sample_sub
