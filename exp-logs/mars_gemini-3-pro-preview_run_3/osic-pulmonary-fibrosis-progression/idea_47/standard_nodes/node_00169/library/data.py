import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from library.config import Config

# ====================================================
# Image Processing & Caching
# ====================================================


def get_img(patient_id, image_dir):
    """
    Loads, preprocesses, and caches the CT scan for a given patient.
    Selects 3 slices: Top Boundary, Anchor (Max Area), Bottom Boundary.

    Args:
        patient_id (str): The patient ID.
        image_dir (str): Path to the directory containing DICOM files.

    Returns:
        np.ndarray: Processed image tensor of shape (3, H, W) with values in [0, 1].
    """
    # 1. Check Cache
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Load DICOMs
    # Handle case where directory might not exist (though metadata validation passed)
    if not os.path.exists(image_dir):
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    files = [f for f in os.listdir(image_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Load and sort slices by InstanceNumber (Z-position)
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(image_dir, f))
            # Ensure InstanceNumber exists, default to 0 if missing
            in_num = (
                int(ds.InstanceNumber)
                if hasattr(ds, "InstanceNumber") and ds.InstanceNumber
                else 0
            )
            slices.append((in_num, ds))
        except:
            continue

    if not slices:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    slices.sort(key=lambda x: x[0])
    sorted_slices = [s[1] for s in slices]

    # 3. Process Slices (HU conversion + Windowing)
    processed_slices = []
    slice_areas = []

    for s in sorted_slices:
        # Convert to HU
        try:
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = s.pixel_array.astype(np.float32) * slope + intercept
        except Exception:
            continue

        # Calculate Lung Area for selection (Threshold approx -1000 to -200 HU)
        # We use a simple count of pixels in the lung density range
        area = np.sum((img > -1000) & (img < -200))
        slice_areas.append(area)

        # Apply Lung Windowing
        img_min = Config.WINDOW_MIN
        img_max = Config.WINDOW_MAX
        img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, 0, 1)

        # Resize
        if img.shape[0] != Config.IMAGE_SIZE or img.shape[1] != Config.IMAGE_SIZE:
            img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))

        processed_slices.append(img)

    # 4. Slice Selection
    num_slices = len(processed_slices)
    slice_areas = np.array(slice_areas)

    if num_slices == 0:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    max_area_idx = np.argmax(slice_areas)
    max_area = slice_areas[max_area_idx]

    # Identify candidate slices (Area > 50% of max)
    candidates = np.where(slice_areas > 0.5 * max_area)[0]

    if len(candidates) < 3:
        # Fallback: take center and +/- stride if candidates are too few
        indices = [max_area_idx]
        if num_slices > 1:
            indices.append(max(0, max_area_idx - 1))
        if num_slices > 2:
            indices.append(min(num_slices - 1, max_area_idx + 1))
        # Pad if still not enough by duplicating anchor
        while len(indices) < 3:
            indices.append(max_area_idx)
    else:
        # Select: Top Boundary (first candidate), Anchor (max), Bottom Boundary (last candidate)
        idx1 = candidates[0]
        idx2 = max_area_idx
        idx3 = candidates[-1]
        indices = [idx1, idx2, idx3]

    # Sort indices to maintain anatomical order (Top -> Bottom) in channels
    indices.sort()

    # Stack to (3, H, W)
    final_volume = np.stack([processed_slices[i] for i in indices], axis=0)

    # 5. Save to Cache
    try:
        np.save(cache_path, final_volume)
    except Exception:
        pass

    return final_volume


# ====================================================
# Dataset Class
# ====================================================


class OSICDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # Use image_path from metadata (relative path)
        # If 'image_path' is missing (e.g. constructed test set), reconstruct it
        if "image_path" in row and pd.notna(row["image_path"]):
            rel_path = row["image_path"]
        else:
            # Fallback for test set construction if needed
            folder = "test" if self.mode == "test" else "train"
            rel_path = os.path.join(folder, patient_id)

        img_dir = os.path.join(Config.INPUT_DIR, rel_path)
        img = get_img(patient_id, img_dir)
        img = torch.tensor(img, dtype=torch.float32)

        # 2. Tabular Features
        # Vector: [Baseline_FVC_Scaled, Relative_Time_Scaled, Age_Scaled, Sex_Code, Smoking_Code]
        tabular = torch.tensor(
            [
                row["Baseline_FVC_Scaled"],
                row["Relative_Week_Scaled"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
            ],
            dtype=torch.float32,
        )

        # 3. Target
        if self.mode != "test":
            target = torch.tensor([row["FVC_Scaled"]], dtype=torch.float32)
            return img, tabular, target
        else:
            return img, tabular


# ====================================================
# Data Preprocessing & Loading
# ====================================================


def preprocess_metadata(df, train_stats=None):
    """
    Augments the dataframe with Baseline info and scales features.

    Args:
        df (pd.DataFrame): Dataframe to process.
        train_stats (dict): Stats from training set. If None, calculated from df.

    Returns:
        pd.DataFrame, dict
    """
    df = df.copy()

    # 1. Identify Baseline for each Patient
    # We assume the dataframe contains history. We sort by Weeks.
    # Baseline is the first visit.
    df = df.sort_values(["Patient", "Weeks"])

    # Broadcast baseline values
    df["Baseline_Week"] = df.groupby("Patient")["Weeks"].transform("first")
    df["Baseline_FVC"] = df.groupby("Patient")["FVC"].transform("first")

    # Calculate Relative Week
    df["Relative_Week"] = df["Weeks"] - df["Baseline_Week"]

    # 2. Encoding
    # Sex: Male=0, Female=1
    df["Sex_Code"] = df["Sex"].map({"Male": 0, "Female": 1}).fillna(0)

    # Smoking: Never=0, Ex=1, Current=2
    smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
    df["Smoking_Code"] = df["SmokingStatus"].map(smoking_map).fillna(1)

    # 3. Scaling
    if train_stats is None:
        stats = {}
        # Use Config to get global FVC stats (ensures consistency with InverseScaler)
        fvc_mean, fvc_std = Config.get_target_stats(load_cached_data=True)
        stats["fvc_mean"] = fvc_mean
        stats["fvc_std"] = fvc_std

        stats["base_fvc_mean"] = df["Baseline_FVC"].mean()
        stats["base_fvc_std"] = df["Baseline_FVC"].std()
        stats["age_mean"] = df["Age"].mean()
        stats["age_std"] = df["Age"].std()
    else:
        stats = train_stats

    # Apply Scaling
    df["Baseline_FVC_Scaled"] = (df["Baseline_FVC"] - stats["base_fvc_mean"]) / stats[
        "base_fvc_std"
    ]
    # Relative Week scaled by 0.01 (no Z-score)
    df["Relative_Week_Scaled"] = df["Relative_Week"] * 0.01
    df["Age_Scaled"] = (df["Age"] - stats["age_mean"]) / stats["age_std"]

    # Target Scaling
    df["FVC_Scaled"] = (df["FVC"] - stats["fvc_mean"]) / stats["fvc_std"]

    return df, stats


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    if debug:
        train_df = train_df.iloc[:64]
        val_df = val_df.iloc[:32]
        # Keep test intact as it's small

    # 2. Preprocess Train/Val
    train_df, stats = preprocess_metadata(train_df, train_stats=None)
    val_df, _ = preprocess_metadata(val_df, train_stats=stats)

    # 3. Construct Test Set
    # We need to predict for every Patient_Week in sample_submission.csv
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Extract Patient and Target Week
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    # Prepare test metadata (Baseline info)
    # Rename columns to indicate they are baseline values
    test_static = test_df.rename(
        columns={"Weeks": "Baseline_Week", "FVC": "Baseline_FVC"}
    )

    # Merge: sample_sub (Target Weeks) + test_static (Baseline Info & Static features)
    test_merged = sample_sub.merge(test_static, on="Patient", how="left")

    # Calculate Features for Test
    test_merged["Relative_Week"] = test_merged["Weeks"] - test_merged["Baseline_Week"]

    # Encode & Scale Test
    test_merged["Sex_Code"] = test_merged["Sex"].map({"Male": 0, "Female": 1}).fillna(0)
    smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
    test_merged["Smoking_Code"] = (
        test_merged["SmokingStatus"].map(smoking_map).fillna(1)
    )

    test_merged["Baseline_FVC_Scaled"] = (
        test_merged["Baseline_FVC"] - stats["base_fvc_mean"]
    ) / stats["base_fvc_std"]
    test_merged["Relative_Week_Scaled"] = test_merged["Relative_Week"] * 0.01
    test_merged["Age_Scaled"] = (test_merged["Age"] - stats["age_mean"]) / stats[
        "age_std"
    ]

    # 4. Create Datasets
    train_dataset = OSICDataset(train_df, mode="train")
    val_dataset = OSICDataset(val_df, mode="val")
    test_dataset = OSICDataset(test_merged, mode="test")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
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

    return train_loader, val_loader, test_loader
