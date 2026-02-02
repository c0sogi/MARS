import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from library.config import Config
from library.utils import seed_everything

# Constants for Tabular Normalization (derived from EDA)
AGE_MEAN = 67.58
AGE_STD = 6.62


class LungDataset(Dataset):
    """
    Dataset class for loading Lung CT scans and Clinical Data.
    Implements caching for processed images to speed up training.
    """

    def __init__(
        self, df, mode="train", cache_dir=Config.CACHE_DIR, load_cached_data=True
    ):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data

        # Encoders
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image Processing (Cached)
        # Returns tensor of shape (3, 260, 260)
        img_tensor = self._load_or_process_image(patient_id, row.get("image_path"))

        # 2. Tabular Features
        # Stream A Input: Baseline FVC, Relative Time, Age, Sex, SmokingStatus

        # Baseline FVC (Standardized)
        base_fvc = (row["Baseline_FVC"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Relative Time (Scaled)
        # t_rel = (Current_Week - Baseline_Week) * 0.01
        t_rel = (row["Weeks"] - row["Baseline_Week"]) * 0.01

        # Age (Standardized)
        age = (row["Age"] - AGE_MEAN) / AGE_STD

        # Categorical Encoding
        sex = self.sex_map.get(row["Sex"], 0)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        # Construct Tabular Vector: [Baseline_FVC, t_rel, Age, Sex, Smoking]
        tab_vector = np.array([base_fvc, t_rel, age, sex, smoke], dtype=np.float32)

        # 3. Target
        if self.mode != "submission":
            target_raw = row["FVC"]
            # Standardize Target using Global Statistics
            target = (target_raw - Config.TARGET_MEAN) / Config.TARGET_STD
            target = np.array([target], dtype=np.float32)
        else:
            # Dummy target for submission
            target = np.array([0.0], dtype=np.float32)

        return (
            torch.tensor(img_tensor, dtype=torch.float32),
            torch.tensor(tab_vector, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )

    def _load_or_process_image(self, patient_id, rel_path):
        """
        Loads processed image from cache if available, otherwise processes DICOMs and caches result.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # 1. Try Loading from Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from Scratch
        if rel_path is None:
            # Fallback logic if path is missing (should not occur with correct metadata)
            if os.path.exists(os.path.join(Config.INPUT_DIR, "train", patient_id)):
                full_dir = os.path.join(Config.INPUT_DIR, "train", patient_id)
            else:
                full_dir = os.path.join(Config.INPUT_DIR, "test", patient_id)
        else:
            full_dir = os.path.join(Config.INPUT_DIR, rel_path)

        img_array = self._process_dicom_directory(full_dir)

        # 3. Save to Cache
        np.save(cache_path, img_array)

        return img_array

    def _process_dicom_directory(self, directory):
        """
        Reads DICOMs, applies windowing, selects slices, and resizes.
        """
        # List files
        files = glob.glob(os.path.join(directory, "*.dcm"))
        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read DICOMs
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                # Ensure necessary attributes exist
                if hasattr(ds, "InstanceNumber") and hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except:
                continue

        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Sort by Instance Number to maintain Z-axis order
        slices.sort(key=lambda x: int(x.InstanceNumber))

        # Extract Pixel Arrays and apply Rescale Slope/Intercept (HU conversion)
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = img * slope + intercept
            images.append(img)

        images = np.array(images)  # Shape: (Depth, Height, Width)

        # Content-Adaptive Slice Selection
        # Calculate Lung Area (approx HU range -1000 to -200)
        lung_mask = (images > -1000) & (images < -200)
        lung_areas = lung_mask.sum(axis=(1, 2))

        if lung_areas.max() == 0:
            # Fallback: take middle slices if no lung detected
            selected_indices = [0, len(images) // 2, len(images) - 1]
        else:
            max_area = lung_areas.max()
            idx_max = np.argmax(lung_areas)  # Anchor slice

            # Find boundaries (> 50% of max area)
            candidates = np.where(lung_areas > 0.5 * max_area)[0]

            if len(candidates) > 0:
                idx_min = candidates[0]
                idx_last = candidates[-1]
            else:
                idx_min = idx_max
                idx_last = idx_max

            # Select 3 slices: Top Boundary, Anchor, Bottom Boundary
            # Sort to maintain anatomical order
            selected_indices = sorted(list(set([idx_min, idx_max, idx_last])))

            # Pad if fewer than 3 unique slices found
            while len(selected_indices) < 3:
                selected_indices.append(selected_indices[-1])

            # Ensure exactly 3
            selected_indices = selected_indices[:3]

        # Extract selected slices
        selected_imgs = images[selected_indices]

        # Windowing (Lung Window: Level -600, Width 1500)
        # Range: [-1350, 150]
        L, W = Config.WINDOW_LEVEL, Config.WINDOW_WIDTH
        lower, upper = L - W // 2, L + W // 2

        processed_slices = []
        for i in range(len(selected_imgs)):
            img = selected_imgs[i]
            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)  # Normalize to [0, 1]

            # Resize to model input size
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
            processed_slices.append(img)

        # Stack -> (3, H, W)
        tensor = np.stack(processed_slices, axis=0)
        return tensor.astype(np.float32)


def prepare_train_dataframe(df):
    """
    Identifies the Baseline FVC and Baseline Week for each patient in the training set.
    Assumes the visit with the minimum 'Weeks' value is the baseline.
    """
    # Find the row index with min Weeks for each patient
    baseline_indices = df.groupby("Patient")["Weeks"].idxmin()
    baseline_df = df.loc[baseline_indices][["Patient", "FVC", "Weeks"]]
    baseline_df.columns = ["Patient", "Baseline_FVC", "Baseline_Week"]

    # Merge baseline info back to the main dataframe
    df = df.merge(baseline_df, on="Patient", how="left")
    return df


def prepare_submission_dataframe(sample_sub_path, test_meta_path):
    """
    Prepares the dataframe for submission by merging sample_submission with test metadata.
    """
    sub_df = pd.read_csv(sample_sub_path)
    test_meta = pd.read_csv(test_meta_path)

    # Extract Patient and Weeks from Patient_Week column (e.g., ID..._12)
    # Use rsplit to handle potential underscores in IDs safely
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.rsplit("_", 1)[1]))

    # Rename columns in test_meta to represent Baseline info
    # test.csv contains the initial measurement
    test_meta = test_meta.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge static features and baseline info
    merged = sub_df.merge(test_meta, on="Patient", how="left")
    return merged


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
    load_cached_data=True,
):
    """
    Creates and returns DataLoaders for Train, Val, and Test (Submission) sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Prepare Train/Val Dataframes (Add Baseline Info)
    train_df = prepare_train_dataframe(train_df)
    val_df = prepare_train_dataframe(val_df)

    # Prepare Submission DF
    sub_df = prepare_submission_dataframe(Config.SAMPLE_SUBMISSION, Config.TEST_CSV)

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        sub_df = sub_df.head(Config.DEBUG_SIZE)

    # Instantiate Datasets
    train_ds = LungDataset(train_df, mode="train", load_cached_data=load_cached_data)
    val_ds = LungDataset(val_df, mode="val", load_cached_data=load_cached_data)
    test_ds = LungDataset(sub_df, mode="submission", load_cached_data=load_cached_data)

    # Create DataLoaders
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
