import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Try importing pydicom, handle gracefully if missing (though required for this task)
try:
    import pydicom
except ImportError:
    pydicom = None


class LungDataset(Dataset):
    """
    Dataset class for Lung Function Prediction.
    Handles loading of cached image tensors and assembly of clinical features.
    """

    def __init__(self, df, image_cache_dir, stats=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.image_cache_dir = image_cache_dir
        self.mode = mode
        self.stats = stats

        # Pre-compute categorical codes
        # Sex: Male=0, Female=1
        self.df["Sex_Code"] = (
            self.df["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(np.float32)
        )

        # Smoking: Never smoked=0, Ex-smoker=1, Currently smokes=2
        self.df["Smoking_Code"] = (
            self.df["SmokingStatus"]
            .map({"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2})
            .fillna(1)
            .astype(np.float32)
        )  # Default to Ex-smoker if missing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image (Cached or Processed)
        img = self._load_patient_image(patient_id, row["image_path"])

        # 2. Prepare Tabular Features
        # Feature Vector: [BaselineFVC_norm, t_rel, Age_norm, Sex_Code, Smoking_Code]

        # Standardize Baseline FVC and Age using global stats
        base_fvc = (row["Baseline_FVC"] - self.stats["BaseFVC_mean"]) / self.stats[
            "BaseFVC_std"
        ]
        age = (row["Age"] - self.stats["Age_mean"]) / self.stats["Age_std"]
        t_rel = row["t_rel"]  # Already scaled by Config.TIME_SCALE

        tabular = np.array(
            [base_fvc, t_rel, age, row["Sex_Code"], row["Smoking_Code"]],
            dtype=np.float32,
        )

        # 3. Prepare Target (if available)
        if self.mode in ["train", "val"]:
            # Z-score standardization for target FVC
            target_raw = row["FVC"]
            target = (target_raw - self.stats["FVC_mean"]) / self.stats["FVC_std"]
            return (
                torch.tensor(img, dtype=torch.float32),
                torch.tensor(tabular, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
        else:
            # Submission mode (no target)
            return torch.tensor(img, dtype=torch.float32), torch.tensor(
                tabular, dtype=torch.float32
            )

    def _load_patient_image(self, patient_id, rel_path):
        """
        Loads processed image from cache if available, otherwise processes DICOMs and caches result.
        """
        cache_path = os.path.join(self.image_cache_dir, f"{patient_id}.npy")

        # Try loading from cache
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, recompute

        # Process from raw DICOMs
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = process_scan(full_path)

        # Save to cache
        try:
            np.save(cache_path, img)
        except Exception:
            pass  # Failed to save, but return the image anyway

        return img


def process_scan(dir_path):
    """
    Reads DICOM files, applies lung windowing, and selects 3 adaptive slices.
    Returns: (3, IMG_SIZE, IMG_SIZE) numpy array, normalized to [0, 1].
    """
    if pydicom is None:
        # Fallback if pydicom is missing (should not happen based on requirements)
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    try:
        if not os.path.exists(dir_path):
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".dcm")
        ]
        # Sort by instance number (filename)
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        slices = []
        for f in files:
            try:
                d = pydicom.dcmread(f)
                img = d.pixel_array.astype(np.float32)

                # Convert to Hounsfield Units (HU)
                intercept = getattr(d, "RescaleIntercept", -1024)
                slope = getattr(d, "RescaleSlope", 1)
                img = img * slope + intercept

                # Apply Lung Window: Width 1500, Level -600
                wl, ww = -600, 1500
                upper, lower = wl + ww // 2, wl - ww // 2
                img = np.clip(img, lower, upper)

                # Normalize to [0, 1]
                img = (img - lower) / (upper - lower)

                slices.append(img)
            except Exception:
                continue

        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Adaptive Slice Selection
        # Heuristic: Calculate "lung area" as pixels in [0.05, 0.95] range of windowed image
        areas = []
        for s in slices:
            mask = (s > 0.05) & (s < 0.95)
            areas.append(np.sum(mask))

        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        # Find boundary slices (50% of max area)
        upper_idx = max_idx
        for i in range(max_idx, len(slices)):
            if areas[i] < 0.5 * max_area:
                upper_idx = i
                break

        lower_idx = max_idx
        for i in range(max_idx, -1, -1):
            if areas[i] < 0.5 * max_area:
                lower_idx = i
                break

        selected_indices = sorted([lower_idx, max_idx, upper_idx])

        final_slices = []
        for idx in selected_indices:
            s = slices[idx]
            # Resize to Config.IMG_SIZE
            if s.shape != (Config.IMG_SIZE, Config.IMG_SIZE):
                s = cv2.resize(
                    s, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
                )
            final_slices.append(s)

        return np.stack(final_slices, axis=0)  # Shape: (3, H, W)

    except Exception as e:
        print(f"Error processing scan {dir_path}: {e}")
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def add_baseline_features(df):
    """
    Identifies the baseline visit for each patient and propagates Baseline_FVC and Baseline_Week.
    Computes relative time t_rel.
    """
    df = df.copy()

    # Identify baseline row: group by patient and take the row with min Weeks
    baseline_df = (
        df.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
    )

    # Extract baseline info
    baseline_df = baseline_df[["Patient", "Weeks", "FVC"]]
    baseline_df.columns = ["Patient", "Baseline_Week", "Baseline_FVC"]

    # Merge baseline info back to main dataframe
    df = df.merge(baseline_df, on="Patient", how="left")

    # Compute relative time
    df["t_rel"] = (df["Weeks"] - df["Baseline_Week"]) * Config.TIME_SCALE

    return df


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, val, and submission sets.
    Computes and returns normalization statistics.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 1. Preprocess Train/Val Dataframes
    train_df = add_baseline_features(train_df)
    val_df = add_baseline_features(val_df)

    # 2. Compute Normalization Statistics (from Training Set ONLY)
    stats = {
        "FVC_mean": float(train_df["FVC"].mean()),
        "FVC_std": float(train_df["FVC"].std()),
        "BaseFVC_mean": float(train_df["Baseline_FVC"].mean()),
        "BaseFVC_std": float(train_df["Baseline_FVC"].std()),
        "Age_mean": float(train_df["Age"].mean()),
        "Age_std": float(train_df["Age"].std()),
    }

    # 3. Prepare Submission Dataframe
    # Load sample submission to get target Patient_Weeks
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Parse Patient and Weeks from "Patient_Week" ID
    # ID format: ID000..._12
    split_data = sub_df["Patient_Week"].str.rsplit("_", n=1, expand=True)
    sub_df["Patient"] = split_data[0]
    sub_df["Weeks"] = split_data[1].astype(int)

    # Prepare Test Metadata for merge
    # In test.csv, 'FVC' is the Baseline FVC, and 'Weeks' is the Baseline Week
    test_meta = test_df.rename(
        columns={"Weeks": "Baseline_Week", "FVC": "Baseline_FVC"}
    )

    # Merge static features (Age, Sex, Smoking, Baseline info, image_path)
    # We drop 'Percent' as per requirements
    cols_to_merge = [
        "Patient",
        "Baseline_Week",
        "Baseline_FVC",
        "Age",
        "Sex",
        "SmokingStatus",
        "image_path",
    ]
    sub_df = sub_df.merge(test_meta[cols_to_merge], on="Patient", how="left")

    # Compute t_rel for submission
    sub_df["t_rel"] = (sub_df["Weeks"] - sub_df["Baseline_Week"]) * Config.TIME_SCALE

    # 4. Create Datasets
    train_ds = LungDataset(train_df, Config.CACHE_DIR, stats, mode="train")
    val_ds = LungDataset(val_df, Config.CACHE_DIR, stats, mode="val")
    test_ds = LungDataset(sub_df, Config.CACHE_DIR, stats, mode="submission")

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, stats
