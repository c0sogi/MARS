import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Attempt to import pydicom
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not found. Image features will be zeroed out.")


def get_img(
    patient_id: str, mode: str = "train", load_cached_data: bool = True
) -> np.ndarray:
    """
    Loads, processes, and selects the representative slice for a patient.
    Implements caching mechanism to speed up subsequent epochs.

    Args:
        patient_id (str): The unique patient identifier.
        mode (str): 'train', 'val', or 'test' to locate the correct directory.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: Processed image array of shape (H, W, 3) normalized to [0, 1].
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            img_array = np.load(cache_path)
            return img_array
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    if not HAS_PYDICOM:
        # Return black image if pydicom is missing
        img_array = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
    else:
        # Determine source directory
        if mode == "test":
            dcm_dir = os.path.join(Config.TEST_DIR, patient_id)
        else:
            # Both train and val come from the train directory structure
            dcm_dir = os.path.join(Config.TRAIN_DIR, patient_id)

        if not os.path.exists(dcm_dir):
            img_array = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
            )
        else:
            files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
            if not files:
                img_array = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                )
            else:
                # Heuristic: Find slice with max lung area
                best_slice = None
                max_lung_pixels = -1

                # Sort files to ensure deterministic order of processing
                files.sort()

                for f in files:
                    try:
                        path = os.path.join(dcm_dir, f)
                        dcm = pydicom.dcmread(path)
                        img = dcm.pixel_array.astype(np.float32)

                        # Rescale to Hounsfield Units (HU)
                        intercept = getattr(dcm, "RescaleIntercept", 0)
                        slope = getattr(dcm, "RescaleSlope", 1)
                        img = img * slope + intercept

                        # Count lung pixels in the specified HU range
                        lung_pixels = np.sum(
                            (img >= Config.LUNG_MIN_HU) & (img <= Config.LUNG_MAX_HU)
                        )

                        if lung_pixels > max_lung_pixels:
                            max_lung_pixels = lung_pixels
                            best_slice = img
                    except Exception:
                        continue

                if best_slice is None:
                    img_array = np.zeros(
                        (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32
                    )
                else:
                    # Apply Windowing
                    img = best_slice
                    img_min = Config.WINDOW_CENTER - Config.WINDOW_WIDTH // 2
                    img_max = Config.WINDOW_CENTER + Config.WINDOW_WIDTH // 2
                    img[img < img_min] = img_min
                    img[img > img_max] = img_max

                    # Normalize to [0, 1]
                    img = (img - img_min) / (img_max - img_min)

                    # Resize to target size
                    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

                    # Stack to 3 channels (for CNN backbone compatibility)
                    img_array = np.stack([img, img, img], axis=-1)

    # 3. Save to cache
    try:
        np.save(cache_path, img_array)
    except Exception:
        pass

    return img_array.astype(np.float32)


class OSICDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, mode: str = "train", load_cached_data: bool = True
    ):
        """
        Dataset class for OSIC Pulmonary Fibrosis Progression.

        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached images.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Normalization Constants (Approximate from EDA)
        self.norm_stats = {
            "Age": (67.0, 15.0),
            "Percent": (77.0, 20.0),
            "FVC": (Config.TARGET_MEAN, Config.TARGET_STD),
            "Weeks": (30.0, 25.0),
        }

        # Data Preparation
        if mode == "test":
            # For test, df is the test.csv (containing baselines).
            # We need to expand this to predict for every Patient_Week in sample_submission.
            if os.path.exists(Config.SAMPLE_SUBMISSION):
                sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
            else:
                # Fallback if sample submission is not found (e.g. during simple debug)
                # Just use the df as is, though this won't match submission format
                sub_df = df.copy()
                sub_df["Patient_Week"] = (
                    sub_df["Patient"] + "_" + sub_df["Weeks"].astype(str)
                )

            # Parse Patient and Weeks from Patient_Week ID
            # ID format: ID00419637202311204720264_12
            sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
            sub_df["Weeks"] = sub_df["Patient_Week"].apply(
                lambda x: int(x.split("_")[1])
            )

            # Merge baseline info from test.csv
            # test.csv has [Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus] (Baseline values)
            base_df = df.rename(
                columns={
                    "Weeks": "Baseline_Weeks",
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Age": "Baseline_Age",
                }
            )

            # Merge left to keep all submission rows
            self.data = sub_df.merge(base_df, on="Patient", how="left")

        else:
            # For train/val, we have the full history in df.
            # We need to identify the baseline (first visit) for each patient to calculate features.

            # Sort by Weeks to find the first visit
            df_sorted = df.sort_values(["Patient", "Weeks"])

            # Extract baseline rows
            baseline_df = df_sorted.groupby("Patient").first().reset_index()

            # Select relevant columns and rename
            baseline_cols = [
                "Patient",
                "Weeks",
                "FVC",
                "Percent",
                "Age",
                "Sex",
                "SmokingStatus",
            ]
            baseline_df = baseline_df[baseline_cols].rename(
                columns={
                    "Weeks": "Baseline_Weeks",
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Age": "Baseline_Age",
                    "Sex": "Baseline_Sex",
                    "SmokingStatus": "Baseline_SmokingStatus",
                }
            )

            # Merge back to the original df
            self.data = df.merge(baseline_df, on="Patient", how="left")

            # Ensure Sex and SmokingStatus are available in the main columns for consistency
            self.data["Sex"] = self.data["Baseline_Sex"]
            self.data["SmokingStatus"] = self.data["Baseline_SmokingStatus"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row["Patient"]

        # --- 1. Tabular Features ---
        # Calculate relative weeks from baseline
        weeks_rel = row["Weeks"] - row["Baseline_Weeks"]

        # Normalize Numerical Features
        feat_weeks = weeks_rel / 100.0
        feat_age = (row["Baseline_Age"] - self.norm_stats["Age"][0]) / self.norm_stats[
            "Age"
        ][1]
        feat_fvc = (row["Baseline_FVC"] - self.norm_stats["FVC"][0]) / self.norm_stats[
            "FVC"
        ][1]
        feat_pct = (
            row["Baseline_Percent"] - self.norm_stats["Percent"][0]
        ) / self.norm_stats["Percent"][1]

        # Encode Categorical Features
        # Sex: Male=0, Female=1
        feat_sex = 1.0 if row["Sex"] == "Female" else 0.0

        # Smoking: [Ex-smoker, Never smoked, Currently smokes]
        feat_smoke = [0.0, 0.0, 0.0]
        status = row["SmokingStatus"]
        if status == "Ex-smoker":
            feat_smoke[0] = 1.0
        elif status == "Never smoked":
            feat_smoke[1] = 1.0
        elif status == "Currently smokes":
            feat_smoke[2] = 1.0

        # Concatenate tabular features
        tabular = np.array(
            [feat_weeks, feat_fvc, feat_pct, feat_age, feat_sex] + feat_smoke,
            dtype=np.float32,
        )

        # --- 2. Image Features ---
        img_array = get_img(
            patient_id, mode=self.mode, load_cached_data=self.load_cached_data
        )
        # Convert to tensor (C, H, W)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)

        # --- 3. Target ---
        if self.mode != "test":
            target_fvc = float(row["FVC"])
            # Apply Target Scaling Cite {solution_lesson_node_00001}
            target_fvc = (target_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
            return img_tensor, tabular, torch.tensor([target_fvc], dtype=torch.float32)
        else:
            # For test, return dummy target
            return img_tensor, tabular, torch.tensor([0.0], dtype=torch.float32)


def get_dataloaders(
    train_df, val_df, test_df, batch_size=Config.BATCH_SIZE, debug=False
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        test_df (pd.DataFrame): Test metadata.
        batch_size (int): Batch size.
        debug (bool): If True, subsets data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        # Test usually small, keep as is or slice if needed

    train_dataset = OSICDataset(train_df, mode="train", load_cached_data=True)
    val_dataset = OSICDataset(val_df, mode="val", load_cached_data=True)
    test_dataset = OSICDataset(test_df, mode="test", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
