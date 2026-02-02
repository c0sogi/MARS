import os
import glob
import numpy as np
import pandas as pd
import torch
import pydicom
import cv2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Statistics from Data Analysis (hardcoded for consistency across train/val/test)
STATS = {
    "fvc_mean": 2654.65,
    "fvc_std": 801.70,
    "age_mean": 67.58,
    "age_std": 6.62,
    "percent_mean": 76.91,
    "percent_std": 19.19,
}


class OSICDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load processed images from cache.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data
        self.data = self._load_metadata()

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def _load_metadata(self):
        """Loads and preprocesses metadata based on the mode."""
        if self.mode == "train":
            df = pd.read_csv(Config.TRAIN_CSV)
        elif self.mode == "val":
            df = pd.read_csv(Config.VAL_CSV)
        elif self.mode == "test":
            # For test, we need to predict for every Patient_Week in sample_submission
            # using the baseline info from test.csv
            test_meta = pd.read_csv(Config.TEST_CSV)
            sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

            # Parse Patient and Weeks from sample_submission
            # Format: ID..._Week
            sub_df[["Patient", "Weeks"]] = sub_df["Patient_Week"].str.rsplit(
                "_", n=1, expand=True
            )
            sub_df["Weeks"] = sub_df["Weeks"].astype(int)

            # Merge with baseline metadata
            # Note: test.csv contains the baseline measurement for each patient
            df = sub_df.merge(
                test_meta.drop(columns=["Weeks", "FVC", "Percent"]),
                on="Patient",
                how="left",
            )

            # Rename baseline columns from test_meta for clarity if they were kept,
            # but here we need to be careful. test.csv has 'FVC' which is the baseline FVC.
            # Let's re-merge properly.
            test_meta = test_meta.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Week",
                }
            )
            df = sub_df.merge(test_meta, on="Patient", how="left")

            # Fill missing image paths if necessary (though merge should handle it)
            df["image_path"] = df["Patient"].apply(lambda x: os.path.join("test", x))

        # Feature Engineering for Train/Val
        if self.mode in ["train", "val"]:
            # Identify baseline for each patient (min weeks)
            # We assume the provided FVC/Percent in the row are the measurements for that week.
            # We need to append the Baseline FVC/Percent/Week to every row for that patient.

            # Group by patient and find the row with minimum weeks
            # Note: In this dataset, the baseline is usually the first visit.
            # We sort by Weeks and take the first one as baseline.
            df = df.sort_values(["Patient", "Weeks"])
            baseline_df = df.groupby("Patient").first().reset_index()
            baseline_df = baseline_df[["Patient", "FVC", "Percent", "Weeks"]]
            baseline_df = baseline_df.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Week",
                }
            )

            df = df.merge(baseline_df, on="Patient", how="left")

        # Calculate Relative Weeks
        df["Relative_Weeks"] = (df["Weeks"] - df["Baseline_Week"]) * Config.TIME_SCALE

        # Normalize Numerical Features
        df["Age_Scaled"] = (df["Age"] - STATS["age_mean"]) / STATS["age_std"]
        df["Base_FVC_Scaled"] = (df["Baseline_FVC"] - STATS["fvc_mean"]) / STATS[
            "fvc_std"
        ]
        df["Base_Percent_Scaled"] = (
            df["Baseline_Percent"] - STATS["percent_mean"]
        ) / STATS["percent_std"]

        # Encode Categorical Features
        # Sex: Male=0, Female=1
        df["Sex_Code"] = df["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(float)

        # SmokingStatus: Ex-smoker=0, Never smoked=1, Currently smokes=2
        # Mapping based on typical frequency or arbitrary encoding
        smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        df["Smoking_Code"] = df["SmokingStatus"].map(smoke_map).fillna(0).astype(float)

        return df

    def _load_and_process_dicom(self, patient_id, rel_image_path):
        """
        Loads DICOMs, selects slices, windows, resizes, and returns a 3D volume.
        Implements caching.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # 1. Try Load from Cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from Scratch
        full_dir_path = os.path.join(Config.INPUT_DIR, rel_image_path)
        if not os.path.exists(full_dir_path):
            # Fallback for missing directories (should not happen based on metadata check)
            # Return zero volume
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        files = glob.glob(os.path.join(full_dir_path, "*.dcm"))
        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read DICOMs
        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                # Ensure we have image data and slice location
                if hasattr(dcm, "pixel_array"):
                    slices.append(dcm)
            except:
                continue

        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Sort by Z-position (ImagePositionPatient[2]) or InstanceNumber
        slices.sort(
            key=lambda x: (
                float(x.ImagePositionPatient[2])
                if hasattr(x, "ImagePositionPatient")
                else float(x.InstanceNumber)
            )
        )

        # Helper: Windowing
        def get_windowed_img(dcm):
            # Intercept/Slope
            slope = getattr(dcm, "RescaleSlope", 1)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            img = dcm.pixel_array.astype(np.float32) * slope + intercept

            # Apply Lung Window
            level = Config.WINDOW_LEVEL
            width = Config.WINDOW_WIDTH
            lower = level - width / 2
            upper = level + width / 2

            img = np.clip(img, lower, upper)
            # Normalize to [0, 1]
            img = (img - lower) / (upper - lower)
            return img

        # Helper: Calculate Lung Area (simple thresholding on windowed image)
        # Lung tissue is roughly -900 to -300 HU.
        # After windowing [-1350, 150] -> [0, 1].
        # -200 HU is roughly 0.76 in [0, 1] scale? No.
        # -200 HU in range [-1350, 150]: (-200 - (-1350)) / 1500 = 1150 / 1500 = 0.76
        # Air is -1000 HU -> (-1000 + 1350)/1500 = 0.23
        # We want to count pixels that are "lung-like".
        # Let's say pixels < -200 HU.

        processed_imgs = []
        areas = []

        for s in slices:
            img = get_windowed_img(s)
            # Resize
            img_resized = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
            )
            processed_imgs.append(img_resized)

            # Calculate area: pixels < -200 HU.
            # In normalized [0, 1] space, -200 HU is approx 0.76.
            # Wait, lower is -1350. -200 is much higher than air.
            # Lung is dark. Air is dark. Tissue is bright.
            # In HU: Air -1000, Lung -500, Tissue 50.
            # We want to exclude background air and dense tissue.
            # Simple heuristic: count pixels < -300 HU.
            # Threshold in [0, 1]: (-300 - (-1350)) / 1500 = 1050 / 1500 = 0.7
            # So pixels < 0.7 are potentially lung/air.
            # To avoid background air, we usually use a segmentation or just max area of low-density pixels.
            # Let's count pixels < 0.7 (approx -300 HU).
            area = np.sum(img_resized < 0.7)
            areas.append(area)

        # Slice Selection
        # 1. Anchor: Max area
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        # 2. Boundaries: > 50% of max area
        threshold = 0.5 * max_area
        valid_indices = [i for i, a in enumerate(areas) if a > threshold]

        if not valid_indices:
            valid_indices = [max_idx]

        top_idx = valid_indices[0]
        bottom_idx = valid_indices[-1]

        # We want 3 slices: Top, Anchor, Bottom
        # Ensure they are distinct if possible, else duplicate
        selected_indices = [top_idx, max_idx, bottom_idx]

        final_volume = np.stack(
            [processed_imgs[i] for i in selected_indices], axis=0
        )  # (3, H, W)
        final_volume = final_volume.astype(np.float32)

        # 3. Save to Cache
        np.save(cache_path, final_volume)

        return final_volume

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        image = self._load_and_process_dicom(patient_id, row["image_path"])
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # 2. Tabular Features
        # [Baseline_FVC, Baseline_Percent, Relative_Time, Age, Sex, Smoking]
        tabular = np.array(
            [
                row["Base_FVC_Scaled"],
                row["Base_Percent_Scaled"],
                row["Relative_Weeks"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
            ],
            dtype=np.float32,
        )
        tabular_tensor = torch.tensor(tabular, dtype=torch.float32)

        # 3. Target
        if self.mode in ["train", "val"]:
            # Z-score normalize target
            target_raw = row["FVC"]
            target_scaled = (target_raw - STATS["fvc_mean"]) / STATS["fvc_std"]
            return {
                "image": image_tensor,
                "tabular": tabular_tensor,
                "target": torch.tensor(target_scaled, dtype=torch.float32),
                "patient_week": f"{patient_id}_{row['Weeks']}",
            }
        else:
            # Inference mode
            return {
                "image": image_tensor,
                "tabular": tabular_tensor,
                "patient_week": row["Patient_Week"],
            }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=2):
    """
    Creates DataLoaders for train, val, and test sets.
    """
    train_ds = OSICDataset(mode="train", load_cached_data=True)
    val_ds = OSICDataset(mode="val", load_cached_data=True)
    test_ds = OSICDataset(mode="test", load_cached_data=True)

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
