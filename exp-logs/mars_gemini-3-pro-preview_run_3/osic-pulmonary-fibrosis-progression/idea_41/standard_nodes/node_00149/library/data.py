import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Try importing pydicom, handle case if missing (though essential for .dcm)
try:
    import pydicom
except ImportError:
    pydicom = None


class CTPreprocessor:
    """
    Handles loading, windowing, and slice selection for CT scans.
    Implements caching to speed up data loading across epochs.
    """

    def __init__(
        self, img_size=260, window_level=-600, window_width=1500, cache_dir=None
    ):
        self.img_size = img_size
        self.window_level = window_level
        self.window_width = window_width
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_windowing(self, img):
        """Applies radiological lung windowing."""
        lower = self.window_level - self.window_width / 2
        upper = self.window_level + self.window_width / 2
        img = (img - lower) / (upper - lower)
        img = np.clip(img, 0, 1)
        return img

    def load_scan(self, path):
        """Loads all DICOM files from a directory."""
        if pydicom is None:
            return []

        slices = []
        if not os.path.exists(path):
            return []

        # List all files
        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        if not files:
            return []

        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f))
                # Access pixel_array to ensure it's readable
                _ = ds.pixel_array
                slices.append(ds)
            except Exception:
                continue

        # Sort slices by Z-position (ImagePositionPatient[2]) or InstanceNumber
        if not slices:
            return []

        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except AttributeError:
                pass  # Keep original order if sorting fails

        return slices

    def select_slices(self, slices):
        """Selects Anchor (max lung area) and 2 boundary slices."""
        if not slices:
            return None

        # Helper to convert to HU
        def get_hu(ds):
            try:
                img = ds.pixel_array.astype(np.float32)
                intercept = getattr(ds, "RescaleIntercept", 0)
                slope = getattr(ds, "RescaleSlope", 1)
                img = slope * img + intercept
                return img
            except:
                return np.zeros((512, 512), dtype=np.float32)

        # Calculate lung area for each slice
        # Approximation: Lung air is roughly -1000 to -320 HU
        areas = []
        for s in slices:
            img = get_hu(s)
            # Threshold check for lung air
            mask = (img > -1024) & (img < -320)
            areas.append(np.sum(mask))

        areas = np.array(areas)
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        if max_area == 0:
            # Fallback if no lung detected
            indices = [0, len(slices) // 2, len(slices) - 1]
        else:
            # Find ROI where area > 50% of max
            roi_indices = np.where(areas > 0.5 * max_area)[0]

            if len(roi_indices) >= 3:
                # Pick start, max, end of ROI
                indices = [roi_indices[0], max_idx, roi_indices[-1]]
            else:
                # Not enough slices in ROI, take neighbors of max
                indices = [
                    max(0, max_idx - 1),
                    max_idx,
                    min(len(slices) - 1, max_idx + 1),
                ]

        # Ensure unique and sorted
        indices = sorted(list(set(indices)))

        # Pad if needed (duplicate last slice)
        while len(indices) < 3:
            indices.append(indices[-1])

        # Select slices
        selected_slices = [slices[i] for i in indices[:3]]

        # Process images
        processed_imgs = []
        for s in selected_slices:
            img = get_hu(s)
            img = self.get_windowing(img)
            img = cv2.resize(img, (self.img_size, self.img_size))
            processed_imgs.append(img)

        # Stack to (H, W, 3)
        final_img = np.dstack(processed_imgs)
        return final_img

    def process(self, patient_id, image_path, load_cached_data=True):
        """Main processing function with caching."""
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                # Cite debug_lesson_11: Invalidate Stale Caches When Modifying Data Pipeline Configurations
                if data.shape == (self.img_size, self.img_size, 3):
                    return data
            except:
                pass  # Corrupt file, recompute

        # 2. Compute
        full_path = os.path.join(Config.INPUT_DIR, image_path)
        slices = self.load_scan(full_path)
        img = self.select_slices(slices)

        if img is None:
            # Return black image if processing fails
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # 3. Save Cache
        try:
            np.save(cache_path, img)
        except:
            pass

        return img


class LungDataset(Dataset):
    def __init__(self, df, preprocessor, stats, mode="train", cache=True):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.stats = stats
        self.mode = mode
        self.cache = cache

        # Mappings
        self.sex_map = {"Male": 0.0, "Female": 1.0}
        self.smoke_map = {
            "Never smoked": 0.0,
            "Ex-smoker": 1.0,
            "Currently smokes": 2.0,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- Image ---
        img = self.preprocessor.process(
            patient_id, row["image_path"], load_cached_data=self.cache
        )
        # Convert to tensor (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        # --- Clinical Features ---
        # 1. Baseline FVC (Normalized)
        base_fvc = (row["Baseline_FVC"] - self.stats["base_fvc_mean"]) / self.stats[
            "base_fvc_std"
        ]

        # 2. Relative Time (Scaled)
        # t_rel = (Current_Week - Baseline_Week) * 0.01
        t_rel = (row["Weeks"] - row["Baseline_Week"]) * 0.01

        # 3. Age (Normalized)
        age = (row["Age"] - self.stats["age_mean"]) / self.stats["age_std"]

        # 4. Sex
        sex = self.sex_map.get(row["Sex"], 0.0)

        # 5. SmokingStatus
        smoke = self.smoke_map.get(row["SmokingStatus"], 0.0)

        # Construct Vector: [BaseFVC, Time, Age, Sex, Smoking]
        clinical_vec = torch.tensor(
            [base_fvc, t_rel, age, sex, smoke], dtype=torch.float32
        )

        # --- Target ---
        if self.mode == "test":
            return img_tensor, clinical_vec, row["Patient_Week"]
        else:
            fvc = row["FVC"]
            # Z-score normalization for target
            fvc_norm = (fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]
            target = torch.tensor(fvc_norm, dtype=torch.float32)
            return img_tensor, clinical_vec, target


def prepare_dataframes():
    """Loads metadata and prepares train/val/test dataframes with baseline info."""
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Helper to add baseline info to train/val
    def add_baseline(df):
        # Identify baseline row for each patient (min Weeks)
        # Note: We group by Patient and find the row with min Weeks
        # We assume this row contains the 'Baseline' FVC and Week
        baseline_indices = df.groupby("Patient")["Weeks"].idxmin()
        baseline_data = df.loc[baseline_indices, ["Patient", "FVC", "Weeks"]]
        baseline_data.columns = ["Patient", "Baseline_FVC", "Baseline_Week"]

        # Merge back
        df = df.merge(baseline_data, on="Patient", how="left")
        return df

    train_df = add_baseline(train_df)
    val_df = add_baseline(val_df)

    # Prepare Test DataFrame
    # 1. Load sample submission to get target (Patient, Week) pairs
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # 2. Parse Patient and Week from Patient_Week column
    # Format: ID000..._12
    # We assume the last underscore separates ID and Week
    sub_df["Patient"] = sub_df["Patient_Week"].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))

    # 3. Merge static info from test.csv
    # In test.csv, FVC and Weeks represent the Baseline measurement
    test_meta = test_df.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge
    test_expanded = sub_df.merge(test_meta, on="Patient", how="left")

    return train_df, val_df, test_expanded


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """Creates DataLoaders for train, val, and test sets."""
    train_df, val_df, test_df = prepare_dataframes()

    # Compute Normalization Stats from Training Data
    stats = {
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
        "base_fvc_mean": train_df["Baseline_FVC"].mean(),
        "base_fvc_std": train_df["Baseline_FVC"].std(),
    }

    # Initialize Preprocessor
    preprocessor = CTPreprocessor(
        img_size=Config.IMG_SIZE,
        window_level=Config.WINDOW_LEVEL,
        window_width=Config.WINDOW_WIDTH,
        cache_dir=Config.CACHE_DIR,
    )

    # Create Datasets
    train_ds = LungDataset(train_df, preprocessor, stats, mode="train", cache=True)
    val_ds = LungDataset(val_df, preprocessor, stats, mode="val", cache=True)
    test_ds = LungDataset(test_df, preprocessor, stats, mode="test", cache=True)

    # Create Loaders
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

    return train_loader, val_loader, test_loader, stats
