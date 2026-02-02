import os
import cv2
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class CTPreprocessor:
    """
    Handles loading, windowing, slice selection, and caching of CT scans.
    """

    def __init__(self, cache_dir=Config.CACHE_DIR, img_size=Config.IMG_SIZE):
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.window_level = Config.WINDOW_LEVEL
        self.window_width = Config.WINDOW_WIDTH

    def get_hu_pixels(self, slices):
        """Converts raw dicom slices to Hounsfield Units."""
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)

        # Convert to HU
        for i, s in enumerate(slices):
            intercept = s.RescaleIntercept if hasattr(s, "RescaleIntercept") else -1024
            slope = s.RescaleSlope if hasattr(s, "RescaleSlope") else 1

            if slope != 1:
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)

            image[i] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def process_patient(self, patient_id, image_dir, load_cached_data=True):
        """
        Loads and processes CT scan for a patient.
        Returns a numpy array of shape (3, H, W) normalized to [0, 1].
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, image_dir)
        if not os.path.exists(full_path):
            # Return zero tensor if directory missing (should not happen with valid metadata)
            return np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        # Load DICOMs
        files = glob.glob(os.path.join(full_path, "*.dcm"))
        if not files:
            return np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                slices.append(dcm)
            except:
                continue

        if not slices:
            return np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        # Sort slices by ImagePositionPatient Z or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: float(x.InstanceNumber))

        # Convert to HU
        image_hu = self.get_hu_pixels(slices)

        # Slice Selection Strategy: Anchor (Max Lung Area) + Boundaries
        # Lung window for area calculation: -1000 to -320 HU
        lung_mask_lower = -1000
        lung_mask_upper = -320

        slice_areas = []
        for i in range(image_hu.shape[0]):
            # Count pixels in lung range
            mask = (image_hu[i] >= lung_mask_lower) & (image_hu[i] <= lung_mask_upper)
            slice_areas.append(np.sum(mask))

        slice_areas = np.array(slice_areas)

        if len(slice_areas) == 0 or np.max(slice_areas) == 0:
            # Fallback: Middle slices
            mid = len(slices) // 2
            selected_indices = [max(0, mid - 1), mid, min(len(slices) - 1, mid + 1)]
        else:
            max_area_idx = np.argmax(slice_areas)
            max_area = slice_areas[max_area_idx]
            threshold = 0.5 * max_area

            # Find boundaries (first and last slice exceeding threshold)
            valid_indices = np.where(slice_areas > threshold)[0]

            if len(valid_indices) > 0:
                min_idx = valid_indices[0]
                max_idx = valid_indices[-1]
            else:
                min_idx = max_area_idx
                max_idx = max_area_idx

            # Ensure we have 3 distinct slices if possible, centered around max area logic
            # Selection: [Lower Boundary, Anchor, Upper Boundary]
            selected_indices = [min_idx, max_area_idx, max_idx]

            # Sort indices to maintain anatomical order
            selected_indices.sort()

        # Extract selected slices
        final_slices = []
        for idx in selected_indices:
            img = image_hu[idx]

            # Apply Windowing
            # (img - (level - width/2)) / width
            lower_bound = self.window_level - self.window_width / 2
            img = (img - lower_bound) / self.window_width
            img = np.clip(img, 0, 1)

            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            final_slices.append(img)

        # Stack to (3, H, W)
        # If we somehow selected the same index multiple times (e.g. only 1 slice available),
        # this correctly duplicates the channel.
        tensor_img = np.stack(final_slices, axis=0).astype(np.float32)

        # 3. Save to cache
        try:
            np.save(cache_path, tensor_img)
        except Exception:
            pass  # Non-critical failure

        return tensor_img


class ClinicalPreprocessor:
    """
    Handles preprocessing of tabular clinical data.
    """

    def __init__(self):
        pass

    def preprocess(self, df, is_train=True):
        """
        Transforms the dataframe into the format required for the model.
        Adds 'Baseline_FVC', 'Relative_Time', and scales features.
        """
        df = df.copy()

        # 1. Identify Baseline FVC and Week
        # For training data, we have full history. We need to find the baseline for each patient.
        # Usually baseline is Week 0 or the minimum week.
        if "Baseline_Week" not in df.columns or "Baseline_FVC" not in df.columns:
            # Group by patient to find baseline
            # We assume the input df might be the raw train.csv or test.csv

            # Helper to find baseline row
            def get_baseline(group):
                # Try to find week 0
                if 0 in group["Weeks"].values:
                    base_row = group[group["Weeks"] == 0].iloc[0]
                else:
                    # Fallback to min week
                    base_row = group.loc[group["Weeks"].idxmin()]
                return pd.Series(
                    {
                        "Baseline_Week": base_row["Weeks"],
                        "Baseline_FVC": base_row["FVC"],
                    }
                )

            baselines = df.groupby("Patient").apply(get_baseline)

            # Merge baseline info back to original df
            # Note: If df is test.csv, it only has 1 row per patient which IS the baseline.
            # The groupby works correctly for that case too.
            if "Baseline_Week" not in df.columns:
                df = df.merge(baselines, on="Patient", how="left")

        # 2. Calculate Relative Time
        # Scaled by 0.01 as per idea
        df["Relative_Time"] = (df["Weeks"] - df["Baseline_Week"]) * Config.TIME_SCALE

        # 3. Encode Categoricals
        # Sex: Male=0, Female=1
        df["Sex_Code"] = df["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

        # SmokingStatus: Never smoked=0, Ex-smoker=1, Currently smokes=2
        smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
        df["Smoking_Code"] = df["SmokingStatus"].map(smoking_map).fillna(0).astype(int)

        # 4. Scale Numerical Features
        # Age
        df["Age_Scaled"] = (df["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        # Baseline FVC
        # We use the global target statistics to scale the baseline FVC as well
        df["Baseline_FVC_Scaled"] = (
            df["Baseline_FVC"] - Config.TARGET_MEAN
        ) / Config.TARGET_STD

        return df


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Prediction.
    """

    def __init__(self, df, image_processor, mode="train"):
        """
        Args:
            df: Pandas dataframe containing patient data.
            image_processor: Instance of CTPreprocessor.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.image_processor = image_processor
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # image_path column contains relative path like "train/ID..."
        image_dir = row["image_path"]
        image = self.image_processor.process_patient(
            patient_id, image_dir, load_cached_data=True
        )
        image_tensor = torch.tensor(image, dtype=torch.float32)

        # 2. Clinical Features
        # [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex_Code, Smoking_Code]
        clinical_features = np.array(
            [
                row["Baseline_FVC_Scaled"],
                row["Relative_Time"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
            ],
            dtype=np.float32,
        )
        clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32)

        # 3. Target
        if self.mode != "test":
            raw_fvc = row["FVC"]
            # Normalize target using global stats
            target_scaled = (raw_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
            target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
            return image_tensor, clinical_tensor, target_tensor
        else:
            return (
                image_tensor,
                clinical_tensor,
                torch.tensor(0.0),
            )  # Dummy target for test


def get_dataloaders(
    train_csv_path=Config.TRAIN_CSV,
    val_csv_path=Config.VAL_CSV,
    batch_size=Config.BATCH_SIZE,
):
    """
    Creates DataLoaders for training and validation.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Preprocess Clinical Data
    clin_preprocessor = ClinicalPreprocessor()
    train_df = clin_preprocessor.preprocess(train_df, is_train=True)
    val_df = clin_preprocessor.preprocess(val_df, is_train=True)

    # Initialize Image Processor
    img_processor = CTPreprocessor()

    # Create Datasets
    train_dataset = LungDataset(train_df, img_processor, mode="train")
    val_dataset = LungDataset(val_df, img_processor, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm/Training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    return train_loader, val_loader
