import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class LungDataset(Dataset):
    """
    Dataset class for Lung Function Decline Prediction.
    Handles loading of DICOM images (with caching), clinical metadata, and target variables.
    """

    def __init__(
        self, df, mode="train", cache_dir=Config.CACHE_DIR, load_cached_data=True
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient metadata.
            mode (str): 'train', 'val', or 'test'.
            cache_dir (str): Directory to store/load processed image arrays.
            load_cached_data (bool): Whether to use cached data if available.
        """
        self.df = df.copy()
        self.mode = mode
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare metadata (calculate baselines, encode features)
        self.meta_data = self._prepare_metadata(self.df)

    def _prepare_metadata(self, df):
        """
        Prepares the metadata by calculating baseline FVC, relative time, and encoding categoricals.
        """
        # Sort by Patient and Weeks to ensure consistent ordering
        df = df.sort_values(["Patient", "Weeks"])

        # Identify Baseline FVC and Week for each patient
        # For train/val, the baseline is the visit with the minimum Week number.
        # For test, the provided row is the baseline.

        # Create a mapping of Patient -> Baseline Data
        baseline_df = df.loc[df.groupby("Patient")["Weeks"].idxmin()]
        baseline_map = baseline_df.set_index("Patient")[["FVC", "Weeks"]].to_dict(
            "index"
        )

        def get_base_fvc(patient):
            return baseline_map.get(patient, {}).get("FVC", 2000)

        def get_base_week(patient):
            return baseline_map.get(patient, {}).get("Weeks", 0)

        df["Base_FVC"] = df["Patient"].apply(get_base_fvc)
        df["Base_Week"] = df["Patient"].apply(get_base_week)

        # Calculate Relative Time (Weeks from Baseline)
        df["Relative_Time"] = df["Weeks"] - df["Base_Week"]

        # Encode Categoricals
        # Sex: Male=1, Female=0
        df["Sex_Code"] = df["Sex"].apply(lambda x: 1.0 if x == "Male" else 0.0)

        # SmokingStatus: Ordinal Encoding
        smoke_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
        df["Smoke_Code"] = df["SmokingStatus"].map(smoke_map).fillna(0.0)

        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image (Cached or Processed)
        image = self._get_image(patient_id, row["image_path"])

        # 2. Prepare Clinical Features (Stream A Input)
        # Normalize features based on Config statistics

        # Base FVC (Z-score)
        base_fvc_std = (row["Base_FVC"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Relative Time (Scaled)
        time_scaled = row["Relative_Time"] * Config.TIME_SCALE

        # Age (Z-score)
        age_std = (row["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        # Sex (Binary) & Smoking (Ordinal)
        sex = row["Sex_Code"]
        smoke = row["Smoke_Code"]

        # Construct Clinical Vector: [Base_FVC, Time, Age, Sex, Smoke]
        clinical_vec = np.array(
            [base_fvc_std, time_scaled, age_std, sex, smoke], dtype=np.float32
        )

        data = {
            "image": torch.tensor(image, dtype=torch.float32),
            "clinical": torch.tensor(clinical_vec, dtype=torch.float32),
            "patient_id": patient_id,
            "weeks": row["Weeks"],
        }

        # 3. Prepare Target (if not test mode)
        if self.mode != "test":
            target_raw = row["FVC"]
            # Target Z-score for training stability
            target_std = (target_raw - Config.TARGET_MEAN) / Config.TARGET_STD

            data["target"] = torch.tensor(target_std, dtype=torch.float32)
            data["raw_fvc"] = torch.tensor(target_raw, dtype=torch.float32)
            data["patient_week"] = f"{patient_id}_{row['Weeks']}"
        else:
            # For inference reconstruction
            data["base_fvc"] = row["Base_FVC"]
            data["base_week"] = row["Base_Week"]

        return data

    def _get_image(self, patient_id, rel_path):
        """
        Retrieves the processed image tensor. Uses caching to speed up access.
        """
        cache_file = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception:
                pass  # Fallback to processing if cache is corrupt

        # Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        image = self._process_dicom_directory(full_path)

        # Save to cache
        np.save(cache_file, image)

        return image

    def _process_dicom_directory(self, dir_path):
        """
        Loads DICOMs, selects 3 slices (Anchor + Boundaries), windows, and resizes them.
        Returns: np.array of shape (3, H, W)
        """
        files = glob.glob(os.path.join(dir_path, "*.dcm"))

        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Load slices
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                if hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except:
                continue

        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Sort slices by Z-position (ImagePositionPatient[2]) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except:
                pass  # Keep file order

        # Convert to HU
        images_hu = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = img * slope + intercept
            images_hu.append(img)

        images_hu = np.array(images_hu)

        # Select Slices (Anchor + Boundaries)
        indices = self._select_slice_indices(images_hu)
        selected_imgs = images_hu[indices]

        # Window and Resize
        processed_channels = []
        for img in selected_imgs:
            # Apply Lung Window
            level = Config.WINDOW_LEVEL
            width = Config.WINDOW_WIDTH
            lower = level - width / 2
            upper = level + width / 2

            img_w = np.clip(img, lower, upper)
            img_n = (img_w - lower) / width  # Normalize to [0, 1]

            # Resize
            img_r = cv2.resize(img_n, (Config.IMG_SIZE, Config.IMG_SIZE))
            processed_channels.append(img_r)

        # Stack to (3, H, W)
        return np.stack(processed_channels, axis=0).astype(np.float32)

    def _select_slice_indices(self, images):
        """
        Selects indices for Anchor (Max Lung Area) and two boundaries (50% threshold).
        """
        n = len(images)
        if n < 3:
            # Pad if insufficient slices
            return [0] * 3 if n == 1 else [0, 0, 1] if n == 2 else [0, 0, 0]

        # Estimate Lung Area via thresholding (-1000 to -200 HU)
        lung_mask = (images > -1000) & (images < -200)
        areas = lung_mask.sum(axis=(1, 2))

        # Anchor: Max Area
        anchor_idx = np.argmax(areas)
        max_area = areas[anchor_idx]

        if max_area == 0:
            # Fallback to middle slice
            mid = n // 2
            return [max(0, mid - 1), mid, min(n - 1, mid + 1)]

        # Boundaries: 50% of max area
        threshold = 0.5 * max_area

        # Find lower boundary (index < anchor)
        lower_idx = anchor_idx
        for i in range(anchor_idx - 1, -1, -1):
            if areas[i] < threshold:
                break
            lower_idx = i

        # Find upper boundary (index > anchor)
        upper_idx = anchor_idx
        for i in range(anchor_idx + 1, n):
            if areas[i] < threshold:
                break
            upper_idx = i

        # Return sorted indices for spatial consistency
        return sorted([lower_idx, anchor_idx, upper_idx])


def get_dataloaders(
    debug=Config.DEBUG, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates DataLoaders for training and validation sets.
    """
    train_path = os.path.join(Config.METADATA_DIR, "train.csv")
    val_path = os.path.join(Config.METADATA_DIR, "val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if debug:
        # Subsample for debugging
        train_patients = train_df["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_patients = val_df["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        train_df = train_df[train_df["Patient"].isin(train_patients)]
        val_df = val_df[val_df["Patient"].isin(val_patients)]

    train_ds = LungDataset(train_df, mode="train", load_cached_data=True)
    val_ds = LungDataset(val_df, mode="val", load_cached_data=True)

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

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoader for the test set.
    """
    test_path = os.path.join(Config.METADATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path)

    test_ds = LungDataset(test_df, mode="test", load_cached_data=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader
