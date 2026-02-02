import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Attempt to import pydicom for DICOM processing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not installed. Image features will be zeroed out.")


class LungDataset(Dataset):
    """
    Dataset for Lung FVC Prediction implementing DDSR-Net requirements.

    Features:
    - Content-Adaptive Slice Selection (Anchor + 2 Boundaries).
    - Caching of processed 3D volumes to disk.
    - Relative Time calculation (Weeks - Baseline).
    - Z-score standardization of inputs and targets.
    """

    def __init__(self, mode="train", transform=None):
        self.mode = mode
        self.transform = transform
        self.root_dir = Config.INPUT_DIR
        self.cache_dir = Config.CACHE_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Load Metadata based on mode
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif mode == "test":
            # For test, we drive the dataset using sample_submission.csv
            # This ensures we generate predictions for every requested Patient_Week
            sub_path = (
                Config.SUBMISSION_PATH
                if os.path.exists(Config.SUBMISSION_PATH)
                else os.path.join(Config.INPUT_DIR, "sample_submission.csv")
            )
            sub_df = pd.read_csv(sub_path)
            test_meta = pd.read_csv(Config.TEST_CSV)

            # Parse Patient and Weeks from "ID..._Week"
            sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
            sub_df["Weeks"] = sub_df["Patient_Week"].apply(
                lambda x: int(x.split("_")[1])
            )

            # Merge with test metadata to get static features (Age, Sex, BaseFVC, etc.)
            # Note: In test.csv, FVC is the baseline measurement.
            self.df = sub_df.merge(
                test_meta, on="Patient", how="left", suffixes=("", "_meta")
            )

        # 2. Pre-compute Baseline Statistics (Base_FVC, Base_Week)
        if mode in ["train", "val"]:
            # For training history, identify the baseline visit (min weeks) for each patient
            baseline_df = (
                self.df.sort_values("Weeks").groupby("Patient").first().reset_index()
            )
            baseline_df = baseline_df[["Patient", "FVC", "Weeks"]].rename(
                columns={"FVC": "Base_FVC", "Weeks": "Base_Week"}
            )
            self.df = self.df.merge(baseline_df, on="Patient", how="left")
        else:
            # For test, the metadata FVC/Weeks ARE the baseline values
            self.df.rename(
                columns={"FVC": "Base_FVC", "Weeks_meta": "Base_Week"}, inplace=True
            )

        # 3. Debug Subsampling
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SIZE]

        # 4. Encoders
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- Image Loading (Cached) ---
        image = self._load_patient_image(patient_id)

        # --- Tabular Feature Engineering ---
        # Extract raw values
        base_fvc = row["Base_FVC"]
        age = row["Age"]
        sex = self.sex_map.get(row["Sex"], 0)
        smoking = self.smoke_map.get(row["SmokingStatus"], 0)

        # Calculate Relative Time: t_rel = (CurrentWeek - BaseWeek)
        current_week = row["Weeks"]
        base_week = row.get("Base_Week", 0)
        rel_week = current_week - base_week

        # Normalize Features (Z-score for FVC/Age, Scale for Time)
        base_fvc_norm = (base_fvc - Config.BASE_FVC_MEAN) / Config.BASE_FVC_STD
        age_norm = (age - Config.AGE_MEAN) / Config.AGE_STD
        rel_week_scaled = rel_week * 0.01  # Scale by 0.01 per Idea

        # Construct Tabular Vector: [BaseFVC, Age, Sex, Smoking, t_rel]
        # The model will split this into Stream A and Stream B inputs
        tabular = torch.tensor(
            [base_fvc_norm, age_norm, float(sex), float(smoking), rel_week_scaled],
            dtype=torch.float32,
        )

        # --- Target Preparation ---
        if self.mode != "test":
            raw_fvc = row["FVC"]
            # Z-score standardization for target
            target_norm = (raw_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
            target = torch.tensor(target_norm, dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        return {
            "image": image,
            "tabular": tabular,
            "target": target,
            "patient_week": f"{patient_id}_{current_week}",
            "raw_fvc": row["FVC"] if self.mode != "test" else 0,
            "rel_week": rel_week,
        }

    def _load_patient_image(self, patient_id):
        """Loads image from cache or processes from scratch."""
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        expected_shape = (3, Config.IMG_SIZE, Config.IMG_SIZE)

        # 1. Try Cache
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                if data.shape == expected_shape:
                    return torch.from_numpy(data).float()
            except Exception:
                pass  # Corrupt file, re-process

        # 2. Process
        img_array = self._process_dicom(patient_id)

        # 3. Save Cache
        try:
            np.save(cache_path, img_array)
        except Exception as e:
            pass  # Non-critical failure

        return torch.from_numpy(img_array).float()

    def _process_dicom(self, patient_id):
        """
        Reads DICOMs, converts to HU, selects 3 adaptive slices, resizes, and normalizes.
        """
        if not HAS_PYDICOM:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Locate directory (could be in train or test folder)
        path_train = os.path.join(self.root_dir, "train", patient_id)
        path_test = os.path.join(self.root_dir, "test", patient_id)

        if os.path.exists(path_train):
            patient_dir = path_train
        elif os.path.exists(path_test):
            patient_dir = path_test
        else:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Read all DICOM files
        files = [f for f in os.listdir(patient_dir) if f.endswith(".dcm")]
        if not files:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(patient_dir, f))
                # Sort by InstanceNumber if available, else filename
                pos = (
                    int(ds.InstanceNumber)
                    if hasattr(ds, "InstanceNumber")
                    else int(f.split(".")[0])
                )
                slices.append((pos, ds))
            except Exception:
                continue

        slices.sort(key=lambda x: x[0])
        if not slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Convert to HU and Calculate Lung Area
        processed_slices = []
        areas = []

        for _, ds in slices:
            try:
                # Get pixels
                img = ds.pixel_array.astype(np.float32)

                # Apply Slope/Intercept
                intercept = getattr(ds, "RescaleIntercept", -1024)
                slope = getattr(ds, "RescaleSlope", 1)
                img = img * slope + intercept

                # Calculate Area (pixels > -1000 HU)
                area = np.sum(img > -1000)

                processed_slices.append(img)
                areas.append(area)
            except Exception:
                continue

        if not processed_slices:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # Content-Adaptive Selection
        # 1. Find Anchor (Max Area)
        idx_anchor = np.argmax(areas)
        max_area = areas[idx_anchor]

        # 2. Find Boundaries (First slice < 50% max area in each direction)
        threshold = 0.5 * max_area

        # Top (earlier indices)
        idx_top = 0
        for i in range(idx_anchor - 1, -1, -1):
            if areas[i] < threshold:
                idx_top = i
                break

        # Bottom (later indices)
        idx_bottom = len(processed_slices) - 1
        for i in range(idx_anchor + 1, len(processed_slices)):
            if areas[i] < threshold:
                idx_bottom = i
                break

        # Select and Resize
        selected_indices = sorted([idx_top, idx_anchor, idx_bottom])
        final_volume = []

        for idx in selected_indices:
            img = processed_slices[idx]

            # Resize
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

            # Windowing (-1000 to 400) and Normalization (0 to 1)
            img = np.clip(img, -1000, 400)
            img = (img - (-1000)) / (400 - (-1000))

            final_volume.append(img)

        return np.stack(final_volume, axis=0).astype(np.float32)


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """Factory for Train/Val DataLoaders."""
    train_ds = LungDataset(mode="train")
    val_ds = LungDataset(mode="val")

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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """Factory for Test DataLoader."""
    test_ds = LungDataset(mode="test")
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader
