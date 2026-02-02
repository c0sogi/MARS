import os
import glob
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything


class CTPreprocessor:
    """
    Handles loading DICOM scans, selecting slices, windowing, and caching.
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.image_size = Config.IMAGE_SIZE
        self.window_level = Config.WINDOW_LEVEL
        self.window_width = Config.WINDOW_WIDTH

        # Albumentations pipeline for resizing and normalization
        self.transform = A.Compose(
            [
                A.Resize(height=self.image_size, width=self.image_size),
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    def get_img_paths(self, patient_id, dataset_type="train"):
        """Constructs path to DICOM directory."""
        # Dataset type logic: check if patient is in train or test folder
        # The metadata 'image_path' already contains 'train/ID...' or 'test/ID...'
        # But here we might just construct it if we only have ID.
        # However, the robust way is to look in both or rely on metadata.
        # For this helper, we'll search in input dir.

        path_train = os.path.join(Config.INPUT_DIR, "train", patient_id)
        path_test = os.path.join(Config.INPUT_DIR, "test", patient_id)

        if os.path.exists(path_train):
            return path_train
        elif os.path.exists(path_test):
            return path_test
        else:
            return None

    def load_scan(self, dir_path):
        """Loads all DICOM files from a directory and sorts them."""
        files = glob.glob(os.path.join(dir_path, "*.dcm"))
        if not files:
            return []

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                slices.append(dcm)
            except:
                continue

        # Sort by ImagePositionPatient Z (if available) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        return slices

    def get_pixels_hu(self, slices):
        """Converts raw DICOM pixel data to Hounsfield Units."""
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

    def select_slices(self, image_hu):
        """
        Selects 3 slices: Anchor (Max Lung Area) and two boundaries (50% threshold).
        """
        if len(image_hu) < 3:
            # Padding for very small scans
            if len(image_hu) == 0:
                return np.zeros((3, 512, 512), dtype=np.float32)

            indices = np.linspace(0, len(image_hu) - 1, 3).astype(int)
            return image_hu[indices]

        # Calculate Lung Area for each slice
        # Lung tissue approx range: -1000 to -400 HU (ignoring very dark air outside body)
        # Simple heuristic: threshold
        lung_areas = []
        for i in range(len(image_hu)):
            # Threshold
            mask = (image_hu[i] > -1000) & (image_hu[i] < -400)
            lung_areas.append(mask.sum())

        lung_areas = np.array(lung_areas)
        max_area = lung_areas.max()

        if max_area == 0:
            # Fallback: middle slices
            mid = len(image_hu) // 2
            indices = [max(0, mid - 2), mid, min(len(image_hu) - 1, mid + 2)]
        else:
            anchor_idx = np.argmax(lung_areas)

            # Find boundaries where area > 50% of max
            threshold = 0.5 * max_area
            valid_indices = np.where(lung_areas > threshold)[0]

            if len(valid_indices) > 0:
                top_idx = valid_indices[0]
                bottom_idx = valid_indices[-1]
            else:
                top_idx = max(0, anchor_idx - 1)
                bottom_idx = min(len(image_hu) - 1, anchor_idx + 1)

            # Ensure we have 3 distinct slices if possible, sorted anatomically
            indices = sorted(list(set([top_idx, anchor_idx, bottom_idx])))

            # If we collapsed to fewer than 3, pad
            while len(indices) < 3:
                if indices[-1] < len(image_hu) - 1:
                    indices.append(indices[-1] + 1)
                elif indices[0] > 0:
                    indices.insert(0, indices[0] - 1)
                else:
                    indices.append(indices[-1])  # Duplicate if stuck

            # Take exactly 3 (Top, Anchor, Bottom logic is implicit in sorted order)
            # Ideally we want [Top, Anchor, Bottom] but sorted by Z is better for CNN consistency
            if len(indices) > 3:
                # Pick first, middle, last of the valid set?
                # Or specifically the computed ones.
                indices = [top_idx, anchor_idx, bottom_idx]
                indices.sort()

        return image_hu[indices]

    def process_patient(self, patient_id, load_cached_data=True):
        """
        Main processing function.
        Checks cache -> Loads or Processes -> Returns tensor.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, reprocess

        # 1. Locate and Load
        dir_path = self.get_img_paths(patient_id)
        if not dir_path:
            # Return zero tensor if data missing (should not happen in valid dataset)
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        slices = self.load_scan(dir_path)
        if not slices:
            return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

        # 2. Convert to HU
        image_hu = self.get_pixels_hu(slices)

        # 3. Select Slices
        selected_hu = self.select_slices(image_hu)

        # 4. Windowing (Lung Window)
        # (val - (level - width/2)) / width
        lower = self.window_level - (self.window_width / 2)
        upper = self.window_level + (self.window_width / 2)

        image_windowed = np.clip(selected_hu, lower, upper)
        image_windowed = (image_windowed - lower) / (
            upper - lower
        )  # Normalize to [0, 1]

        # 5. Resize & Normalize (Channel-wise)
        # Albumentations expects HWC, so we transpose
        # Shape: (3, H, W) -> (H, W, 3)
        image_hwc = np.transpose(image_windowed, (1, 2, 0)).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image_hwc)
            image_tensor = augmented["image"]  # Returns (3, 260, 260)

        # 6. Cache
        np_image = image_tensor.numpy()
        np.save(cache_path, np_image)

        return np_image


class OSICDataset(Dataset):
    def __init__(self, df, stats, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            stats (dict): Dictionary containing mean/std for scaling.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.stats = stats
        self.mode = mode
        self.preprocessor = CTPreprocessor()

        # Pre-encode categorical mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- 1. Clinical Features ---
        # Baseline FVC (Standardized)
        # Note: In train/val df, we need to ensure we have the baseline FVC available.
        # The preprocessing in get_dataloaders ensures 'Baseline_FVC' column exists.
        baseline_fvc = row["Baseline_FVC"]
        baseline_fvc_scaled = (baseline_fvc - self.stats["fvc_mean"]) / self.stats[
            "fvc_std"
        ]

        # Age (Standardized)
        age = row["Age"]
        age_scaled = (age - self.stats["age_mean"]) / self.stats["age_std"]

        # Sex (Encoded)
        sex = self.sex_map.get(row["Sex"], 0)

        # Smoking (Encoded)
        smoking = self.smoking_map.get(row["SmokingStatus"], 0)

        # Relative Time (Scaled)
        # t_rel = (CurrentWeek - BaselineWeek)
        # In train/val, 'Weeks' is the current week. We need BaselineWeek.
        # In test, 'Weeks' in row is the Target Week.
        weeks = row["Weeks"]
        baseline_week = row["Baseline_Week"]
        rel_time = weeks - baseline_week
        rel_time_scaled = rel_time * 0.01

        # Vector: [Baseline_FVC, Age, Sex, Smoking, Time]
        # Note: The model architecture expects specific inputs.
        # Stream A: [Baseline, Time, Age, Sex, Smoking]
        # We will return them as a dict to be flexible.

        clinical_vec = torch.tensor(
            [
                baseline_fvc_scaled,
                rel_time_scaled,
                age_scaled,
                float(sex),
                float(smoking),
            ],
            dtype=torch.float32,
        )

        # --- 2. Image ---
        # Load cached image
        img_array = self.preprocessor.process_patient(patient_id, load_cached_data=True)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)

        # --- 3. Target ---
        data = {
            "image": img_tensor,
            "clinical": clinical_vec,
            "patient_week": f"{patient_id}_{weeks}",
        }

        if self.mode != "test":
            fvc_raw = row["FVC"]
            # Z-score target
            fvc_target = (fvc_raw - self.stats["fvc_mean"]) / self.stats["fvc_std"]

            data["target"] = torch.tensor(fvc_target, dtype=torch.float32)
            data["fvc_raw"] = torch.tensor(fvc_raw, dtype=torch.float32)

        return data


def get_dataloaders(debug=False):
    """
    Prepares DataLoaders for Train, Val, and Test.
    Handles caching and feature engineering.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        # Keep test small but functional

    # 2. Preprocess Images (Cache Generation)
    # We iterate over all unique patients to ensure cache is populated
    print("Verifying/Generating Image Cache...")
    preprocessor = CTPreprocessor()
    all_patients = pd.concat(
        [train_df["Patient"], val_df["Patient"], test_meta_df["Patient"]]
    ).unique()

    for pid in tqdm(all_patients, desc="Processing Images"):
        preprocessor.process_patient(pid, load_cached_data=True)

    # 3. Feature Engineering: Identify Baselines
    # For Train/Val: The baseline is usually the first visit (min Weeks) OR the visit at Week 0.
    # However, standard practice is to take the initial FVC provided in the dataset as baseline.
    # In this dataset structure, we group by patient and find the row with min(Weeks).

    def add_baseline_info(df, source_df=None):
        # If source_df is provided (e.g. for test set where we look up metadata), use it.
        # Otherwise calculate from self.
        if source_df is None:
            source_df = df

        # Find baseline for each patient (row with min weeks)
        # We sort by Weeks and take first
        baseline_df = (
            source_df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
        )

        # Merge back
        # Note: If df already has these columns, drop them first to avoid collision
        cols_to_drop = [c for c in ["Baseline_FVC", "Baseline_Week"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        return df.merge(baseline_df, on="Patient", how="left")

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # 4. Compute Scaling Stats (from Train only)
    stats = {
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
    }

    # 5. Prepare Test Set
    # The submission file has Patient_Week. We need to explode this.
    # test_meta_df contains the static info (Baseline FVC, Age, Sex, etc) for the test patients.
    # It essentially IS the baseline info.

    test_meta_df = test_meta_df.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Parse submission file
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge metadata into submission rows
    test_df = sub_df.merge(test_meta_df, on="Patient", how="left")

    # 6. Create Datasets
    train_dataset = OSICDataset(train_df, stats, mode="train")
    val_dataset = OSICDataset(val_df, stats, mode="val")
    test_dataset = OSICDataset(test_df, stats, mode="test")

    # 7. Create Loaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
