import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def process_image(patient_id, image_rel_path, load_cached_data=True):
    """
    Processes DICOM images for a patient: loads, windows, selects slices, and resizes.
    Implements caching mechanism.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_dir_path = os.path.join(Config.INPUT_DIR, image_rel_path)

    # Handle missing directory
    if not os.path.exists(full_dir_path):
        # Return black image if data missing (should not happen based on metadata validation)
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        if load_cached_data:
            np.save(cache_path, img)
        return img

    files = [f for f in os.listdir(full_dir_path) if f.endswith(".dcm")]
    if not files:
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        if load_cached_data:
            np.save(cache_path, img)
        return img

    # Read and sort DICOMs
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(full_dir_path, f))
            # Sort by SliceLocation if available, else InstanceNumber
            pos = getattr(ds, "SliceLocation", 0)
            if pos == 0:
                pos = getattr(ds, "InstanceNumber", 0)
            slices.append((pos, ds))
        except:
            continue

    slices.sort(key=lambda x: x[0])
    slices = [s[1] for s in slices]

    if not slices:
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        if load_cached_data:
            np.save(cache_path, img)
        return img

    # Convert to HU and Apply Windowing
    processed_slices = []
    for ds in slices:
        try:
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            img = ds.pixel_array.astype(np.float32)
            img = img * slope + intercept

            # Lung Window
            level = Config.WINDOW_LEVEL
            width = Config.WINDOW_WIDTH
            lower = level - width / 2
            upper = level + width / 2

            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)  # Normalize to [0, 1]
            processed_slices.append(img)
        except:
            continue

    if not processed_slices:
        img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        if load_cached_data:
            np.save(cache_path, img)
        return img

    processed_slices = np.array(processed_slices)  # (D, H, W)

    # Content-Adaptive Slice Selection
    # Heuristic: Calculate "lung area" by counting pixels in a specific intensity range.
    # In normalized [0,1] range, lung tissue is approx 0.05 to 0.7.
    areas = []
    for s in processed_slices:
        mask = (s > 0.05) & (s < 0.70)
        areas.append(np.sum(mask))

    if not areas:
        max_idx = len(processed_slices) // 2
    else:
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        # Find boundaries (approx 50% of max area)
        # Search upwards
        upper_idx = len(processed_slices) - 1
        for i in range(max_idx + 1, len(processed_slices)):
            if areas[i] < 0.5 * max_area:
                upper_idx = i
                break

        # Search downwards
        lower_idx = 0
        for i in range(max_idx - 1, -1, -1):
            if areas[i] < 0.5 * max_area:
                lower_idx = i
                break

        # Select indices
        indices = [lower_idx, max_idx, upper_idx]
        indices.sort()

        # Stack and Resize
        img_3ch = []
        for idx in indices:
            slc = processed_slices[idx]
            slc = cv2.resize(
                slc, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )
            img_3ch.append(slc)

        img_final = np.stack(img_3ch, axis=-1)  # (H, W, 3)

        # Save to cache
        if load_cached_data:
            try:
                np.save(cache_path, img_final)
            except:
                pass

        return img_final


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", stats=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'submission'.
            stats (dict): Normalization statistics (mean/std for Age, FVC).
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.stats = stats or {}

        # Encodings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        image = process_image(patient_id, row["image_path"], load_cached_data=True)
        # Convert to tensor (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # 2. Clinical Features
        # Age (Standardized)
        age = row["Age"]
        age_mean = self.stats.get("age_mean", 65.0)
        age_std = self.stats.get("age_std", 10.0)
        age_norm = (age - age_mean) / age_std

        # Sex (Encoded)
        sex = self.sex_map.get(row["Sex"], 0)

        # Smoking (Encoded)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        # Relative Time (Scaled)
        # For train/val, Weeks is relative. For submission, we computed it.
        rel_time = row["Weeks"] * Config.TIME_SCALE

        # Construct Feature Vector: [Age, Sex, Smoking, RelativeTime]
        # Matches Config.CLINICAL_INPUT_DIM = 4
        clinical = torch.tensor([age_norm, sex, smoke, rel_time], dtype=torch.float32)

        # 3. Target & Metadata
        raw_fvc = row["FVC"] if "FVC" in row else 0.0

        # Normalize Target (Z-score)
        fvc_mean = self.stats.get("fvc_mean", 2500.0)
        fvc_std = self.stats.get("fvc_std", 500.0)

        if self.mode != "submission":
            fvc_norm = (raw_fvc - fvc_mean) / fvc_std
        else:
            fvc_norm = 0.0  # Dummy for submission

        target = torch.tensor([fvc_norm], dtype=torch.float32)

        # Baseline FVC (Critical for model logic, passed in meta)
        # 'Baseline_FVC' should be in df. If not, fallback to raw_fvc.
        baseline_fvc = row.get("Baseline_FVC", raw_fvc)

        meta = {
            "Patient_Week": str(
                row.get("Patient_Week", f"{patient_id}_{row['Weeks']}")
            ),
            "FVC_raw": float(raw_fvc),
            "Percent": float(row.get("Percent", 0.0)),
            "Baseline_FVC": float(baseline_fvc),
            "Weeks": int(row["Weeks"]),
        }

        return image, clinical, target, meta


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
):
    """
    Creates DataLoaders for train, validation, and submission.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_meta_df = pd.read_csv(Config.TEST_META_PATH)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # --- Helper: Add Baseline FVC ---
    def add_baseline_fvc(df):
        # Assumes df contains history. Finds earliest FVC per patient.
        df_sorted = df.sort_values(["Patient", "Weeks"])
        baseline_df = df_sorted.groupby("Patient")["FVC"].first().reset_index()
        baseline_df.rename(columns={"FVC": "Baseline_FVC"}, inplace=True)
        return df.merge(baseline_df, on="Patient", how="left")

    # Add Baseline FVC to Train/Val
    train_df = add_baseline_fvc(train_df)
    val_df = add_baseline_fvc(val_df)

    # --- Compute Stats (from Train) ---
    stats = {
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
    }

    # --- Prepare Submission Data ---
    # 1. Parse Patient and Weeks from sample_submission
    sub_df = sample_sub.copy()
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # 2. Prepare Test Metadata
    # In test set, the provided FVC IS the baseline FVC.
    test_meta_df["Baseline_FVC"] = test_meta_df["FVC"]

    # 3. Merge Metadata into Submission DF
    # We need static features: Age, Sex, Smoking, image_path, Baseline_FVC
    cols_to_merge = [
        "Patient",
        "Age",
        "Sex",
        "SmokingStatus",
        "image_path",
        "Baseline_FVC",
    ]
    test_merged = sub_df.merge(test_meta_df[cols_to_merge], on="Patient", how="left")

    # --- Instantiate Datasets ---
    train_dataset = OSICDataset(train_df, mode="train", stats=stats)
    val_dataset = OSICDataset(val_df, mode="val", stats=stats)
    test_dataset = OSICDataset(test_merged, mode="submission", stats=stats)

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
