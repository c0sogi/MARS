import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from library.config import Config

# Handle pydicom import gracefully
try:
    import pydicom
except ImportError:
    pydicom = None
    print("Warning: pydicom not found. DICOM loading will fail.")


class LungDataset(Dataset):
    """
    Dataset for Lung Function Decline Prediction.
    Handles loading of cached images or processing raw DICOMs on the fly.
    """

    def __init__(self, df, mode="train", transform=None, baseline_df=None):
        """
        Args:
            df: DataFrame containing the samples (visits).
            mode: 'train', 'val', or 'test'.
            transform: Optional transforms (not used in this implementation).
            baseline_df: Optional DataFrame containing baseline info per patient.
                         If None, it is derived from df (assuming df contains full history).
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Build a lookup for patient baseline information
        self.patient_baselines = {}

        if baseline_df is not None:
            # Use provided baseline info (useful for test set where df is exploded predictions)
            for _, row in baseline_df.iterrows():
                self.patient_baselines[row["Patient"]] = {
                    "FVC": row["FVC"],
                    "Weeks": row["Weeks"],
                    "Age": row["Age"],
                    "Sex": row["Sex"],
                    "SmokingStatus": row["SmokingStatus"],
                    "image_path": row["image_path"],
                }
        else:
            # Derive baseline from history (find visit with min Weeks)
            unique_patients = self.df["Patient"].unique()
            for p in unique_patients:
                patient_rows = self.df[self.df["Patient"] == p]
                # Baseline is the visit with minimum Weeks
                idx = patient_rows["Weeks"].idxmin()
                row = patient_rows.loc[idx]

                self.patient_baselines[p] = {
                    "FVC": row["FVC"],
                    "Weeks": row["Weeks"],
                    "Age": row["Age"],
                    "Sex": row["Sex"],
                    "SmokingStatus": row["SmokingStatus"],
                    "image_path": row["image_path"],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image (Cached or Processed)
        # We use the image path from baseline info
        image = self._load_image(
            patient_id, self.patient_baselines[patient_id]["image_path"]
        )

        # 2. Prepare Clinical Features
        baseline_info = self.patient_baselines[patient_id]

        # Extract raw values
        base_fvc = baseline_info["FVC"]
        base_week = baseline_info["Weeks"]
        current_week = row["Weeks"]  # Target week

        # Relative Time (Scaled by 0.01)
        t_rel = (current_week - base_week) * 0.01

        # Normalize Baseline FVC (Standardize)
        base_fvc_norm = (base_fvc - Config.TARGET_MEAN) / Config.TARGET_STD

        # Normalize Age
        age_norm = (baseline_info["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        # Encode Sex (Male=0, Female=1)
        sex_map = {"Male": 0, "Female": 1}
        sex_enc = sex_map.get(baseline_info["Sex"], 0)

        # Encode Smoking (Never=0, Ex=1, Current=2)
        smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
        smoke_enc = smoke_map.get(baseline_info["SmokingStatus"], 0)

        # Construct Feature Vector
        # [Baseline_FVC, t_rel, Age, Sex, Smoking]
        clinical_vec = torch.tensor(
            [base_fvc_norm, t_rel, age_norm, float(sex_enc), float(smoke_enc)],
            dtype=torch.float32,
        )

        # 3. Prepare Target
        if self.mode == "test":
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target_raw = row["FVC"]
            target_norm = (target_raw - Config.TARGET_MEAN) / Config.TARGET_STD
            target = torch.tensor(target_norm, dtype=torch.float32)

        return image, clinical_vec, target

    def _load_image(self, patient_id, rel_path, load_cached_data=True):
        """
        Loads image from cache or processes from DICOMs.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                img_array = np.load(cache_path)
                return torch.tensor(img_array, dtype=torch.float32)
            except Exception:
                pass  # Fallback to processing if load fails

        # Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img_array = process_dicom_directory(full_path)

        # Save to cache
        try:
            np.save(cache_path, img_array)
        except Exception as e:
            print(f"Failed to cache image for {patient_id}: {e}")

        return torch.tensor(img_array, dtype=torch.float32)


def process_dicom_directory(dir_path):
    """
    Reads DICOMs, selects slices, windows, and resizes.
    Returns: numpy array of shape (3, IMG_SIZE, IMG_SIZE)
    """
    if not os.path.exists(dir_path):
        # Return zeros if directory missing
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    files = glob.glob(os.path.join(dir_path, "*.dcm"))
    if not files or pydicom is None:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # 1. Read DICOMs
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure instance number exists
            if not hasattr(dcm, "InstanceNumber"):
                setattr(dcm, "InstanceNumber", 0)
            slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort by InstanceNumber
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # 2. Convert to HU
    images_hu = []
    for s in slices:
        try:
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = s.pixel_array.astype(np.float32)
            img = img * slope + intercept
            images_hu.append(img)
        except Exception:
            continue

    if not images_hu:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # 3. Content-Adaptive Slice Selection
    # Calculate lung area. We use a threshold of -400 HU to capture all lung tissue and air.
    # We do NOT use a lower bound (e.g., > -1000) to ensure we include air voxels (Cite 00090).
    areas = []
    for img in images_hu:
        mask = img < -400
        areas.append(mask.sum())

    # Anchor: Max lung area
    max_idx = np.argmax(areas)

    # Boundaries: Select slices with a stride
    # We want to capture some volume. Use 10% of depth as stride, min 1.
    n_slices = len(images_hu)
    stride = max(1, n_slices // 10)

    indices = [max_idx - stride, max_idx, max_idx + stride]
    # Clamp indices
    indices = [max(0, min(i, n_slices - 1)) for i in indices]

    selected_imgs = [images_hu[i] for i in indices]

    # 4. Windowing and Resizing
    processed_imgs = []
    # Lung Window
    L, W = -600, 1500
    lower, upper = L - W // 2, L + W // 2

    for img in selected_imgs:
        # Clip
        img = np.clip(img, lower, upper)
        # Normalize to [0, 1]
        img = (img - lower) / (upper - lower)

        # Resize
        try:
            img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        except Exception:
            img_resized = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        processed_imgs.append(img_resized)

    # Stack -> (3, H, W)
    final_tensor = np.stack(processed_imgs, axis=0)
    return final_tensor.astype(np.float32)


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for training and validation.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.head(Config.N_DEBUG_SAMPLES)
        val_df = val_df.head(Config.N_DEBUG_SAMPLES)

    # Create Datasets
    # For train/val, baseline info is derived from the history within the DF
    train_dataset = LungDataset(train_df, mode="train")
    val_dataset = LungDataset(val_df, mode="val")

    # Create Loaders
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

    return train_loader, val_loader


def get_submission_dataloader():
    """
    Creates a DataLoader for the test set (submission format).
    Merges sample_submission.csv with test.csv metadata.
    """
    # Load sample submission and test metadata
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    test_meta_path = Config.TEST_CSV

    sub_df = pd.read_csv(sample_sub_path)
    test_meta_df = pd.read_csv(test_meta_path)

    # Parse Patient and Weeks from Patient_Week column in submission
    # Format: ID000..._12
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.rsplit("_", 1)[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.rsplit("_", 1)[1]))

    # We need to pass the baseline info to LungDataset.
    # test_meta_df contains the baseline info (Weeks=BaseWeek, FVC=BaseFVC, etc.)
    # LungDataset expects a dataframe of samples (sub_df) and a dataframe of baselines (test_meta_df).

    dataset = LungDataset(df=sub_df, mode="test", baseline_df=test_meta_df)

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, sub_df
