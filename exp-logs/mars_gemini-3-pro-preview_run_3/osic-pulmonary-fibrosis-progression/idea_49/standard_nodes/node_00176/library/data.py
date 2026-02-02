import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_img_pipeline(path, resize_dim=260):
    """
    Reads DICOM files from a directory, applies lung windowing,
    selects 3 slices (Adaptive), and resizes.

    Returns:
        numpy array: (H, W, 3) normalized to [0, 1]
    """
    if not os.path.exists(path):
        return np.zeros((resize_dim, resize_dim, 3), dtype=np.float32)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((resize_dim, resize_dim, 3), dtype=np.float32)

    # Load slices
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(path, f))
            # Ensure we have pixel data
            if hasattr(ds, "pixel_array"):
                slices.append(ds)
        except:
            continue

    if not slices:
        return np.zeros((resize_dim, resize_dim, 3), dtype=np.float32)

    # Sort by ImagePositionPatient Z (robust) or InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.InstanceNumber))
        except AttributeError:
            # Fallback: sort by filename
            pass

    # Convert to HU and Normalize
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)
        img = img * slope + intercept
        images.append(img)

    images = np.array(images)  # (D, H, W)

    # Lung Windowing
    # Level: -600, Width: 1500 -> Range: [-1350, 150]
    level = Config.WINDOW_LEVEL
    width = Config.WINDOW_WIDTH
    lower = level - width / 2
    upper = level + width / 2

    images = np.clip(images, lower, upper)
    images = (images - lower) / (upper - lower)

    # Slice Selection Strategy
    # Calculate "Lung Area" proxy: sum of pixels in a specific intensity range.
    # In normalized [0, 1]: Lung (-600HU) is ~0.5. Air is ~0.23. Tissue is >0.8.
    # We look for pixels between 0.1 and 0.8 to capture lung structure.
    pixel_sums = []
    for i in range(len(images)):
        mask = (images[i] > 0.1) & (images[i] < 0.8)
        pixel_sums.append(np.sum(mask))

    if len(pixel_sums) == 0:
        idx_anchor = 0
    else:
        idx_anchor = np.argmax(pixel_sums)

    max_area = pixel_sums[idx_anchor]

    # Find boundaries (approx 50% area)
    # Search Up
    idx_top = idx_anchor
    for i in range(idx_anchor + 1, len(images)):
        if pixel_sums[i] < 0.5 * max_area:
            idx_top = i
            break

    # Search Down
    idx_bottom = idx_anchor
    for i in range(idx_anchor - 1, -1, -1):
        if pixel_sums[i] < 0.5 * max_area:
            idx_bottom = i
            break

    # Select indices, ensuring they are sorted anatomically
    indices = sorted([idx_bottom, idx_anchor, idx_top])

    # Resize and Stack
    final_slices = []
    for idx in indices:
        slc = images[idx]
        slc = cv2.resize(slc, (resize_dim, resize_dim))
        final_slices.append(slc)

    img_stack = np.stack(final_slices, axis=-1)  # (H, W, 3)

    return img_stack.astype(np.float32)


class OSICDataset(Dataset):
    def __init__(
        self, df, mode="train", transform=None, cache_dir=None, load_cached=True
    ):
        self.df = df.copy()
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.load_cached = load_cached

        # --- Feature Engineering ---

        # Sex: Male=0, Female=1
        self.df["Sex_Code"] = self.df["Sex"].apply(lambda x: 0 if x == "Male" else 1)

        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        self.df["Smoking_Code"] = self.df["SmokingStatus"].apply(
            lambda x: smoke_map.get(x, 0)
        )

        # Normalize Age (Global Stats)
        # Mean=67.58, Std=6.62 (from EDA)
        self.age_mean = 67.58
        self.age_std = 6.62
        self.df["Age_Scaled"] = (self.df["Age"] - self.age_mean) / self.age_std

        # --- Baseline Extraction ---
        # If Baseline columns are not pre-injected (e.g. for training), calculate them.
        if "Baseline_FVC" not in self.df.columns:
            # Sort by Patient and Weeks to find the initial visit
            temp_df = self.df.sort_values(["Patient", "Weeks"])
            baseline_df = temp_df.groupby("Patient").first().reset_index()

            baseline_map = dict(zip(baseline_df["Patient"], baseline_df["FVC"]))
            baseline_week_map = dict(zip(baseline_df["Patient"], baseline_df["Weeks"]))

            self.df["Baseline_FVC"] = self.df["Patient"].map(baseline_map)
            self.df["Baseline_Week"] = self.df["Patient"].map(baseline_week_map)

        # Calculate Relative Time
        self.df["Relative_Week"] = self.df["Weeks"] - self.df["Baseline_Week"]
        self.df["Time_Scaled"] = self.df["Relative_Week"] * Config.TIME_SCALE

        # Normalize Baseline FVC (using Global Target Stats)
        self.df["Baseline_FVC_Scaled"] = (
            self.df["Baseline_FVC"] - Config.TARGET_MEAN
        ) / Config.TARGET_STD

        # Normalize Target FVC (if available and relevant)
        if "FVC" in self.df.columns:
            self.df["FVC_Scaled"] = (
                self.df["FVC"] - Config.TARGET_MEAN
            ) / Config.TARGET_STD

        # Unique Patients and Image Paths
        self.patient_ids = self.df["Patient"].values
        self.image_paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image (with Caching)
        image = None
        cache_path = None

        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
            if self.load_cached and os.path.exists(cache_path):
                try:
                    image = np.load(cache_path)
                except Exception:
                    image = None  # Corrupt file, reprocess

        if image is None:
            # Process from scratch
            full_img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            image = get_img_pipeline(full_img_path, resize_dim=Config.IMAGE_SIZE)

            # Save to cache
            if self.cache_dir and cache_path:
                np.save(cache_path, image)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Convert to tensor (C, H, W)
            image = torch.tensor(image).permute(2, 0, 1)

        # 2. Tabular Features
        # [Baseline FVC (Scaled), Relative Time (Scaled), Age (Scaled), Sex (Code), Smoking (Code)]
        tab_vec = np.array(
            [
                row["Baseline_FVC_Scaled"],
                row["Time_Scaled"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
            ],
            dtype=np.float32,
        )

        # 3. Return
        if self.mode in ["train", "val"]:
            target = np.array([row["FVC_Scaled"]], dtype=np.float32)
            return image, tab_vec, target
        else:
            # For inference, return identifiers to map back predictions
            return image, tab_vec, patient_id, row["Weeks"]


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
):
    """
    Creates DataLoaders for Train, Val, and Test (Submission).
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)

    # Define Transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.2
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=Config.IMAGE_SIZE // 10,
                max_width=Config.IMAGE_SIZE // 10,
                p=0.2,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # Train/Val Datasets
    train_dataset = OSICDataset(
        train_df,
        mode="train",
        transform=train_transform,
        cache_dir=Config.CACHE_DIR,
        load_cached=True,
    )

    val_dataset = OSICDataset(
        val_df,
        mode="val",
        transform=val_transform,
        cache_dir=Config.CACHE_DIR,
        load_cached=True,
    )

    # Test/Submission Dataset Logic
    test_loader = None
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")

    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)

        # Parse Patient and Week from Patient_Week column
        # Example: ID0000_12
        sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

        # Prepare Metadata for Merge
        # Rename columns to avoid collision with submission target columns
        # The metadata contains the BASELINE info
        test_meta_prep = test_meta_df.rename(
            columns={"Weeks": "Baseline_Week", "FVC": "Baseline_FVC"}
        )

        # Merge submission targets with baseline metadata
        submission_combined = pd.merge(sub_df, test_meta_prep, on="Patient", how="left")

        # Create Dataset
        # Note: We pass the pre-calculated Baseline columns so OSICDataset won't re-calculate
        test_dataset = OSICDataset(
            submission_combined,
            mode="test",
            transform=val_transform,
            cache_dir=Config.CACHE_DIR,
            load_cached=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
