import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from tqdm import tqdm

from library.config import (
    INPUT_DIR,
    TRAIN_DICOM_DIR,
    TEST_DICOM_DIR,
    METADATA_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    CACHE_DIR,
    IMG_SIZE,
    N_SLICES,
    SLICE_AREA_THRESHOLD,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import DataScaler, seed_everything

# Ensure reproducibility
seed_everything(SEED)


def get_img(path):
    """
    Loads a DICOM file, converts to HU, applies lung windowing, and normalizes.
    """
    try:
        d = pydicom.dcmread(path)
        img = d.pixel_array.astype(np.float32)

        # Convert to HU
        if hasattr(d, "RescaleIntercept") and hasattr(d, "RescaleSlope"):
            intercept = d.RescaleIntercept
            slope = d.RescaleSlope
            img = slope * img + intercept

        # Lung Windowing: Width 1500, Level -600 (Range approx -1350 to 150)
        # We clip to a standard range [-1000, 400] to cover lung tissue
        img = np.clip(img, -1000, 400)

        # Normalize to [0, 1]
        img = (img - (-1000)) / (400 - (-1000))

        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        return img
    except Exception as e:
        # Return zero image in case of failure
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)


def calculate_lung_area(img):
    """
    Approximates lung area by counting pixels within a threshold range.
    Input img is normalized [0, 1]. Lung tissue is roughly mid-range.
    """
    # Thresholding to identify tissue (exclude air ~0 and bone/dense tissue ~1)
    # 0.05 corresponds to approx -930 HU, 0.5 corresponds to -300 HU
    binary = ((img > 0.05) & (img < 0.5)).astype(np.float32)
    return np.sum(binary)


def process_patient(patient_id, dicom_dir):
    """
    Selects 3 slices (Apical, Middle, Basal) based on lung area.
    """
    patient_path = os.path.join(dicom_dir, patient_id)
    if not os.path.exists(patient_path):
        return np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    files = glob.glob(os.path.join(patient_path, "*.dcm"))

    # Sort by instance number to ensure anatomical order
    # Files are usually named 1.dcm, 2.dcm etc.
    try:
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        files.sort()  # Fallback to string sort if naming is weird

    if not files:
        return np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # If few slices, just take what we have and pad/repeat
    if len(files) <= N_SLICES:
        imgs = [get_img(f) for f in files]
        while len(imgs) < N_SLICES:
            imgs.append(imgs[-1])
        return np.array(imgs[:N_SLICES])

    # Load images to calculate areas.
    # Optimization: Subsample if too many slices to speed up
    step = 1 if len(files) < 100 else 2
    sampled_indices = list(range(0, len(files), step))
    areas = []
    valid_indices = []

    for i in sampled_indices:
        img = get_img(files[i])
        area = calculate_lung_area(img)
        areas.append(area)
        valid_indices.append(i)

    areas = np.array(areas)

    if np.max(areas) == 0:
        # Fallback to equidistant if no lung tissue detected
        indices = [0, len(files) // 2, len(files) - 1]
    else:
        max_ind_local = np.argmax(areas)
        max_area = areas[max_ind_local]
        max_ind_global = valid_indices[max_ind_local]

        threshold = (SLICE_AREA_THRESHOLD / 100.0) * max_area

        # Find Apical (Upper boundary) - search backwards from max
        apical_ind = valid_indices[0]
        for i in range(max_ind_local, -1, -1):
            if areas[i] < threshold:
                apical_ind = valid_indices[i]
                break

        # Find Basal (Lower boundary) - search forwards from max
        basal_ind = valid_indices[-1]
        for i in range(max_ind_local, len(areas)):
            if areas[i] < threshold:
                basal_ind = valid_indices[i]
                break

        # Select indices and ensure they are sorted
        indices = sorted(list(set([apical_ind, max_ind_global, basal_ind])))

        # Fill if fewer than 3
        if len(indices) < 3:
            if len(indices) == 1:
                indices = [0, indices[0], len(files) - 1]
            elif len(indices) == 2:
                mid = (indices[0] + indices[1]) // 2
                indices.insert(1, mid)

        indices = indices[:3]

    # Load the final 3 slices
    final_imgs = []
    for idx in indices:
        final_imgs.append(get_img(files[idx]))

    return np.array(final_imgs)


def prepare_cache(patient_ids, load_cached_data=True):
    """
    Iterates through patients and caches their processed images.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    for pid in tqdm(patient_ids, desc="Caching Images"):
        save_path = os.path.join(CACHE_DIR, f"{pid}.npy")

        if load_cached_data and os.path.exists(save_path):
            continue

        # Locate patient directory (could be in train or test)
        p_dir = os.path.join(TRAIN_DICOM_DIR, pid)
        if not os.path.exists(p_dir):
            p_dir = os.path.join(TEST_DICOM_DIR, pid)

        if not os.path.exists(p_dir):
            # If directory missing, save zeros
            dummy = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            np.save(save_path, dummy)
            continue

        # Process and save
        img_vol = process_patient(pid, os.path.dirname(p_dir))
        np.save(save_path, img_vol)


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, scaler=None, mode="train"):
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.scaler = scaler
        self.mode = mode

        # Encode Categoricals
        # Sex: Male=0, Female=1
        self.df["Sex"] = self.df["Sex"].map({"Male": 0, "Female": 1})
        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        self.df["SmokingStatus"] = self.df["SmokingStatus"].map(
            {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        )

        # Fill NaNs if any (categorical mapping might produce NaNs for unknown labels)
        self.df = self.df.fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # Load Image
        img_path = os.path.join(self.cache_dir, f"{pid}.npy")
        if os.path.exists(img_path):
            imgs = np.load(img_path)
        else:
            imgs = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

        imgs = torch.tensor(imgs, dtype=torch.float32)

        # Prepare Tabular Features
        baseline = row["Baseline"]
        weeks = row["Weeks"]
        age = row["Age"]

        if self.scaler:
            baseline = self.scaler.transform_value("Baseline", baseline)
            weeks = self.scaler.transform_value("Weeks", weeks)
            age = self.scaler.transform_value("Age", age)

        # Tabular vector: [Baseline, Weeks, Age, Sex, Smoking]
        tab = torch.tensor(
            [baseline, weeks, age, row["Sex"], row["SmokingStatus"]],
            dtype=torch.float32,
        )

        # Target or ID
        if self.mode in ["train", "val"]:
            target = row["FVC"]
            if self.scaler:
                target = self.scaler.transform_value("FVC", target)
            return imgs, tab, torch.tensor(target, dtype=torch.float32)
        else:
            return imgs, tab, row["Patient_Week"]


def get_dataloaders(load_cached_data=True):
    """
    Prepares dataloaders for train, val, and test.
    Handles caching and scaling.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 2. Add Baseline FVC to Train/Val
    # Baseline is the FVC at the earliest week for each patient
    all_train = pd.concat([train_df, val_df])
    baseline_lookup = (
        all_train.sort_values("Weeks").groupby("Patient").first()[["FVC"]].reset_index()
    )
    baseline_lookup.columns = ["Patient", "Baseline"]

    train_df = train_df.merge(baseline_lookup, on="Patient", how="left")
    val_df = val_df.merge(baseline_lookup, on="Patient", how="left")

    # 3. Prepare Test Dataframe
    # We need to predict for rows in sample_submission.csv
    sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

    # Extract Patient and Week from ID
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.split("_")[1])
    )

    # Prepare test metadata: Rename FVC to Baseline
    # test.csv contains the baseline measurement
    test_meta = test_df.rename(columns={"FVC": "Baseline"})
    # Drop 'Weeks' from metadata as it refers to baseline week, not target week
    if "Weeks" in test_meta.columns:
        test_meta = test_meta.drop(columns=["Weeks"])

    # Merge metadata into sample submission
    test_expanded = sample_sub.merge(test_meta, on="Patient", how="left")

    # 4. Prepare Cache
    all_patients = pd.concat(
        [train_df["Patient"], val_df["Patient"], test_df["Patient"]]
    ).unique()
    prepare_cache(all_patients, load_cached_data=load_cached_data)

    # 5. Fit Scaler
    scaler = DataScaler()
    scaler.fit(train_df)

    # 6. Create Datasets & Loaders
    train_dataset = OSICDataset(train_df, CACHE_DIR, scaler, mode="train")
    val_dataset = OSICDataset(val_df, CACHE_DIR, scaler, mode="val")
    test_dataset = OSICDataset(test_expanded, CACHE_DIR, scaler, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, scaler
