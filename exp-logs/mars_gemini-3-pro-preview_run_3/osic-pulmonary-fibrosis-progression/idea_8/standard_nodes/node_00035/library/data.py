import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import (
    STATS,
    CACHE_DIR,
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    SEED,
)
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(SEED)


def get_baseline_features(df):
    """
    Extracts the baseline (initial visit) features for each patient.
    Returns a DataFrame indexed by Patient ID.
    """
    # Sort by Weeks to ensure the first record is the baseline
    df_sorted = df.sort_values(["Patient", "Weeks"])
    # Drop duplicates to keep only the first visit per patient
    baseline_df = df_sorted.drop_duplicates(subset=["Patient"], keep="first")
    return baseline_df.set_index("Patient")


class LungDataset(Dataset):
    def __init__(self, df, baseline_df, image_dir, mode="train"):
        """
        Args:
            df: DataFrame containing the rows to predict (Patient, Weeks, [FVC]).
            baseline_df: DataFrame containing static baseline info for each patient.
            image_dir: Directory containing cached .npy image files.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.baseline_df = baseline_df
        self.image_dir = image_dir
        self.mode = mode

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        # Smoking: Ex-smoker, Never smoked, Currently smokes
        # We will use one-hot encoding manually
        self.smoking_categories = ["Ex-smoker", "Never smoked", "Currently smokes"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        image_path = os.path.join(self.image_dir, f"{patient_id}.npy")
        if os.path.exists(image_path):
            try:
                # Shape: (3, H, W)
                image = np.load(image_path).astype(np.float32)
            except Exception:
                image = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            # Fallback for missing images
            image = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # 2. Get Baseline Features
        try:
            base_data = self.baseline_df.loc[patient_id]
        except KeyError:
            # Should not happen if data is consistent
            base_data = row

        # Normalize Static Features
        base_fvc = (base_data["FVC"] - STATS["FVC_MEAN"]) / STATS["FVC_STD"]
        base_percent = (base_data["Percent"] - STATS["PERCENT_MEAN"]) / STATS[
            "PERCENT_STD"
        ]
        base_age = (base_data["Age"] - STATS["AGE_MEAN"]) / STATS["AGE_STD"]

        sex = self.sex_map.get(base_data["Sex"], 0)

        # One-hot encoding for smoking
        smoking_status = base_data["SmokingStatus"]
        smoking_vec = [0, 0, 0]
        if smoking_status in self.smoking_categories:
            smoking_vec[self.smoking_categories.index(smoking_status)] = 1

        # Construct tabular vector
        # [Base_FVC, Base_Percent, Base_Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current]
        tabular = np.array(
            [base_fvc, base_percent, base_age, float(sex)]
            + [float(x) for x in smoking_vec],
            dtype=np.float32,
        )

        # 3. Dynamic Input (Time)
        weeks = row["Weeks"]
        weeks_scaled = (weeks - STATS["WEEKS_MEAN"]) / STATS["WEEKS_STD"]

        # 4. Targets
        if self.mode != "test":
            fvc_raw = row["FVC"]
            fvc_scaled = (fvc_raw - STATS["FVC_MEAN"]) / STATS["FVC_STD"]
        else:
            fvc_scaled = 0.0

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "weeks": torch.tensor([weeks_scaled], dtype=torch.float32),
            "target_fvc": torch.tensor([fvc_scaled], dtype=torch.float32),
            "patient_id": patient_id,
            "raw_weeks": weeks,
        }


def get_train_val_loaders(debug=False):
    """
    Prepares DataLoaders for training and validation.
    """
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    if debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:20]

    # Extract baseline features
    # Note: For training, we want the baseline features to be consistent per patient
    # We derive baseline info from the training history itself
    train_baseline = get_baseline_features(train_df)
    val_baseline = get_baseline_features(val_df)

    train_dataset = LungDataset(train_df, train_baseline, CACHE_DIR, mode="train")

    val_dataset = LungDataset(val_df, val_baseline, CACHE_DIR, mode="val")

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

    return train_loader, val_loader


def get_submission_loader(submission_file_path):
    """
    Prepares DataLoader for submission/inference.
    Parses sample_submission.csv to get the required Patient_Week queries.
    """
    # Load submission template
    sub_df = pd.read_csv(submission_file_path)

    # Parse Patient and Weeks from "Patient_Week" column
    # Format: ID0000..._123
    split_data = sub_df["Patient_Week"].str.rsplit("_", n=1, expand=True)
    sub_df["Patient"] = split_data[0]
    sub_df["Weeks"] = split_data[1].astype(int)

    # Load test baseline data
    test_df = pd.read_csv(TEST_CSV)
    test_baseline = get_baseline_features(test_df)

    test_dataset = LungDataset(sub_df, test_baseline, CACHE_DIR, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
