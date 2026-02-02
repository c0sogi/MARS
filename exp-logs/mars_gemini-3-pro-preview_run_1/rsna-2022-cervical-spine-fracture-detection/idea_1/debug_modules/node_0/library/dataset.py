import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_2_5d_stack, sort_filenames_numerically


class FractureSliceDataset(Dataset):
    """
    Dataset for loading 2.5D slice stacks for fracture detection.
    """

    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'StudyInstanceUID', 'slice_number',
                                      and target columns.
            image_dir (str): Root directory containing study folders.
            transform (A.Compose, optional): Albumentations transforms.
            is_test (bool): Whether this is a test dataset (no labels required).
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        # Pre-define targets if not testing
        if not self.is_test:
            # Ensure columns exist and are float32
            self.labels = self.dataframe[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        slice_num = int(row["slice_number"])

        study_path = os.path.join(self.image_dir, study_uid)

        # Get all slice filenames for the study
        # Note: In a high-performance setting, we would cache directory listings.
        # Here we rely on OS caching for simplicity and memory safety.
        try:
            if os.path.exists(study_path):
                filenames = os.listdir(study_path)
            else:
                filenames = []
        except OSError:
            filenames = []

        # Filter for DICOMs and sort numerically
        filenames = [f for f in filenames if f.endswith(".dcm")]
        filenames = sort_filenames_numerically(filenames)

        # Find the index of the requested slice number
        target_file = f"{slice_num}.dcm"

        try:
            slice_index = filenames.index(target_file)
        except ValueError:
            # If the specific slice file is missing (e.g. mismatch in bbox data),
            # default to the middle slice or 0 to prevent crash.
            slice_index = len(filenames) // 2 if filenames else 0

        # Load 2.5D stack (Slice i-1, i, i+1)
        # Returns (H, W, 3) with values in [0, 1]
        img_stack = load_2_5d_stack(
            study_path,
            filenames,
            slice_index,
            Config.WINDOW_CENTER,
            Config.WINDOW_WIDTH,
            Config.IMG_SIZE,
        )

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img_stack)
            img_tensor = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            img_tensor = torch.from_numpy(img_stack.transpose(2, 0, 1))

        if self.is_test:
            return img_tensor, study_uid, slice_num
        else:
            labels = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, labels


def get_transforms(data="train"):
    """
    Returns albumentations transforms for training or validation/test.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_training_data(load_cached_data=True):
    """
    Generates a balanced dataset of positive (fractured) and negative slices.

    Returns:
        pd.DataFrame: DataFrame with columns [StudyInstanceUID, slice_number, C1..C7, patient_overall]
    """
    cache_path = os.path.join(Config.WORKING_DIR, "train_slice_df.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating training data from scratch...")

    # 2. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Load Bounding Boxes (Positives)
    if os.path.exists(Config.BBOX_PATH):
        df_bbox = pd.read_csv(Config.BBOX_PATH)
    else:
        df_bbox = pd.DataFrame(columns=["StudyInstanceUID", "slice_number"])

    # --- Process Positive Slices ---
    # Filter bboxes to only those in our training split
    train_uids = set(df_train["StudyInstanceUID"].unique())
    df_bbox = df_bbox[df_bbox["StudyInstanceUID"].isin(train_uids)].copy()

    # We only need UID and slice_number. Drop duplicates.
    pos_slices = df_bbox[["StudyInstanceUID", "slice_number"]].drop_duplicates()

    # Merge with patient-level labels. Positive slices inherit patient labels.
    pos_df = pd.merge(pos_slices, df_train, on="StudyInstanceUID", how="left")

    # --- Process Negative Slices ---
    # Sample from patients with patient_overall == 0 (Healthy controls)
    neg_patients = df_train[df_train["patient_overall"] == 0]

    # Calculate number of negatives needed to satisfy POSITIVE_RATIO
    n_pos = len(pos_df)
    if n_pos > 0:
        # n_pos / (n_pos + n_neg) = ratio  => n_neg = n_pos * (1/ratio - 1)
        ratio = Config.POSITIVE_RATIO
        n_neg = int(n_pos * (1.0 / ratio - 1.0))
    else:
        n_neg = 1000  # Fallback default

    neg_samples = []

    # Shuffle negative patients
    neg_patients = neg_patients.sample(frac=1, random_state=Config.SEED).reset_index(
        drop=True
    )

    collected_neg = 0
    patient_idx = 0

    # Iterate through healthy patients and pick random slices
    while collected_neg < n_neg and patient_idx < len(neg_patients):
        row = neg_patients.iloc[patient_idx]
        uid = row["StudyInstanceUID"]
        path = os.path.join(Config.TRAIN_IMAGES_DIR, uid)

        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".dcm")]
            if files:
                # Extract slice numbers from filenames
                slice_nums = [int(os.path.splitext(f)[0]) for f in files]

                # Pick up to 5 random slices per patient
                k = min(len(slice_nums), 5)
                selected_slices = np.random.choice(slice_nums, size=k, replace=False)

                for s in selected_slices:
                    sample = {
                        "StudyInstanceUID": uid,
                        "slice_number": s,
                        "patient_overall": 0,
                    }
                    # Set all specific vertebrae labels to 0
                    for c in Config.TARGET_COLS:
                        if c != "patient_overall":
                            sample[c] = 0
                    neg_samples.append(sample)
                    collected_neg += 1

                    if collected_neg >= n_neg:
                        break

        patient_idx += 1

    neg_df = pd.DataFrame(neg_samples)

    # --- Combine and Save ---
    # Ensure consistent columns
    cols = ["StudyInstanceUID", "slice_number"] + Config.TARGET_COLS

    pos_df = pos_df[cols]
    if not neg_df.empty:
        neg_df = neg_df[cols]
        full_df = pd.concat([pos_df, neg_df], axis=0)
    else:
        full_df = pos_df

    # Shuffle final dataset
    full_df = full_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)

    # Debug mode: truncate dataset
    if Config.DEBUG:
        full_df = full_df.head(Config.DEBUG_DATASET_SIZE)

    # Cache result
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path)
    print(f"Saved training data to {cache_path}. Shape: {full_df.shape}")

    return full_df


def prepare_inference_data(load_cached_data=True):
    """
    Generates a dataframe for inference containing slices for all test studies
    using the defined stride.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_slice_df.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating test data...")
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    samples = []

    # Iterate over all test studies
    for idx, row in df_test_meta.iterrows():
        uid = row["StudyInstanceUID"]
        path = os.path.join(Config.TEST_IMAGES_DIR, uid)

        if not os.path.exists(path):
            continue

        files = [f for f in os.listdir(path) if f.endswith(".dcm")]
        # Sort numerically to ensure correct stride order
        files = sort_filenames_numerically(files)

        # Apply Stride: Select every Nth slice
        for i in range(0, len(files), Config.INFERENCE_STRIDE):
            fname = files[i]
            slice_num = int(os.path.splitext(fname)[0])
            samples.append({"StudyInstanceUID": uid, "slice_number": slice_num})

    df_test = pd.DataFrame(samples)

    # Cache result
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_test.to_parquet(cache_path)
    print(f"Saved test data to {cache_path}. Shape: {df_test.shape}")

    return df_test
