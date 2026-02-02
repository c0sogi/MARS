import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from library.config import Config


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Holds images in memory as numpy arrays to maximize throughput.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, 3) containing images.
            labels (np.ndarray): Array of shape (N, NumClasses) containing labels.
            transform (A.Compose): Albumentations transform pipeline.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Convert label to float tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_transforms(phase="train", policy_params=None):
    """
    Constructs the Albumentations transform pipeline based on phase and policy.

    Args:
        phase (str): 'train', 'val', or 'test'.
        policy_params (dict, optional): Dictionary containing 'cutout_params'.
                                        Used for Teachers/Student policies.
    """
    transforms = []

    # 1. Base Augmentations (Train only)
    if phase == "train":
        # Horizontal Flip is standard for this dataset/task per requirements
        transforms.append(A.HorizontalFlip(p=0.5))

        # Apply Policy-Specific Regularization (Cutout / CoarseDropout)
        if policy_params and "cutout_params" in policy_params:
            cp = policy_params["cutout_params"]
            transforms.append(
                A.CoarseDropout(
                    max_holes=cp.get("num_holes", 1),
                    max_height=cp.get("max_h_size", 16),
                    max_width=cp.get("max_w_size", 16),
                    min_holes=1,
                    min_height=1,
                    min_width=1,
                    fill_value=0,
                    p=cp.get("p", 0.5),
                )
            )

    # 2. Normalization and Tensor Conversion (All phases)
    # Note: Resizing is handled during data loading/caching to save memory/compute
    transforms.extend(
        [
            A.Normalize(
                mean=Config.NORM_MEAN,
                std=Config.NORM_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def _load_cache_data(metadata_df, cache_prefix, load_cached_data=True):
    """
    Loads images and labels, utilizing caching to .npy files.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_array, labels_array)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"cache_{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"cache_{cache_prefix}_labels.npy")

    # Disable caching if debugging with subset
    if Config.MAX_SAMPLES is not None:
        load_cached_data = False

    # Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(lbl_cache_path)
    ):
        try:
            images = np.load(img_cache_path)
            labels = np.load(lbl_cache_path)
            # Verify length matches current dataframe (in case metadata changed)
            if len(images) == len(metadata_df):
                print(f"Loaded {cache_prefix} data from cache.")
                return images, labels
        except Exception as e:
            print(
                f"Failed to load cache for {cache_prefix}: {e}. Reloading from source."
            )

    # Process from scratch
    print(f"Processing {cache_prefix} data from source...")

    images_list = []
    labels_list = []

    # Identify label columns
    label_cols = [c for c in metadata_df.columns if c.startswith("species_")]

    for idx, row in metadata_df.iterrows():
        # Map wav path to bmp path
        # metadata file_path: essential_data/src_wavs/PC...wav
        wav_path = row["file_path"]
        basename = os.path.basename(wav_path)
        bmp_name = os.path.splitext(basename)[0] + ".bmp"

        # Construct full path to spectrogram
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)

        if not os.path.exists(img_path):
            # Fallback: create black image if missing (should not happen per EDA)
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
        else:
            # Load as Grayscale
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH, 3), dtype=np.uint8)
            else:
                # Resize to target resolution
                img_resized = cv2.resize(
                    img_gray,
                    (Config.IMG_WIDTH, Config.IMG_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                # Channel Replication: Gray -> RGB
                img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

        images_list.append(img)

        # Extract labels
        if len(label_cols) > 0:
            labels_list.append(row[label_cols].values.astype(np.float32))
        else:
            # Placeholder for test set if columns missing
            labels_list.append(np.zeros(Config.NUM_CLASSES, dtype=np.float32))

    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=np.float32)

    # Save to cache if not debugging
    if Config.MAX_SAMPLES is None:
        np.save(img_cache_path, images)
        np.save(lbl_cache_path, labels)
        print(f"Saved {cache_prefix} data to cache.")

    return images, labels


def get_dataloaders(
    load_cached_data=True, teacher_policy=None, use_pseudo_labels=False
):
    """
    Creates Train and Validation DataLoaders.

    Args:
        load_cached_data (bool): Use cached .npy files if available.
        teacher_policy (str, optional): Key in Config for specific teacher policy (e.g., 'POLICY_TEACHER_1').
                                        If None, uses 'POLICY_BALANCED' (Student default).
        use_pseudo_labels (bool): If True, merges Test set with Pseudo-labels into Training set (for Student).

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    if Config.MAX_SAMPLES:
        df_train = df_train.iloc[: Config.MAX_SAMPLES]
        df_val = df_val.iloc[: Config.MAX_SAMPLES]

    # 2. Load Data Arrays
    X_train, y_train = _load_cache_data(df_train, "train", load_cached_data)
    X_val, y_val = _load_cache_data(df_val, "val", load_cached_data)

    # 3. Handle Pseudo Labels (Student Training)
    if use_pseudo_labels:
        print("Loading Pseudo-Labeled Data...")
        df_test = pd.read_csv(Config.TEST_CSV)
        if Config.MAX_SAMPLES:
            df_test = df_test.iloc[: Config.MAX_SAMPLES]

        # Load Test Images
        X_test, _ = _load_cache_data(df_test, "test", load_cached_data)

        # Load Pseudo Labels from Parquet
        if os.path.exists(Config.PSEUDO_LABEL_PATH):
            df_pseudo = pd.read_parquet(Config.PSEUDO_LABEL_PATH)

            # Ensure alignment: Merge df_test with df_pseudo on 'rec_id'
            # We assume df_pseudo has columns: rec_id, species_0, ..., species_18
            df_test_merged = pd.merge(
                df_test[["rec_id"]], df_pseudo, on="rec_id", how="left"
            )

            # Extract label columns
            pseudo_label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
            y_test_pseudo = df_test_merged[pseudo_label_cols].values.astype(np.float32)

            # Check for NaNs (safety)
            if np.isnan(y_test_pseudo).any():
                print("Warning: NaNs found in pseudo labels. Filling with 0.")
                y_test_pseudo = np.nan_to_num(y_test_pseudo)

            # Concatenate Train and Pseudo-Test
            X_train = np.concatenate([X_train, X_test], axis=0)
            y_train = np.concatenate([y_train, y_test_pseudo], axis=0)
            print(f"Combined Train + Pseudo-Test Size: {len(X_train)}")
        else:
            print(
                f"Warning: Pseudo label file {Config.PSEUDO_LABEL_PATH} not found. Using only Train data."
            )

    # 4. Determine Transform Policy
    if teacher_policy:
        policy_params = getattr(Config, teacher_policy, Config.POLICY_BALANCED)
    else:
        policy_params = Config.POLICY_BALANCED

    train_transform = get_transforms(phase="train", policy_params=policy_params)
    val_transform = get_transforms(phase="val")

    # 5. Create Datasets
    train_dataset = BirdDataset(X_train, y_train, transform=train_transform)
    val_dataset = BirdDataset(X_val, y_val, transform=val_transform)

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates a DataLoader for the Test set (Fold 1).
    Used for inference and pseudo-label generation.
    """
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.MAX_SAMPLES:
        df_test = df_test.iloc[: Config.MAX_SAMPLES]

    # Load Test Data (Labels are placeholders/zeros)
    X_test, y_test = _load_cache_data(df_test, "test", load_cached_data)

    # Use Val transform (Resize + Normalize only)
    test_transform = get_transforms(phase="test")

    test_dataset = BirdDataset(X_test, y_test, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Return IDs for submission mapping
    return test_loader, df_test["rec_id"].values
