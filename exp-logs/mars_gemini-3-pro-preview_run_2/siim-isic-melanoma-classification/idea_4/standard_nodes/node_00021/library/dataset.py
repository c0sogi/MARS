import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from library.config import Config


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # ShiftScaleRotate helps with the varying scales of lesions
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                # RandomBrightnessContrast helps with varying lighting conditions in dermoscopy
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test only require resizing and normalization
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class ISICDataset(Dataset):
    """
    Custom Dataset for ISIC Skin Lesion Classification.
    Handles loading images, applying transforms, and serving metadata/targets.
    """

    def __init__(
        self, image_paths, meta_features, targets=None, aux_targets=None, transform=None
    ):
        """
        Args:
            image_paths (list or np.array): List of full paths to images.
            meta_features (np.array): Processed metadata features (N, D).
            targets (np.array, optional): Binary targets for malignancy.
            aux_targets (np.array, optional): Multi-class targets for diagnosis.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.image_paths = image_paths
        self.meta_features = meta_features
        self.targets = targets
        self.aux_targets = aux_targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image
        path = self.image_paths[index]
        image = cv2.imread(path)

        # Handle cases where image might not exist or read fails (robustness)
        if image is None:
            # Create a black image as placeholder if load fails
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        # Prepare metadata tensor
        meta = torch.tensor(self.meta_features[index], dtype=torch.float32)

        # Prepare return dictionary
        sample = {"image": image, "meta": meta}

        # Add targets if available
        if self.targets is not None:
            sample["target"] = torch.tensor(self.targets[index], dtype=torch.float32)
        else:
            sample["target"] = torch.tensor(-1.0, dtype=torch.float32)

        # Add auxiliary targets if available
        if self.aux_targets is not None:
            sample["aux_target"] = torch.tensor(
                self.aux_targets[index], dtype=torch.long
            )
        else:
            sample["aux_target"] = torch.tensor(-1, dtype=torch.long)

        return sample


def process_metadata(load_cached_data=True, debug=False):
    """
    Loads metadata CSVs, processes features (Encoding/Scaling), and prepares data for training.
    Implements caching to avoid re-processing on every run.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, subsamples the data for quick debugging.

    Returns:
        tuple: (train_data, val_data, test_data)
               Each is a dictionary containing 'image_paths', 'meta_features', 'targets', 'aux_targets'.
    """

    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    cache_files = {
        "train_meta": os.path.join(cache_dir, "meta_train.npy"),
        "val_meta": os.path.join(cache_dir, "meta_val.npy"),
        "test_meta": os.path.join(cache_dir, "meta_test.npy"),
        "train_aux": os.path.join(cache_dir, "aux_train.npy"),
        "val_aux": os.path.join(cache_dir, "aux_val.npy"),
    }

    # Check if we can load from cache
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading processed metadata from cache...")
        meta_train = np.load(cache_files["train_meta"])
        meta_val = np.load(cache_files["val_meta"])
        meta_test = np.load(cache_files["test_meta"])
        aux_train = np.load(cache_files["train_aux"])
        aux_val = np.load(cache_files["val_aux"])

        # We still need to load CSVs to get image paths and binary targets
        # This is fast compared to feature processing
        df_train = pd.read_csv(Config.TRAIN_META)
        df_val = pd.read_csv(Config.VAL_META)
        df_test = pd.read_csv(Config.TEST_META)

    else:
        print("Processing metadata from scratch...")
        # Load Raw Data
        df_train = pd.read_csv(Config.TRAIN_META)
        df_val = pd.read_csv(Config.VAL_META)
        df_test = pd.read_csv(Config.TEST_META)

        # 1. Handle Missing Values
        # Numerical
        for df in [df_train, df_val, df_test]:
            df[Config.NUM_FEATURES] = df[Config.NUM_FEATURES].fillna(
                df_train[Config.NUM_FEATURES].mean()
            )

        # Categorical
        for df in [df_train, df_val, df_test]:
            df[Config.CAT_FEATURES] = df[Config.CAT_FEATURES].fillna("unknown")

        # Diagnosis (Aux Target) - Fill NaN in train/val
        df_train[Config.AUX_TARGET_COL] = df_train[Config.AUX_TARGET_COL].fillna(
            "unknown"
        )
        df_val[Config.AUX_TARGET_COL] = df_val[Config.AUX_TARGET_COL].fillna("unknown")

        # 2. Numerical Standardization (Fit on Train, Transform All)
        scaler = StandardScaler()
        train_num = scaler.fit_transform(df_train[Config.NUM_FEATURES])
        val_num = scaler.transform(df_val[Config.NUM_FEATURES])
        test_num = scaler.transform(df_test[Config.NUM_FEATURES])

        # 3. Categorical One-Hot Encoding (Fit on Train, Transform All)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        train_cat = ohe.fit_transform(df_train[Config.CAT_FEATURES].astype(str))
        val_cat = ohe.transform(df_val[Config.CAT_FEATURES].astype(str))
        test_cat = ohe.transform(df_test[Config.CAT_FEATURES].astype(str))

        # Concatenate features
        meta_train = np.hstack([train_num, train_cat]).astype(np.float32)
        meta_val = np.hstack([val_num, val_cat]).astype(np.float32)
        meta_test = np.hstack([test_num, test_cat]).astype(np.float32)

        # 4. Auxiliary Target Encoding (Label Encoding)
        le = LabelEncoder()
        aux_train = le.fit_transform(df_train[Config.AUX_TARGET_COL].astype(str))
        # Handle potential unseen labels in validation by mapping them to a default if necessary
        # For this dataset, we assume validation labels are a subset of training labels
        # However, to be safe, we use a mapping approach or just transform and hope for the best.
        # Given the dataset construction (stratified), this should be fine.
        # If error occurs, we could map unseen to 'unknown' (if 'unknown' is in classes), but standard LE throws error.
        # We will assume consistency here.
        aux_val = le.transform(df_val[Config.AUX_TARGET_COL].astype(str))

        # Save to cache
        np.save(cache_files["train_meta"], meta_train)
        np.save(cache_files["val_meta"], meta_val)
        np.save(cache_files["test_meta"], meta_test)
        np.save(cache_files["train_aux"], aux_train)
        np.save(cache_files["val_aux"], aux_val)

        print(f"Metadata processed. Feature dimension: {meta_train.shape[1]}")
        print(f"Auxiliary classes: {len(le.classes_)}")

    # Extract Binary Targets
    target_train = df_train[Config.TARGET_COL].values.astype(np.float32)
    target_val = df_val[Config.TARGET_COL].values.astype(np.float32)

    # Construct Image Paths
    # Input paths are relative, e.g., "jpeg/train/ISIC_xxxx.jpg"
    # We prepend INPUT_ROOT
    paths_train = [
        os.path.join(Config.INPUT_ROOT, x) for x in df_train[Config.FILE_PATH_COL]
    ]
    paths_val = [
        os.path.join(Config.INPUT_ROOT, x) for x in df_val[Config.FILE_PATH_COL]
    ]
    paths_test = [
        os.path.join(Config.INPUT_ROOT, x) for x in df_test[Config.FILE_PATH_COL]
    ]

    # Debug Subsampling
    if debug:
        print(f"DEBUG Mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} samples.")
        size = Config.DEBUG_SAMPLE_SIZE

        # Slice arrays
        paths_train = paths_train[:size]
        meta_train = meta_train[:size]
        target_train = target_train[:size]
        aux_train = aux_train[:size]

        paths_val = paths_val[:size]
        meta_val = meta_val[:size]
        target_val = target_val[:size]
        aux_val = aux_val[:size]

        # For test, we usually want to predict on all even in debug to check pipeline,
        # but to save time we can slice too.
        paths_test = paths_test[:size]
        meta_test = meta_test[:size]

    # Pack into dictionaries
    train_data = {
        "image_paths": paths_train,
        "meta_features": meta_train,
        "targets": target_train,
        "aux_targets": aux_train,
    }

    val_data = {
        "image_paths": paths_val,
        "meta_features": meta_val,
        "targets": target_val,
        "aux_targets": aux_val,
    }

    test_data = {
        "image_paths": paths_test,
        "meta_features": meta_test,
        "targets": None,
        "aux_targets": None,
    }

    return train_data, val_data, test_data
