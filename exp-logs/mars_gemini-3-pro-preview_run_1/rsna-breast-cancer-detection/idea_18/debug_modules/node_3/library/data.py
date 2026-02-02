import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_dicom_image
from library.transforms import PairedAugmentation


def get_age_stats(metadata_path, cache_dir, load_cached_data=True):
    """
    Computes or loads cached age statistics (mean, std) from the training data.
    """
    cache_path = os.path.join(cache_dir, "age_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        stats = np.load(cache_path)
        return stats[0], stats[1]

    # Compute from scratch
    df = pd.read_csv(metadata_path)
    # Filter out NaNs for calculation
    valid_ages = df["age"].dropna()
    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Save
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, np.array([mean_age, std_age]))

    return mean_age, std_age


def process_metadata(metadata_path, split_name, cache_dir, load_cached_data=True):
    """
    Loads metadata and adds a 'contra_file_path' column by pairing images
    based on Patient ID and View. Caches the result to Parquet.
    """
    cache_path = os.path.join(cache_dir, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Process from scratch
    df = pd.read_csv(metadata_path)

    # Create lookup dictionary: (patient_id, view) -> {laterality: file_path}
    # This optimizes the pairing process from O(N^2) to O(N)
    lookup = {}
    for _, row in df.iterrows():
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]
        path = row["file_path"]

        key = (pid, view)
        if key not in lookup:
            lookup[key] = {}
        lookup[key][lat] = path

    # Find contralateral paths
    contra_paths = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        view = row["view"]
        lat = row["laterality"]

        # Determine target opposite laterality
        target_lat = "R" if lat == "L" else "L"
        key = (pid, view)

        # Retrieve path if exists, else None
        c_path = lookup.get(key, {}).get(target_lat, None)
        contra_paths.append(c_path)

    df["contra_file_path"] = contra_paths

    # Save
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class SiameseMammogramDataset(Dataset):
    """
    PyTorch Dataset for the Pyramid Symmetry-Difference Siamese Network.
    Yields pairs of (Target, Contralateral) images with spatially broadcasted metadata.
    """

    def __init__(self, df, age_mean, age_std, transform=None, mode="train"):
        self.df = df
        self.age_mean = age_mean
        self.age_std = age_std
        self.transform = transform
        self.mode = mode

        # Pre-convert columns to lists for faster access
        self.file_paths = df["file_path"].tolist()
        self.contra_paths = df["contra_file_path"].tolist()
        self.ages = df["age"].tolist()
        self.implants = df["implant"].tolist()

        # Targets are only available in train/val
        if "cancer" in df.columns:
            self.labels = df["cancer"].tolist()
        else:
            self.labels = None

        # Prediction IDs for test
        if "prediction_id" in df.columns:
            self.prediction_ids = df["prediction_id"].tolist()
        else:
            self.prediction_ids = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Target Image
        target_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])
        try:
            img_target = load_dicom_image(target_path)
        except Exception as e:
            # Fail loudly as per requirements
            raise FileNotFoundError(
                f"Critical Error: Could not load target image {target_path}. {e}"
            )

        # 2. Load Contralateral Image
        contra_rel_path = self.contra_paths[idx]
        img_contra = None

        if contra_rel_path is not None:
            contra_full_path = os.path.join(Config.INPUT_DIR, contra_rel_path)
            if os.path.exists(contra_full_path):
                try:
                    img_contra = load_dicom_image(contra_full_path)
                except:
                    # If corrupt, treat as missing
                    img_contra = None

        # Handle missing contralateral (physically missing or load failure)
        # Create a zero-tensor of same shape as target
        if img_contra is None:
            img_contra = np.zeros_like(img_target)

        # 3. Apply Paired Augmentation
        # Returns (C, H, W) tensors. Since input is grayscale, C=1.
        if self.transform:
            tensor_target, tensor_contra = self.transform(
                img_target, img_contra, mode=self.mode
            )
        else:
            # Fallback (should not happen in this pipeline)
            tensor_target = torch.from_numpy(img_target).float().unsqueeze(0)
            tensor_contra = torch.from_numpy(img_contra).float().unsqueeze(0)

        # 4. Construct Metadata Channels (Age, Implant)
        # Get raw values
        age_raw = self.ages[idx]
        implant_raw = self.implants[idx]

        # Handle missing age (impute with mean -> normalized value 0)
        if pd.isna(age_raw):
            age_norm = 0.0
        else:
            age_norm = (age_raw - self.age_mean) / (self.age_std + 1e-7)

        # Handle missing implant (assume 0)
        if pd.isna(implant_raw):
            implant_val = 0.0
        else:
            implant_val = float(implant_raw)

        # Spatial Broadcasting
        # Create (2, H, W) tensor for metadata
        _, h, w = tensor_target.shape
        meta_maps = torch.zeros((2, h, w), dtype=torch.float32)
        meta_maps[0, :, :] = age_norm
        meta_maps[1, :, :] = implant_val

        # 5. Concatenate to form (3, H, W) inputs
        # Target Input: [Image_T, Age, Implant]
        input_target = torch.cat([tensor_target, meta_maps], dim=0)

        # Contra Input: [Image_C, Age, Implant]
        # Note: Age and Implant are identical for the same patient
        input_contra = torch.cat([tensor_contra, meta_maps], dim=0)

        # 6. Prepare Output
        sample = {
            "image": input_target,
            "image_contra": input_contra,
        }

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.prediction_ids is not None:
            sample["prediction_id"] = self.prediction_ids[idx]

        return sample


def get_loaders(load_cached_data=True):
    """
    Orchestrates the creation of DataLoaders for Train, Val, and Test.
    """
    # 1. Get Age Stats (computed on Train, applied to all)
    age_mean, age_std = get_age_stats(
        Config.TRAIN_METADATA_PATH,
        Config.WORKING_DIR,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Transform Pipeline
    augmenter = PairedAugmentation()

    loaders = {}

    # -------------------------------------------------------------------------
    # Training Loader
    # -------------------------------------------------------------------------
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        df_train = process_metadata(
            Config.TRAIN_METADATA_PATH,
            "train",
            Config.WORKING_DIR,
            load_cached_data=load_cached_data,
        )

        # Debug Mode: Subsample
        if Config.DEBUG:
            df_train = df_train.head(100)

        train_dataset = SiameseMammogramDataset(
            df_train, age_mean, age_std, transform=augmenter, mode="train"
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
        )

    # -------------------------------------------------------------------------
    # Validation Loader
    # -------------------------------------------------------------------------
    if os.path.exists(Config.VAL_METADATA_PATH):
        df_val = process_metadata(
            Config.VAL_METADATA_PATH,
            "val",
            Config.WORKING_DIR,
            load_cached_data=load_cached_data,
        )

        if Config.DEBUG:
            df_val = df_val.head(50)

        val_dataset = SiameseMammogramDataset(
            df_val, age_mean, age_std, transform=augmenter, mode="val"
        )

        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

    # -------------------------------------------------------------------------
    # Test Loader
    # -------------------------------------------------------------------------
    if os.path.exists(Config.TEST_METADATA_PATH):
        df_test = process_metadata(
            Config.TEST_METADATA_PATH,
            "test",
            Config.WORKING_DIR,
            load_cached_data=load_cached_data,
        )

        if Config.DEBUG:
            df_test = df_test.head(50)

        test_dataset = SiameseMammogramDataset(
            df_test,
            age_mean,
            age_std,
            transform=augmenter,
            mode="test",  # Deterministic transform
        )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

    return loaders
