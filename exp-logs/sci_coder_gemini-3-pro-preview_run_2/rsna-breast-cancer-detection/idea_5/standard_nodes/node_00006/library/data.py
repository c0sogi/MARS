import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.utils import get_logger

logger = get_logger(name="data")


def process_metadata(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads, processes, and caches metadata for a specific split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_metadata_{split}.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached metadata for {split} from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Load Raw Data
    if split == "train":
        file_path = Config.TRAIN_METADATA
    elif split == "val":
        file_path = Config.VAL_METADATA
    elif split == "test":
        file_path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    logger.info(f"Processing metadata for {split} from {file_path}")
    df = pd.read_csv(file_path)

    # 3. Construct Full File Paths
    # The metadata contains 'file_path' relative to input dir (e.g., "train_images/pat/img.dcm")
    # We prepend INPUT_DIR
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # 4. Process Meta Features

    # Age: Fill NaN with mean (computed roughly from training set analysis ~58) and normalize
    # We use a fixed mean/std to avoid leakage or inconsistency between splits
    AGE_MEAN = 58.0
    AGE_STD = 10.0
    if "age" in df.columns:
        df["age"] = df["age"].fillna(AGE_MEAN)
        df["age_norm"] = (df["age"] - AGE_MEAN) / AGE_STD
    else:
        # Should not happen based on dataset desc, but safety fallback
        df["age_norm"] = 0.0

    # Implant: 0 or 1
    if "implant" in df.columns:
        df["implant"] = df["implant"].fillna(0).astype(int)

    # Mappings
    # We map strings to integers using Config maps.
    # If a value is not in the map, we assign a default (e.g. 0) or handle it.

    # Laterality
    df["laterality_enc"] = (
        df["laterality"].map(Config.LATERALITY_MAP).fillna(0).astype(int)
    )

    # View
    df["view_enc"] = df["view"].map(Config.VIEW_MAP).fillna(0).astype(int)

    # Site ID
    df["site_id_enc"] = df["site_id"].map(Config.SITE_ID_MAP).fillna(0).astype(int)

    # 5. Process Targets (Train/Val only)
    if split in ["train", "val"]:
        # Cancer (Primary Target)
        df["cancer"] = df["cancer"].fillna(0).astype(int)

        # BIRADS (Aux) - Map and fill missing with -1
        if "BIRADS" in df.columns:
            df["birads_enc"] = df["birads_map_col"] = df["BIRADS"].map(
                Config.BIRADS_MAP
            )
            df["birads_enc"] = df["birads_enc"].fillna(-1).astype(int)
        else:
            df["birads_enc"] = -1

        # Density (Aux) - Map and fill missing with -1
        if "density" in df.columns:
            df["density_enc"] = df["density"].map(Config.DENSITY_MAP)
            df["density_enc"] = df["density_enc"].fillna(-1).astype(int)
        else:
            df["density_enc"] = -1

    # 6. Save Cache
    try:
        df.to_parquet(cache_path, index=False)
        logger.info(f"Saved processed metadata to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return df


class MammographyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, transforms=None):
        self.df = df
        self.split = split
        self.transforms = transforms
        self.is_test = split == "test"

        # Pre-extract columns to numpy arrays for faster access
        self.paths = self.df["full_path"].values

        # Meta features: [age_norm, implant, laterality, view, site_id]
        # We will pass these as separate tensors or a single vector.
        # For the model defined in idea, it likely takes a vector or specific inputs.
        # Let's prepare a vector for the MLP: [age_norm, implant, lat_enc, view_enc, site_enc]
        self.meta_features = np.stack(
            [
                self.df["age_norm"].values,
                self.df["implant"].values,
                self.df["laterality_enc"].values,
                self.df["view_enc"].values,
                self.df["site_id_enc"].values,
            ],
            axis=1,
        ).astype(np.float32)

        if not self.is_test:
            self.targets_cancer = self.df["cancer"].values.astype(np.float32)
            self.targets_birads = self.df["birads_enc"].values.astype(np.int64)
            self.targets_density = self.df["density_enc"].values.astype(np.int64)

        # For submission mapping
        if self.is_test:
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # 1. Load Image
        # Using byte-level reading + imdecode to handle potential DICOM/JPEG2000 issues
        try:
            # Read file as byte stream
            with open(path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

            # Decode
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("cv2.imdecode returned None")

        except Exception as e:
            # Fallback for corrupt images: return black image
            # logger.warning(f"Error loading image {path}: {e}")
            img = np.zeros(Config.IMG_SIZE, dtype=np.uint8)

        # Convert to 3 channels (RGB) for EfficientNet
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 2. Augmentations
        if self.transforms:
            augmented = self.transforms(image=img)
            img_tensor = augmented["image"]
        else:
            # Basic to tensor if no transforms provided (should not happen based on Config)
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # 3. Metadata
        meta_vec = torch.tensor(self.meta_features[idx], dtype=torch.float32)

        # 4. Return
        result = {"image": img_tensor, "meta": meta_vec}

        if not self.is_test:
            result["target_cancer"] = torch.tensor(
                self.targets_cancer[idx], dtype=torch.float32
            )
            result["target_birads"] = torch.tensor(
                self.targets_birads[idx], dtype=torch.long
            )
            result["target_density"] = torch.tensor(
                self.targets_density[idx], dtype=torch.long
            )
        else:
            result["prediction_id"] = self.prediction_ids[idx]

        return result


def get_dataloaders(load_cached_data: bool = True, debug: bool = False):
    """
    Creates DataLoaders for train, val, and test splits.

    Args:
        load_cached_data (bool): Whether to use cached metadata.
        debug (bool): If True, subsamples data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Process Metadata
    train_df = process_metadata("train", load_cached_data)
    val_df = process_metadata("val", load_cached_data)
    test_df = process_metadata("test", load_cached_data)

    # Debug Subsampling
    if debug:
        logger.info("Debug mode: Subsampling datasets...")
        train_df = train_df.sample(
            n=min(200, len(train_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(100, len(val_df)), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(50, len(test_df)), random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Create Datasets
    train_dataset = MammographyDataset(
        train_df, split="train", transforms=Config.get_transforms("train")
    )

    val_dataset = MammographyDataset(
        val_df, split="val", transforms=Config.get_transforms("val")
    )

    test_dataset = MammographyDataset(
        test_df, split="test", transforms=Config.get_transforms("test")
    )

    # 3. Create Samplers
    # WeightedRandomSampler for Training to handle class imbalance
    # Calculate weights based on 'cancer' column in train_df
    targets = train_df["cancer"].values
    class_counts = np.bincount(targets)

    # Avoid division by zero if a class is missing (unlikely in this dataset but good practice)
    if len(class_counts) < 2:
        class_weights = np.ones(2)
    else:
        # Weight = 1 / count
        class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches."
    )

    return train_loader, val_loader, test_loader
