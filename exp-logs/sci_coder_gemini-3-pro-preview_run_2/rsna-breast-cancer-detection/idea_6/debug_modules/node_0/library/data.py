import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="data_loader")


def process_dicom(path, target_size=None):
    """
    Reads a DICOM file by scanning for the JPEG header and decoding the byte stream.
    This avoids using pydicom and handles compressed transfer syntaxes.
    """
    try:
        with open(path, "rb") as f:
            content = f.read()

        # Search for JPEG Start of Image (SOI) marker: FF D8
        # This works for both standard JPEG and often JPEG 2000 codestreams in DICOM
        idx = content.find(b"\xff\xd8")

        if idx == -1:
            # Fallback: try to load as a standard image if it happens to be converted
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            # Decode from the found header onwards
            img_buffer = np.frombuffer(content[idx:], dtype=np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError("Image decoding failed")

        # Handle 16-bit images (rescale to 8-bit)
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        # Ensure 3 channels (RGB) for ImageNet models
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Resize if requested (though Albumentations usually handles this)
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        return img

    except Exception as e:
        # Return a black image of correct size on failure to prevent crashing
        # logger.warning(f"Failed to load {path}: {e}")
        sz = target_size if target_size else Config.IMG_SIZE
        return np.zeros((sz[0], sz[1], 3), dtype=np.uint8)


class BreastCancerDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train", machine_id_map=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode
        self.machine_id_map = machine_id_map if machine_id_map else {}

        # Pre-process tabular columns to avoid overhead in __getitem__
        self.paths = self.df["file_path"].values
        self.ages = self.df["age"].fillna(self.df["age"].mean()).values
        self.implants = self.df["implant"].values.astype(int)
        self.views = (
            self.df["view"].map(Config.VIEW_MAPPING).fillna(0).values.astype(int)
        )

        # Handle machine_id mapping
        # If ID not in map, assign to len(map) (unknown category)
        self.machine_ids = (
            self.df["machine_id"]
            .apply(lambda x: self.machine_id_map.get(x, len(self.machine_id_map)))
            .values.astype(int)
        )

        # Targets
        if self.mode != "test":
            self.cancer_labels = self.df[Config.TARGET_COL].values.astype(np.float32)

            # Auxiliary: BIRADS
            # Fill NaNs with 0 (follow-up/indeterminate) or a specific class
            self.birads_labels = self.df["BIRADS"].fillna(0).values.astype(np.int64)

            # Auxiliary: Density
            # Map A,B,C,D to 0,1,2,3. Fill NaN with B (most common) or separate class
            self.density_labels = (
                self.df["density"]
                .map(Config.DENSITY_MAPPING)
                .fillna(1)
                .values.astype(np.int64)
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Image Processing
        full_path = os.path.join(Config.INPUT_DIR, self.paths[idx])
        image = process_dicom(full_path)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 2. Tabular Processing
        # Normalize Age: approximate range 20-100 -> 0.2-1.0
        age_norm = self.ages[idx] / 100.0

        # One-hot encode View (6 classes)
        view_ohe = np.zeros(len(Config.VIEW_MAPPING), dtype=np.float32)
        if 0 <= self.views[idx] < len(Config.VIEW_MAPPING):
            view_ohe[self.views[idx]] = 1.0

        # Construct tabular vector
        # [age, implant, machine_id_embedding_index (handled by model), view_ohe...]
        # Here we pass raw indices for embeddings and floats for others.
        # To simplify fusion, we'll return a dict and let the model handle embeddings vs floats.

        # For this specific architecture description, we are concatenating a "normalized tabular vector".
        # We will pass the raw values, and the model's forward pass will handle embedding lookup for machine_id.
        tabular = {
            "age": torch.tensor(age_norm, dtype=torch.float32),
            "implant": torch.tensor(self.implants[idx], dtype=torch.float32),
            "machine_id": torch.tensor(self.machine_ids[idx], dtype=torch.long),
            "view": torch.tensor(view_ohe, dtype=torch.float32),
        }

        # 3. Targets
        if self.mode == "test":
            return {
                "image": image,
                "tabular": tabular,
                "prediction_id": self.df.iloc[idx]["prediction_id"],
            }
        else:
            return {
                "image": image,
                "tabular": tabular,
                "target": torch.tensor(self.cancer_labels[idx], dtype=torch.float32),
                "aux_birads": torch.tensor(self.birads_labels[idx], dtype=torch.long),
                "aux_density": torch.tensor(self.density_labels[idx], dtype=torch.long),
            }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train or validation/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.2,
                ),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE[0], Config.IMG_SIZE[1]),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Implements WeightedRandomSampler for the training set.
    """
    logger.info("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        logger.info("Debug mode: subsampling data.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Generate Machine ID Mapping from Training Data
    unique_machines = train_df["machine_id"].unique()
    machine_id_map = {mid: i for i, mid in enumerate(unique_machines)}
    logger.info(f"Generated Machine ID map with {len(machine_id_map)} devices.")

    # Datasets
    train_dataset = BreastCancerDataset(
        train_df,
        transforms=get_transforms("train"),
        mode="train",
        machine_id_map=machine_id_map,
    )

    val_dataset = BreastCancerDataset(
        val_df,
        transforms=get_transforms("val"),
        mode="val",
        machine_id_map=machine_id_map,
    )

    test_dataset = BreastCancerDataset(
        test_df,
        transforms=get_transforms("test"),
        mode="test",
        machine_id_map=machine_id_map,
    )

    # Sampler for Training (Balanced Sampling)
    logger.info("Configuring WeightedRandomSampler for training...")
    targets = train_df[Config.TARGET_COL].values
    class_counts = np.bincount(targets.astype(int))

    # Calculate weights to achieve BATCH_POS_RATIO
    # Target: P(pos) = ratio, P(neg) = 1 - ratio
    # Weight_pos * N_pos = ratio * Total_Weight
    # Weight_neg * N_neg = (1 - ratio) * Total_Weight
    # W_pos = ratio / N_pos
    # W_neg = (1 - ratio) / N_neg

    ratio = Config.BATCH_POS_RATIO
    w_pos = ratio / (class_counts[1] + 1e-6)
    w_neg = (1.0 - ratio) / (class_counts[0] + 1e-6)

    sample_weights = np.where(targets == 1, w_pos, w_neg)
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders
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

    return train_loader, val_loader, test_loader
