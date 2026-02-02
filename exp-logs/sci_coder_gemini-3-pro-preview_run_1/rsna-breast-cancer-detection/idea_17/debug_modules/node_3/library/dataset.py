import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_image, get_logger

# Initialize logger
logger = get_logger("dataset")


def get_age_stats(load_cached_data=True):
    """
    Computes or loads the mean and standard deviation of patient age from the training set.
    Used for normalizing the age channel in the dataset.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.
                                 If False or file missing, recomputes.

    Returns:
        tuple: (mean_age, std_age)
    """
    cache_path = Config.AGE_STATS_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path)
            logger.info(
                f"Loaded age stats from cache: Mean={stats[0]:.4f}, Std={stats[1]:.4f}"
            )
            return stats[0], stats[1]
        except Exception as e:
            logger.warning(f"Failed to load age stats cache: {e}. Recomputing.")

    # 2. Recompute
    logger.info("Computing age stats from training metadata...")
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA}"
        )

    df = pd.read_csv(Config.TRAIN_METADATA)

    # Handle missing values before computation
    if df["age"].isnull().any():
        logger.warning(
            f"Found {df['age'].isnull().sum()} missing age values in train. Imputing with mean for stats calc."
        )
        temp_age = df["age"].fillna(df["age"].mean())
    else:
        temp_age = df["age"]

    mean_age = temp_age.mean()
    std_age = temp_age.std()

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, np.array([mean_age, std_age]))
    logger.info(
        f"Saved age stats to {cache_path}: Mean={mean_age:.4f}, Std={std_age:.4f}"
    )

    return mean_age, std_age


class SiameseBreastCancerDataset(Dataset):
    """
    Dataset for Pyramid Symmetry-Difference Siamese Network.

    Yields pairs of (Target, Contralateral) images with synchronized augmentations.
    Input format: 3 Channels [Image, Age Map, Implant Map].
    """

    def __init__(self, csv_path, mode="train", debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and return values.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.mode = mode
        self.is_train = mode == "train"

        # Load Metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Debugging: Limit size
        if debug or Config.DEBUG:
            logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
            self.df = self.df.sample(
                n=min(len(self.df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
            ).reset_index(drop=True)

        # Load Age Stats for Normalization
        self.age_mean, self.age_std = get_age_stats(load_cached_data=True)

        # Fill missing age in current dataframe
        if self.df["age"].isnull().sum() > 0:
            self.df["age"] = self.df["age"].fillna(self.age_mean)

        # Fill missing implant (assume 0 if missing)
        if "implant" not in self.df.columns:
            self.df["implant"] = 0
        else:
            self.df["implant"] = self.df["implant"].fillna(0).astype(int)

        # Build Contralateral Lookup
        # We need to quickly find the image of the same patient, same view, opposite laterality.
        # Structure: lookup[(patient_id, view)][laterality] = row_index
        self.lookup = {}
        for idx, row in self.df.iterrows():
            pid = row["patient_id"]
            view = row["view"]
            lat = row["laterality"]

            key = (pid, view)
            if key not in self.lookup:
                self.lookup[key] = {}

            # Store the row index (or the row itself)
            self.lookup[key][lat] = idx

        # Define Augmentations
        # Synchronized: We use 'additional_targets' to apply same transform to contra image
        if self.is_train:
            self.transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image_contra": "image"},
            )
        else:
            self.transforms = A.Compose(
                [ToTensorV2()], additional_targets={"image_contra": "image"}
            )

    def __len__(self):
        return len(self.df)

    def _construct_input(self, image, age, implant):
        """
        Constructs the 3-channel input tensor (H, W, 3).
        Channels:
        0: Image (normalized [0,1])
        1: Age Map (normalized scalar broadcasted)
        2: Implant Map (binary scalar broadcasted)
        """
        h, w = image.shape[:2]

        # Normalize Age
        norm_age = (age - self.age_mean) / (self.age_std + 1e-7)

        # Create maps
        age_map = np.full((h, w), norm_age, dtype=np.float32)
        implant_map = np.full((h, w), implant, dtype=np.float32)

        # Stack
        # image is (H, W) or (H, W, 1) -> ensure (H, W)
        if len(image.shape) == 3:
            image = image[:, :, 0]

        combined = np.stack([image, age_map, implant_map], axis=-1)  # (H, W, 3)
        return combined

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Identify Target
        target_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        patient_id = row["patient_id"]
        view = row["view"]
        laterality = row["laterality"]

        # 2. Identify Contralateral
        contra_laterality = "R" if laterality == "L" else "L"
        contra_key = (patient_id, view)

        has_contra = False
        contra_idx = None

        if contra_key in self.lookup:
            if contra_laterality in self.lookup[contra_key]:
                contra_idx = self.lookup[contra_key][contra_laterality]
                has_contra = True

        # 3. Load Target Image (Fail Loudly)
        try:
            target_img = load_image(target_path, size=Config.IMG_SIZE)
        except (FileNotFoundError, ValueError) as e:
            # In training, we might want to skip, but for this strict implementation we raise
            raise e

        # 4. Load Contralateral Image
        if has_contra:
            contra_row = self.df.iloc[contra_idx]
            contra_path = os.path.join(Config.INPUT_DIR, contra_row["file_path"])
            try:
                contra_img = load_image(contra_path, size=Config.IMG_SIZE)
            except (FileNotFoundError, ValueError):
                # If metadata says it exists but file is missing -> Fail Loudly
                raise FileNotFoundError(f"Contralateral image missing: {contra_path}")
        else:
            # Substitute zero tensor (H, W)
            # Note: load_image returns float32 [0, 1]
            contra_img = np.zeros(Config.IMG_SIZE, dtype=np.float32)

        # 5. Construct 3-Channel Inputs
        # Target
        target_input = self._construct_input(target_img, row["age"], row["implant"])

        # Contralateral
        # Even if image is missing (zeros), we construct the tensor.
        # If we want strict zero-tensor substitution for missing pairs as per prompt:
        if has_contra:
            contra_input = self._construct_input(contra_img, row["age"], row["implant"])
        else:
            # "substitute a zero-tensor"
            # This implies (3, H, W) are all zeros.
            contra_input = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.float32
            )

        # 6. Apply Synchronized Augmentation
        # Albumentations expects inputs as named arguments
        augmented = self.transforms(image=target_input, image_contra=contra_input)

        target_tensor = augmented["image"]  # (3, H, W)
        contra_tensor = augmented["image_contra"]  # (3, H, W)

        # 7. Prepare Output
        sample = {
            "target": target_tensor,
            "contra": contra_tensor,
            "patient_id": patient_id,
            "view": view,
            "laterality": laterality,
        }

        # Add Label if available
        if "cancer" in row:
            sample["label"] = torch.tensor(row["cancer"], dtype=torch.float32)

        # Add Prediction ID for submission
        if "prediction_id" in row:
            sample["prediction_id"] = row["prediction_id"]
        else:
            # Fallback for train/val if needed for tracking
            sample["prediction_id"] = f"{patient_id}_{laterality}"

        return sample
