import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config
from library.preprocessing import process_image


class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for Breast Cancer Detection.
    Loads images on-the-fly using the pipeline defined in library.preprocessing.
    """

    def __init__(self, df: pd.DataFrame, mode: str = "train"):
        """
        Args:
            df: DataFrame containing metadata (file_path, cancer, prediction_id, etc.)
            mode: 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode

        # Pre-construct full file paths to avoid overhead in __getitem__
        # Config.INPUT_DIR is "./input", file_path is like "train_images/..."
        self.file_paths = [
            os.path.join(Config.INPUT_DIR, fp) for fp in self.df["file_path"].values
        ]

        # Pre-load labels or IDs
        if self.mode != "test":
            # Ensure labels are float32 for BCEWithLogitsLoss
            self.labels = self.df["cancer"].values.astype(np.float32)
        else:
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image Path
        path = self.file_paths[idx]

        # 2. Process Image (Load -> ROI Crop -> Resize -> Norm -> Tensor)
        # process_image handles errors and returns a zero-tensor if loading fails
        image = process_image(path)

        # 3. Return based on mode
        if self.mode != "test":
            # Wrap label in tensor (1,)
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return image, label
        else:
            # Return prediction_id for submission mapping
            pred_id = self.prediction_ids[idx]
            return image, pred_id


def get_dataloaders(
    train_path=Config.TRAIN_META_PATH,
    val_path=Config.VAL_META_PATH,
    test_path=Config.TEST_META_PATH,
):
    """
    Constructs DataLoaders for Train, Validation, and Test sets.
    Implements WeightedRandomSampler for Training to ensure balanced batches.
    """

    # 1. Load Metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 2. Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Subsetting data to {Config.DEBUG_SUBSET_SIZE} samples.")
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)
        df_test = df_test.head(Config.DEBUG_SUBSET_SIZE)

    # 3. Initialize Datasets
    train_dataset = BreastCancerDataset(df_train, mode="train")
    val_dataset = BreastCancerDataset(df_val, mode="val")
    test_dataset = BreastCancerDataset(df_test, mode="test")

    # 4. Setup WeightedRandomSampler for Training
    # We want to balance the batches 50/50 (Config.POS_RATIO = 0.5)
    # Calculate weights based on inverse class frequency
    targets = df_train["cancer"].values.astype(int)
    class_counts = np.bincount(targets)

    # Handle potential edge case where a class is missing (unlikely but safe)
    class_counts = np.maximum(class_counts, 1)

    # Weight = 1 / Count
    class_weights = 1.0 / class_counts

    # Assign weight to each sample corresponding to its class
    sample_weights = class_weights[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch for BatchNorm stability
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
