import os
import random
import numpy as np
import torch
import pandas as pd
from skmultilearn.model_selection import IterativeStratification


class Config:
    def __init__(self, debug=False, epochs=60, batch_size=32):
        """
        Configuration class for the Bird Species Classification task.

        Args:
            debug (bool): If True, subsets the data for rapid debugging.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        self.DEBUG = debug
        self.SEED = 42

        # Directory Paths
        self.INPUT_ROOT = "./input"
        self.ESSENTIAL_DATA = os.path.join(self.INPUT_ROOT, "essential_data")
        self.SUPPLEMENTAL_DATA = os.path.join(self.INPUT_ROOT, "supplemental_data")
        # Strictly use standard spectrograms as per strategy
        self.SPECTROGRAM_DIR = os.path.join(self.SUPPLEMENTAL_DATA, "spectrograms")
        self.METADATA_DIR = "./metadata"

        # Output Directory
        self.OUTPUT_DIR = "./working/idea_23"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # Data Parameters
        self.IMAGE_SIZE = (224, 224)
        self.NUM_CLASSES = 19
        self.N_FOLDS = 5
        self.BATCH_SIZE = 16 if debug else batch_size
        self.NUM_WORKERS = 2

        # Model Architecture Ensemble
        self.ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]
        self.PRETRAINED = True

        # Training Hyperparameters
        self.EPOCHS = 5 if debug else epochs
        self.PATIENCE = 15  # Aggressive early stopping
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-2

        # Optimization Strategy
        self.OPTIMIZER_NAME = "Lookahead_AdamW"
        self.LOOKAHEAD_K = 5
        self.LOOKAHEAD_ALPHA = 0.5
        self.SCHEDULER_NAME = "CosineAnnealingLR"
        self.MIN_LR = 1e-6

        # Regularization & Augmentation
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = 0.4  # Reduced alpha for micro-dataset

        # Augmentation Constraints
        self.BRIGHTNESS_LIMIT = 0.2
        self.CONTRAST_LIMIT = 0.2
        self.SHIFT_LIMIT = 0.1  # Horizontal translation via zero-padding
        self.NO_HFLIP = True  # Strictly disable horizontal flipping

        # Inference Strategy
        self.TOP_K_CHECKPOINTS = 3  # Snapshot ensemble
        self.TTA_STEPS = 3  # Original, Left Shift, Right Shift

    def __repr__(self):
        return str(self.__dict__)


def set_seed(seed=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_data_splits(config, load_cached_data=True):
    """
    Loads and processes dataset metadata.

    Implements a caching mechanism using Parquet files.
    Recombines the provided 'train' and 'val' splits to perform a fresh
    5-fold Iterative Stratification, ensuring robust cross-validation.

    Args:
        config (Config): Configuration object containing paths and settings.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        folds_df (pd.DataFrame): Combined training data with 'kfold' column.
        test_df (pd.DataFrame): Test data.
    """
    cache_dir = config.OUTPUT_DIR
    folds_cache_path = os.path.join(cache_dir, "folds.parquet")
    test_cache_path = os.path.join(cache_dir, "test.parquet")

    # 1. Attempt to load from cache
    if (
        load_cached_data
        and os.path.exists(folds_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached metadata from {cache_dir}...")
        folds_df = pd.read_parquet(folds_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        return folds_df, test_df

    print("Processing metadata from scratch...")

    # 2. Load Source Metadata
    train_source = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
    val_source = pd.read_csv(os.path.join(config.METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(config.METADATA_DIR, "test.csv"))

    # 3. Combine Train and Val for Stratification
    # We ignore the 'fold' column in source as we want to create 5 new folds
    dev_df = pd.concat([train_source, val_source], ignore_index=True)

    if config.DEBUG:
        dev_df = dev_df.sample(
            n=min(len(dev_df), 50), random_state=config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), 20), random_state=config.SEED
        ).reset_index(drop=True)

    # 4. Prepare Labels for Iterative Stratification
    # Convert label string "0 4" to binary matrix
    X = dev_df["rec_id"].values.reshape(-1, 1)

    # Parse labels to list of ints
    def parse_labels(label_str):
        if pd.isna(label_str) or label_str == "?" or str(label_str).strip() == "":
            return []
        return [int(x) for x in str(label_str).split()]

    dev_df["label_list"] = dev_df["labels"].apply(parse_labels)

    # Create binary matrix
    y = np.zeros((len(dev_df), config.NUM_CLASSES))
    for idx, labels in enumerate(dev_df["label_list"]):
        for lbl in labels:
            if 0 <= lbl < config.NUM_CLASSES:
                y[idx, lbl] = 1

    # 5. Perform Iterative Stratification
    stratifier = IterativeStratification(n_splits=config.N_FOLDS, order=1)

    # Initialize kfold column
    dev_df["kfold"] = -1

    # IterativeStratification returns train_indices, test_indices for each fold
    # We assign the fold number to the test_indices
    for fold_idx, (_, test_idx) in enumerate(stratifier.split(X, y)):
        dev_df.loc[test_idx, "kfold"] = fold_idx

    # Drop temporary column
    dev_df = dev_df.drop(columns=["label_list"])

    # 6. Cache Results
    print(f"Saving processed metadata to {cache_dir}...")
    # Parquet handles list/string columns efficiently
    dev_df.to_parquet(folds_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return dev_df, test_df
