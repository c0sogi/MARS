import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Paths
    # Using original train.csv for Group K-Fold Cross Validation to use all data
    ORIGINAL_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (provided for reference/validation splits)
    META_TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    META_VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    META_TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "google/muril-base-cased"
    TOKENIZER_NAME = "google/muril-base-cased"

    # Sliding Window / Input Settings
    MAX_LENGTH = 384
    DOC_STRIDE = 128
    PAD_TO_MAX_LENGTH = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_FOLDS = 3
    EPOCHS = 3
    BATCH_SIZE = 16  # Adjusted for A100 and MuRIL Base memory footprint

    # Optimization
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # =========================================================================
    # Post-Processing & Inference
    # =========================================================================
    N_BEST_SIZE = 20
    MAX_ANSWER_LENGTH = 30

    # =========================================================================
    # Compute
    # =========================================================================
    NUM_WORKERS = 4
    FP16 = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seeding immediately upon import
seed_everything(Config.SEED)
