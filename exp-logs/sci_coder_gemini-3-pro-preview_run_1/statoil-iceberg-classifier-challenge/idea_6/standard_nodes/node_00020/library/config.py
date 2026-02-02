import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Create working directory for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    IMG_SIZE = 224  # Upsampled size for ResNet
    CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)
    NUM_FOLDS = 5  # Stratified K-Fold

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01  # For AdamW
    NUM_EPOCHS = 30
    PATIENCE = 5  # Early Stopping Patience
    LABEL_SMOOTHING = 0.0

    # Scheduler Settings (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 2

    # =========================================================================
    # Hardware & Debugging
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flags to control dataset size for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
