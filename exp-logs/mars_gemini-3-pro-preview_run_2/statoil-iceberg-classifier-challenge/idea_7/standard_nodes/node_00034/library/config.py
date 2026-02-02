import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration and hyperparameters for the SEA-HN pipeline.
    Acts as a central source of truth for the project.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea/experiment
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Outputs
    PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_data.npz")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    IMAGE_SIZE = 75

    # Image Channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    IN_CHANNELS = 3

    # Statistical Features:
    # 3 Bands * 5 Stats (Mean, Std, Min, Max, Median) + 1 Incidence Angle
    NUM_STAT_FEATURES = 16

    # =========================================================================
    # Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 100
    PATIENCE = 15  # Early stopping patience
    DROPOUT_RATE = 0.2
    WEIGHT_DECAY = 1e-4

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Conservative worker count for data loading
    NUM_WORKERS = 4

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # If set to an integer, limits the number of samples for rapid testing
    MAX_SAMPLES = None

    @classmethod
    def make_dirs(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_deterministic(seed=42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
