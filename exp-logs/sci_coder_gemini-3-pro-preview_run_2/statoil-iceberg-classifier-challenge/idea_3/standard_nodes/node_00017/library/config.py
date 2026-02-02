import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Central configuration for the Residual Hybrid Transfer Network (RHTN) pipeline.
    """

    # ==========================================
    # 1. GENERAL SETTINGS
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==========================================
    # 2. PATHS
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 3
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Files (Raw)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Generated)
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    # Cache file for processed tensors to avoid re-processing JSONs
    PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_data.npz")
    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "rhtn_model.pth")
    # Final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. DATA PARAMETERS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: Band 1, Band 2, Mean(Band 1, Band 2)
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary: 0=Ship, 1=Iceberg

    # Normalization (Min-Max scaling is handled in data loader,
    # but we can define target range here)
    SCALE_MIN = 0.0
    SCALE_MAX = 1.0

    # ==========================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================
    BACKBONE_NAME = "resnet18"
    PRETRAINED = True

    # Feature Dimensions
    BACKBONE_OUT_DIM = 512  # ResNet18 default after GAP
    METADATA_INPUT_DIM = 1  # inc_angle
    METADATA_HIDDEN_DIM = 64

    # Fusion Head
    FUSION_HIDDEN_DIM = 256
    DROPOUT_RATE = 0.25  # Moderate dropout as per idea

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4

    # Early Stopping
    PATIENCE = 6

    # Learning Rate Scheduler (ReduceLROnPlateau)
    LR_FACTOR = 0.5
    LR_PATIENCE = 2
    LR_MIN = 1e-6

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of subprocesses for data loading

    @classmethod
    def setup(cls):
        """
        Create necessary directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seed immediately upon setup
        set_seed(cls.SEED)


# Execute setup when module is imported
Config.setup()
