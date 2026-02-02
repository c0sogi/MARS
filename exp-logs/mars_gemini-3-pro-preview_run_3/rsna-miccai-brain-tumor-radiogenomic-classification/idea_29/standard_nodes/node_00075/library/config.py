import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the VAMS-HD (View-Adaptive Modality-Structured High-Density) Network.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Directory (Required for deterministic data processing)
    # Using 'idea_30' for this iteration
    CACHE_DIR = "./working/idea_30"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Output Directory for Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    # VAMS-HD Specifics:
    # 32 slices per modality * 4 modalities = 128 input channels
    NUM_SLICES_PER_MODALITY = 32
    NUM_MODALITIES = 4
    IN_CHANNELS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES

    IMG_SIZE = 224

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone: EfficientNet-B0
    BACKBONE_NAME = "efficientnet_b0"

    # Stem Configuration
    # Compresses 128 channels -> 64 channels before backbone
    STEM_OUT_CHANNELS = 64

    # Regularization
    DROP_PATH_RATE = 0.2
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Compute: 1 NVIDIA A100-SXM4-40GB GPU available
    # 128 channels * 224 * 224 is memory intensive, but A100 is large.
    BATCH_SIZE = 16

    # Optimization
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0  # Explicitly set to 0.0 as per strategy (rely on DropPath)
    PATIENCE = 5  # Early stopping patience

    # Hardware / Dataloader
    NUM_WORKERS = 12  # 12 vCPUs available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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
