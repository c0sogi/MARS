import os
import torch
import random
import numpy as np

# Ensure the output directory exists as required
os.makedirs("./working/idea_7", exist_ok=True)


class Config:
    # =======================
    # General Configuration
    # =======================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    OUTPUT_DIR = "./working/idea_7"

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =======================
    # Model Architecture
    # =======================
    BACKBONE = "tf_efficientnetv2_m"
    PRETRAINED = True
    NUM_CLASSES = 1  # Regression output (scalar)

    # Pooling
    USE_GEM_POOLING = True
    GEM_P = 3.0  # Power for Generalized Mean Pooling

    # =======================
    # Training Strategy (Progressive Resizing)
    # =======================
    # Phase 1: Coarse Feature Learning (70% of training)
    PHASE_1_RES = 384
    PHASE_1_EPOCHS = 14
    PHASE_1_BATCH_SIZE = 32  # Larger batch size for lower resolution

    # Phase 2: Fine-Grained Refinement (30% of training)
    PHASE_2_RES = 512
    PHASE_2_EPOCHS = 6
    PHASE_2_BATCH_SIZE = 16  # Smaller batch size for high resolution

    # General Training Params
    BATCH_SIZE = 16  # Default fallback
    NUM_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =======================
    # Optimization
    # =======================
    LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Stochastic Weight Averaging (SWA) - Active in Phase 2
    USE_SWA = True
    SWA_LR = 1e-5

    # =======================
    # Data Augmentation
    # =======================
    CLAHE_PROB = 0.5
    ROTATION_LIMIT = 30

    # =======================
    # Inference
    # =======================
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip)


def seed_everything(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seeds immediately upon import
seed_everything(Config.SEED)
