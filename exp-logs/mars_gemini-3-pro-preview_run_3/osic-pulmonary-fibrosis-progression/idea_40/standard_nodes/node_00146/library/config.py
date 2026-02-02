import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Context-Injected Dual-Stream Network (CIDS-Net).
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata file paths (generated in previous steps)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directories for the current idea iteration
    WORKING_DIR = "./working/idea_40"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data & Preprocessing
    # -------------------------------------------------------------------------
    # Image parameters
    IMG_SIZE = 260
    SLICES_PER_PATIENT = 3  # Anchor slice + 2 boundary slices

    # DICOM Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Feature Scaling
    TIME_SCALE = 0.01  # Scale relative weeks by 0.01

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True

    # Input Dimensions
    # Stream A (Clinical Anchor): Baseline FVC, Relative Time, Age, Sex, SmokingStatus
    # Note: Sex and Smoking are encoded, Age/FVC/Time are scalars.
    N_TABULAR_FEATURES = 5

    # Stream B (Visual Residual): Image Features + Baseline FVC + Relative Time
    # These are injected into the visual stream's MLP
    N_CONTEXT_FEATURES = 2

    HIDDEN_DIM = 128
    OUTPUT_DIM = 2  # Mean, Std (Uncertainty)
    DROPOUT = 0.0  # Explicitly excluded for residual stream as per Idea

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Hardware / Misc
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs

    # Debugging / Development flags
    DEBUG = False
    MAX_TRAIN_SAMPLES = None  # Set to an integer to limit training data for debugging


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
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


# Initialize reproducibility immediately upon import
setup_reproducibility()
