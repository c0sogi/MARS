import os
import torch


class Config:
    """
    Configuration class for the Context-Injected Over-Parameterized Dual-Stream Network.
    """

    # ====================================================
    # Paths & Directories
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Working directories for the specific idea
    WORKING_DIR = "./working/idea_39"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ====================================================
    # Data Preprocessing & Augmentation
    # ====================================================
    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Image specifications
    IMG_SIZE = 260
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices

    # Tabular features
    # Note: 'Percent' is explicitly excluded to prevent proxy leakage
    TABULAR_COLS = ["Weeks", "Age", "Sex", "SmokingStatus"]

    # ====================================================
    # Model Architecture
    # ====================================================
    BACKBONE = "efficientnet_b2"
    HIDDEN_DIM = 128
    OUT_DIM = 2  # Predicts Mean (FVC) and Uncertainty (Sigma)
    DROPOUT = 0.0  # Explicitly rejected based on solution lessons

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    SEED = 42
    BATCH_SIZE = 32  # Strictly >= 32
    EPOCHS = 30

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3

    # Optimization
    WEIGHT_DECAY = 0.01
    T_MAX = 30  # Linked to EPOCHS for Cosine Annealing

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================================================
    # Metric & Loss Constants
    # ====================================================
    # Metric calculation constants
    MAX_ERROR = 1000
    MIN_SIGMA = 70

    # ====================================================
    # Debugging
    # ====================================================
    DEBUG = False  # Set to True to run on a small subset for testing
