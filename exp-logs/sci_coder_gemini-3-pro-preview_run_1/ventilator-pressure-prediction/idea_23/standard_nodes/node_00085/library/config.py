import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_NAME = "idea_23"

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-split)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Cache & Models)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CACHE_DIR = WORKING_DIR  # Using experiment dir as cache dir
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data & Feature Engineering
    # =========================================================================
    # Column Definitions
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Raw Controls & Attributes
    U_IN_COL = "u_in"
    U_OUT_COL = "u_out"
    R_COL = "R"
    C_COL = "C"

    # Feature Engineering Parameters
    USE_LAGS = True
    LAG_STEPS = [1, 2, 3, 4]
    USE_DIFFS = True
    USE_CUMULATIVE_VOLUME = True  # Integral of u_in * dt
    USE_INTERACTIONS = True  # R*u_in, Volume/C, etc.

    # Scaling
    # u_out is binary and must strictly NOT be scaled
    # All other continuous inputs will be RobustScaled

    # =========================================================================
    # Model Architecture: Graduated-Capacity Physics-Context Composite Network
    # =========================================================================
    # Stem (Bottleneck Initialization)
    STEM_KERNEL_SIZES = [3, 5, 7]  # Multi-scale 1D CNN
    BOTTLENECK_DIM = 512  # Compressed initialization space

    # Backbone (Wide-State Identity Blocks)
    WIDE_DIM = 1024  # High-capacity state space
    LSTM_HIDDEN_DIM = 512  # Per direction (Bidirectional -> 1024)
    NUM_BLOCKS = 4  # Block 1 (Expansion) + Blocks 2-4 (Identity)

    # Aux Head
    USE_AUX_HEAD = True
    AUX_HEAD_BLOCK_IDX = 1  # 0-indexed, attached after Block 2 output

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 512  # Fixed budget
    EPOCHS = 35  # Extended horizon for OneCycle annealing

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.3  # Percentage of cycle to increase LR
    DIV_FACTOR = 25.0  # Initial LR = Max LR / 25
    FINAL_DIV_FACTOR = 1e4  # Final LR = Initial LR / 10000

    # Loss
    AUX_LOSS_WEIGHT = 0.3  # Weight for auxiliary supervision

    # Gradient Stability
    GRAD_CLIP = 1.0  # Strict clipping for LSTM stability

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
