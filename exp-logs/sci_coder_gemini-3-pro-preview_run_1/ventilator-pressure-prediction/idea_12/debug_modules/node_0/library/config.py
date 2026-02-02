import os
import random
import numpy as np
import torch


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


class Config:
    """
    Global configuration for the High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    VAL_PATH = os.path.join(INPUT_DIR, "val.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Working directory for caching intermediate files (scalers, processed tensors)
    WORKING_DIR = "./working/idea_12"

    # Output directory for the final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility & Compute
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Pipeline & Debugging
    # =========================================================================
    # Set DEBUG to True to run on a small subset of breaths for rapid testing
    DEBUG = False
    DEBUG_BREATHS = 2000

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 512
    EPOCHS = 35  # Extended horizon for OneCycleLR annealing

    # Optimizer settings (AdamW + OneCycleLR)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM
    HIDDEN_DIM = 512
    NUM_LAYERS = 4  # Number of Composite Blocks
    CNN_KERNELS = [3, 5, 7]  # Multi-scale stem kernels

    # Auxiliary Loss
    AUX_WEIGHT = 0.3  # Weight for the auxiliary head attached to Block 2
    AUX_BLOCK_INDEX = 1  # 0-indexed, so 1 is the 2nd block

    DROPOUT = 0.1

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # Sequence length is fixed by the dataset (approximately 3 seconds / 80 steps)
    SEQ_LEN = 80

    # List of all input features to be used by the model.
    # Includes raw control signals, static attributes, and engineered physics features.
    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        # Engineered Features
        "dt",  # Time delta
        "volume",  # Cumulative integration of flow
        "u_in_R",  # Interaction: u_in * R
        "vol_C",  # Interaction: volume / C
        "u_in_lag1",  # Lag features for system inertia
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_diff1",  # First derivative approximation
        "u_in_diff2",  # Second derivative approximation
    ]

    # Subset of features to be re-injected into the LSTM at each composite block
    # These represent the "Static Physics Features" and key state variables
    PHYSICS_COLS = ["R", "C", "u_in_R", "vol_C"]

    INPUT_DIM = len(FEATURE_COLS)
