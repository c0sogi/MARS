import os
import torch
import numpy as np
import random


def set_seed(seed=42):
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
    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Pre-split)
    INPUT_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    VAL_PATH = os.path.join(INPUT_DIR, "validation.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Original Input (for sample submission structure)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working Directory for Cache and Models
    # Specific to Idea 18 to avoid conflicts
    WORKING_DIR = "./working/idea_18"
    OUTPUT_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data & Feature Engineering Configuration
    # ==========================================
    # Columns expected in the raw CSV files
    ID_COL = "id"
    BREATH_COL = "breath_id"
    TARGET_COL = "pressure"

    # Configuration for the feature engineering pipeline
    FEATURE_CONFIG = {
        # Physical Integration
        "use_physics_area": True,  # Calculate Area = cumsum(u_in * dt)
        # Derivatives / PID State
        "use_derivatives": True,  # Calculate u_in_diff (acceleration)
        # Interaction Terms
        "use_interaction": True,  # Calculate R*u_in and Area/C
        # Explicit Lookahead (Zero-lag context)
        "lookahead_steps": 4,  # Generate u_in(t+1) ... u_in(t+4)
        "lookahead_diff_steps": 1,  # Generate u_in_diff(t+1)
        # Context
        "use_time_step": True,  # Explicitly retain time_step
        "use_R_C": True,  # Explicitly retain R and C columns
    }

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Training Regime: Noise-Injected Critical Mass
    BATCH_SIZE = 128  # Small batch size for gradient noise
    EPOCHS = 80  # Extended training for hybrid convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler Settings (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    EARLY_STOPPING_PATIENCE = 15

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use in debug mode

    # ==========================================
    # Model Architecture: DCLK-Net
    # ==========================================
    # Branch 1: Deep Dense Large-Kernel TCN (Resistive Stream)
    CNN_KERNEL_SIZE = 9  # Large Kernel (receptive field)
    CNN_DILATION = 1  # Dense (no holes/gridding)
    CNN_LAYERS = 6  # Deep stack
    CNN_CHANNELS_START = 64
    CNN_CHANNELS_MAX = 512

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_LAYERS = 3
    LSTM_HIDDEN = 512
    LSTM_BIDIRECTIONAL = True

    # General
    DROPOUT = 0.1

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader
