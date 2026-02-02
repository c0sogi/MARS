import os
import torch


class Config:
    """
    Configuration for the Context-Aware Parallel Hybrid Network (CAP-Net) pipeline.
    Centralizes all file paths, hyperparameters, and feature engineering flags to
    ensure synchronization between data processing and model architecture.
    """

    # ==========================
    # File Paths & Directories
    # ==========================
    # Input Metadata (Pre-split CSVs)
    INPUT_DIR = "./metadata"
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    VAL_CSV = os.path.join(INPUT_DIR, "validation.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")

    # Working Directory for Artifacts (Cache, Models)
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Processing Parameters
    # ==========================
    # Dataset specifics
    BREATH_LEN = 80  # Fixed time steps per breath in this dataset

    # Feature Engineering Flags
    # These control the generation of the PID and Physics-informed features
    USE_PID = True  # Generate Integral (Area) and Derivative (Diff) of u_in
    USE_PHYSICS = True  # Generate R*u_in and Area/C interaction terms

    # Input Dimension Calculation
    # 1. Raw: time_step, u_in, u_out, R, C (5 features)
    # 2. PID: area (Integral), u_in_diff (Derivative) (2 features)
    # 3. Physics: R_u_in (Resistive), vol_C (Elastic) (2 features)
    # Total Input Dimension = 9
    INPUT_DIM = 9

    # Caching Logic
    CLEAN_START = (
        True  # If True, delete existing .npy/.pth files in WORKING_DIR at start
    )
    LOAD_CACHE = True  # If True, attempt to load pre-processed .npy files

    # ==========================
    # Model Architecture (CAP-Net)
    # ==========================
    # Branch 1: LSTM (Contextualized Elastic Stream)
    # Captures low-frequency, integral-based dynamics (P = V/C)
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True

    # Branch 2: TCN (Contextualized Resistive Stream)
    # Captures high-frequency, derivative-based dynamics (P = R*Flow)
    # List defines the number of channels for each dilated layer
    TCN_CHANNELS = [64, 128, 256, 512]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # Fusion Head
    # Combines outputs from both branches
    FC_HIDDEN_SIZE = 128
    DROPOUT = 0.1

    # ==========================
    # Training Hyperparameters
    # ==========================
    SEED = 42

    # Debugging
    DEBUG = False  # Set to True to train on a small subset for verification
    DEBUG_SIZE = 2000  # Number of breaths to use in debug mode

    # Optimization
    EPOCHS = 60  # Extended training budget as per strategy
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW optimizer

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
