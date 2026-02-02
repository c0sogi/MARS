import os
import torch


class Config:
    """
    Configuration for the Physically-Structured Additive Hybrid Network (PSA-Net).
    Defines hyperparameters, file paths, and data settings.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of breaths for rapid testing
    DEBUG = False
    DEBUG_SAMPLES = 2000  # Number of breaths to use if DEBUG is True

    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    # Input Metadata (assumed to be generated in ./metadata)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching (Idea 9: CAP-Net Optimized)
    WORKING_DIR = "./working/idea_9"

    # Cache File Paths (npy format for fast loading)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_x.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_x.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_x.npy")
    # We also cache u_out separately or as part of X to handle masking easily
    CACHE_TRAIN_UOUT = os.path.join(WORKING_DIR, "train_u_out.npy")
    CACHE_VAL_UOUT = os.path.join(WORKING_DIR, "val_u_out.npy")
    CACHE_TEST_UOUT = os.path.join(WORKING_DIR, "test_u_out.npy")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------------------------
    # Optimization
    BATCH_SIZE = 128  # "Small Batch Size" strategy for gradient noise regularization
    EPOCHS = 60  # Increased epochs for convergence with small batch size
    LEARNING_RATE = 1e-3  # Standard AdamW starting rate
    WEIGHT_DECAY = 1e-2  # Regularization
    MAX_GRAD_NORM = 1000.0  # Gradient clipping

    # Scheduler
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    EARLY_STOPPING_PATIENCE = 15

    # Model Architecture (PSA-Net)
    HIDDEN_DIM = 256  # High capacity for LSTM branch
    LSTM_LAYERS = 3  # Deep Bidirectional LSTM
    TCN_KERNEL_SIZE = 5  # Wide kernel for Resistive branch
    TCN_CHANNELS = [64, 128, 256]  # Pyramidal channel scaling
    DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Data & Feature Configuration
    # --------------------------------------------------------------------------
    # Time steps per breath (fixed for this dataset)
    SEQ_LEN = 80

    # Raw columns to read from CSV
    RAW_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    TEST_RAW_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    # Final Input Features for the Model
    # These correspond to the "Deep Contextual Injection" strategy
    FEATURE_COLS = [
        "u_in",  # Control input (Proportional)
        "u_in_cumsum",  # Integral approximation (Volume)
        "u_in_diff1",  # 1st Derivative (Flow/Acceleration)
        "u_in_diff2",  # 2nd Derivative (Jerk)
        "R",  # Lung Resistance
        "C",  # Lung Compliance
        "R_u_in",  # Physics Interaction: Resistive Pressure proxy (R * u_in)
        "vol_C",  # Physics Interaction: Elastic Pressure proxy (Volume / C)
        "u_out",  # Phase indicator (Expiratory=1), used for masking
    ]

    # Input dimension for the model
    INPUT_DIM = len(FEATURE_COLS)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
