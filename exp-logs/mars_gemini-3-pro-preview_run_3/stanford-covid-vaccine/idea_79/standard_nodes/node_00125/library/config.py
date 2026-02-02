import os
import torch


class Config:
    """
    Configuration for the Deep Residual High-Capacity GLU-BiGRU model.
    Centralizes file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_79"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Parquet)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Numpy .npz)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Constants
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Features:
    # 4 (A,G,U,C) + 3 (.,(,)) + 7 (Loop Types) = 14
    INPUT_DIM = 14

    # Target Columns
    # Order matches sample_submission.csv and metadata analysis
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # Indices of targets used for MCRMSE scoring:
    # reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    SCORED_TARGET_INDICES = [0, 1, 3]

    # Debugging: Set to an integer (e.g., 100) to train on a subset, or None for full data
    DEBUG_SUBSET_SIZE = None

    # =========================================================================
    # Model Architecture (High-Capacity Residual GLU-BiGRU)
    # =========================================================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Backbone: Deep BiGRU
    NUM_LAYERS = 4
    HIDDEN_DIM = 384  # Per direction (Total = 768)
    BIDIRECTIONAL = True

    # Interaction Module
    # Projects concatenated hidden states (768+768) -> Hidden -> Gate

    # Regularization
    DROPOUT = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Optimization
    EPOCHS = 25
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
