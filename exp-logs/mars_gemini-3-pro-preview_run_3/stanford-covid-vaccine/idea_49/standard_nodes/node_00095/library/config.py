import os
import torch


class Config:
    """
    Configuration for the Deep Stabilized Bias-Refined Decoupled BiGRU strategy.
    Encapsulates file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to prevent conflicts
    WORKING_DIR = "./working/idea_49"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format for efficiency)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission File
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (for deterministic data processing)
    # Using .npy for fast loading of preprocessed tensors
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Input Channels: 4 (Nucleotides) + 3 (Structure) + 7 (Loop Types)
    # Strictly One-Hot Encoding as per strategy
    INPUT_CHANNELS = 14

    # Backbone Architecture
    # Hidden dimension maximized to 384 for capacity (Lesson 26)
    HIDDEN_DIM = 384
    # Deep backbone with 4 layers (Lesson 26, 68)
    N_LAYERS = 4

    # Convolutional Stem
    CONV_KERNEL = 3
    CONV_FILTERS = 256

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # Data Dimensions
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization & Scheduling
    MAX_EPOCHS = 50
    PATIENCE = 5  # Early stopping patience

    # Stability Control
    # Mandatory gradient clipping for 4-layer hybrid architecture
    CLIP_GRAD = 1.0

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Columns and Targets
    # ==========================================
    # All 5 targets are predicted during training (Multi-Task Learning)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Only these 3 are used for the official MCRMSE metric calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50
