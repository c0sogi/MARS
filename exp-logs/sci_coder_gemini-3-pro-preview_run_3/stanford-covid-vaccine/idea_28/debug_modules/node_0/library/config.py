import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the settings for the 'Deep Post-Norm BiGRU with Zero-Masked Channel-Gating' strategy.
    """

    # =========================================================================
    # System & Environment
    # =========================================================================
    PROJECT_NAME = "idea_28"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Pre-split and stratified)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Dimensions & Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Feature Channels (One-Hot Encoding)
    # 4 (Nucleotides: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Architecture
    # Strategy: Deep Post-Norm BiGRU with Zero-Masked Channel-Gating
    # =========================================================================
    HIDDEN_DIM = 384  # Balanced capacity and trainability
    NUM_LAYERS = 4  # Deep backbone
    CNN_FILTERS = 256  # Convolutional stem filters
    CNN_KERNEL_SIZE = 3  # Local motif extraction
    DROPOUT = 0.1  # Regularization

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # For AdamW
    GRAD_CLIP_NORM = 1.0  # Mandatory for deep hybrid stability
    EPOCHS = 50  # Sufficient for convergence with early stopping
    PATIENCE = 10  # Early stopping patience

    # Scheduler Settings (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    DEBUG = False  # Toggle for rapid testing on subsets
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode
