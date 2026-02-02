import os


class Config:
    """
    Configuration for the Deep Stabilized Bias-Refined Decoupled BiGRU architecture.
    Centralizes hyperparameters, file paths, and strategy-specific settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for the specific strategy (Idea 47)
    WORKING_DIR = "./working/idea_47"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Parquet format for efficient loading of list columns)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features: 4 (Nucleotides A,G,C,U) + 3 (Structure .,(,)) + 7 (Loop Types)
    NUM_FEATURES = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these columns are used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Hyperparameters (Deep Stabilized Backbone)
    # =========================================================================
    # Hidden dimension for the BiGRU and interaction modules
    HIDDEN_DIM = 384

    # Number of stacked blocks (BiGRU + Structural Interaction)
    NUM_LAYERS = 4

    # Convolutional Stem settings
    KERNEL_SIZE = 3
    CONV_FILTERS = 256

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Batch size (A100 40GB can handle 32-64 easily with this seq_len)
    BATCH_SIZE = 32

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (Cosine Annealing)
    NUM_EPOCHS = 100
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 10

    # Stability: Mandatory Gradient Clipping for deep RNNs
    MAX_GRAD_NORM = 1.0
