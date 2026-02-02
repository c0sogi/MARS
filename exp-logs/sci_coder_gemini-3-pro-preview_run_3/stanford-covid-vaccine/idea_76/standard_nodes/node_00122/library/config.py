import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Topology-Disentangled BiGRU strategy.
    Centralizes file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_76"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data paths (using pre-generated metadata Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Dimensions
    # 4 (Nucleotide: A, G, C, U) + 3 (Structure: ., (, )) + 7 (Loop: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture (HC-TD-BiGRU)
    # =========================================================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Backbone
    # High-Capacity: 384 hidden units per direction -> 768 total
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # Adjusted for A100 40GB
    EPOCHS = 50  # Sufficient time for convergence with early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization
    MAX_GRAD_NORM = 1.0  # Mandatory for stability as per strategy
    PATIENCE = 10  # Early stopping patience

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # =========================================================================
    # Runtime / Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
