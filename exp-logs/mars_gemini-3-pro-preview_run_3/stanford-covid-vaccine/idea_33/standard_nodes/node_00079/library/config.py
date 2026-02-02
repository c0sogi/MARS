import os
import torch


class Config:
    """
    Centralized configuration for the RNA Degradation Prediction task.
    Implements the settings for the Deep Input-Normalized Channel-Gated BiGRU strategy.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_33"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npz")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Feature Dimensions:
    # 4 Nucleotides (A, G, C, U) +
    # 3 Structure states ((, ), .) +
    # 7 Loop types (S, M, I, B, H, E, X)
    NUM_FEATURES = 14

    # Target Columns available in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone settings
    HIDDEN_DIM = 384
    NUM_LAYERS = 4

    # Convolutional Stem settings
    KERNEL_SIZE = 3
    FILTERS = 256

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10

    # Stability: Gradient Clipping is mandatory for the deep hybrid architecture
    GRADIENT_CLIP = 1.0

    # Scheduler parameters (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # System & Debugging
    # =========================================================================
    NUM_WORKERS = 4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
