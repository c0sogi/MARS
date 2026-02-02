import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Stabilized GLU-Decoupled BiGRU strategy.
    Centralizes all hyperparameters, file paths, and execution settings.
    """

    # ==========================================
    # Experiment Identity & Reproducibility
    # ==========================================
    IDEA_NAME = "idea_89"
    SEED = 42

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Ensure working directory exists for outputs and cache
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Data Paths (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths (Numpy format for processed tensors)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==========================================
    # Data Processing & Dimensions
    # ==========================================
    LOAD_CACHED_DATA = True  # If True, attempts to load .npy files from WORKING_DIR
    DEBUG = False  # If True, runs on a small subset for debugging
    DEBUG_SUBSET_SIZE = 50  # Number of samples in debug mode

    SEQ_LENGTH = 107  # Total sequence length
    SEQ_SCORED = 68  # Number of positions scored

    # Input Feature Dimension
    # 4 (One-hot Nucleotide) + 3 (One-hot Structure) + 7 (One-hot Loop Type)
    INPUT_DIM = 14

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Backbone (BiGRU)
    HIDDEN_DIM = 384  # Dimension per direction (Total = 768)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    BIDIRECTIONAL = True
    DROPOUT = 0.1  # Conservative dropout

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32  # Optimized for A100 with large model
    EPOCHS = 50  # Maximum training epochs
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    MAX_GRAD_NORM = 1.0  # Gradient clipping threshold (Critical for stability)
    PATIENCE = 15  # Early stopping patience

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Targets
    # ==========================================
    # Full list of targets for Multi-Task Learning
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Subset of targets used for validation metric calculation
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Hardware Settings
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of data loader workers
