import os
import torch


class Config:
    """
    Configuration for the Stabilized Decoupled Bias-Refined BiGRU (SDBR-BiGRU) strategy.

    Strategy Overview:
    - Architecture: 1D Conv Stem -> 3-Layer BiGRU with Decoupled Structural Interaction -> Linear Head.
    - Key Innovation: Decoupled Gating with Bias-Driven Refinement for unpaired bases.
    - Stability: Internal LayerNorm in MLP gates, Gradient Clipping (1.0).
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    SUBSET_SIZE = (
        None  # If set (e.g., 100), limits the dataset size. None = Full dataset.
    )

    # ==========================================
    # Data Paths
    # ==========================================
    # Metadata files (Pre-stratified)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Raw input (for submission format reference)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_50"
    SUBMISSION_DIR = "./submission"

    # Specific Output Files
    # Caching paths for deterministic data processing
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # Model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    # Final submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Dimensions & Features
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Channels: 4 (Nucleotide) + 3 (Structure) + 7 (Loop Type) = 14
    # Strictly One-Hot Encoding as per strategy
    INPUT_DIM = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these columns are used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Feature Mappings
    TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
    TOKEN_MAP_STRUCT = {"(": 0, ")": 1, ".": 2}
    TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Backbone: Stabilized 3-Layer Backbone (Lesson 68)
    NUM_LAYERS = 3

    # Hidden Dimension: 384 (Lesson 63)
    HIDDEN_DIM = 384

    # Convolutional Stem
    KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32  # Adjusted for A100 memory and stability
    EPOCHS = 25  # Sufficient for convergence with Cosine Annealing
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping: Mandatory 1.0 for hybrid architecture stability (Lesson 46)
    MAX_GRAD_NORM = 1.0

    # Optimization
    OPTIMIZER_NAME = "AdamW"
    SCHEDULER_NAME = "CosineAnnealing"
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
