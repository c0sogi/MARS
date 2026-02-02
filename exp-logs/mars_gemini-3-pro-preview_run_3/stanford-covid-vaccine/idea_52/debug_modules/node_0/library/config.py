import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Paths
    # =========================================================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_52"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths (Parquet format)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Raw data paths (JSON) - kept for reference, though metadata is preferred
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache paths
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # Model save path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Final submission path
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input feature dimensions
    # 4 (nucleotides) + 3 (structure) + 7 (loop type) = 14
    INPUT_DIM = 14

    # Output targets
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    NUM_TARGETS = 5

    # Columns used for scoring in validation
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture (Deep Stabilized Bias-Refined Decoupled BiGRU)
    # =========================================================================
    # Convolutional Stem
    CONV_KERNEL_SIZE = 3
    CONV_FILTERS = 256

    # Backbone
    HIDDEN_DIM = 384  # High capacity within safe limits
    NUM_LAYERS = 4  # Deep 4-layer backbone
    DROPOUT = 0.1  # Standard regularization

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64  # Safe for A100 40GB with this seq len
    EPOCHS = 20  # Sufficient for convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization
    MAX_GRAD_NORM = 1.0  # Mandatory for 4-layer hybrid stability
    PATIENCE = 5  # Early stopping patience

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    @classmethod
    def print_config(cls):
        """Prints the configuration settings."""
        print("=" * 40)
        print(f"CONFIG: {cls.PROJECT_NAME}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.NUM_LAYERS} Layers, {cls.HIDDEN_DIM} Hidden Dim")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Max Grad Norm: {cls.MAX_GRAD_NORM}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("=" * 40)
