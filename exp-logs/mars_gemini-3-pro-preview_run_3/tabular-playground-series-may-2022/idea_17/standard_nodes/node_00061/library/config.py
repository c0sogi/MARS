import os
import torch


class Config:
    """
    Central configuration for the Tree-Structured Funnel Ensemble strategy.
    Defines global hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Input Data Paths (using metadata splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cached Data Paths
    TRAIN_PROCESSED_PARQUET = os.path.join(CACHE_DIR, "train_processed.parquet")
    VAL_PROCESSED_PARQUET = os.path.join(CACHE_DIR, "val_processed.parquet")
    TEST_PROCESSED_PARQUET = os.path.join(CACHE_DIR, "test_processed.parquet")
    VOCAB_SIZES_NPY = os.path.join(CACHE_DIR, "vocab_sizes.npy")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Entity Embeddings
    EMBED_DIM = 16

    # Architecture
    NUM_HEADS = 1
    TRUNK_HIDDEN_DIM = 512
    HEAD_HIDDEN_DIMS = [256, 128]  # Funnel structure: 512 -> 256 -> 128 -> 1
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-2  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-5  # Calibrated weight decay
    EPOCHS = 30  # Sufficient for OneCycle convergence
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and cache directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Automatically setup directories when the module is imported
Config.setup()
