import os
import torch


class Config:
    """
    Centralized configuration for the Self-Normalizing Funnel Network (SNN) pipeline.
    Includes file paths, model architecture hyperparameters, and training settings.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a smaller subset for debugging
    DEBUG_SAMPLES = 10000  # Number of samples to use if DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Caching Paths (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")
    PREPROCESSOR_CACHE = os.path.join(WORKING_DIR, "preprocessor_meta.npy")

    # ==========================================
    # Model Architecture (SNN)
    # ==========================================
    # Entity Embeddings for categorical features
    EMBEDDING_DIM = 16

    # Funnel Network Topology (decreasing width)
    HIDDEN_LAYERS = [512, 256, 128]

    # Self-Normalizing specific settings
    # Alpha Dropout maintains mean/variance for SELU activations
    DROPOUT_RATE = 0.05

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 30

    # Optimization
    # Using AdamW with calibrated weight decay for tabular data
    LEARNING_RATE = 1e-3  # Max LR for OneCycle Policy
    WEIGHT_DECAY = 1e-5

    # Scheduler
    SCHEDULER_TYPE = "OneCycleLR"
    PCT_START = 0.3  # Percentage of training to increase LR
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # Compute & Hardware
    # ==========================================
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if torch.cuda.is_available() else False

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior for PyTorch if needed
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
