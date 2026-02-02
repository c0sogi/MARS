import os
import torch


class Config:
    """
    Configuration class for Asymmetric Parallel Vector-DCN-ResNet with Manifold Cluster Augmentation.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True for quick debugging runs

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for this specific experimental idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_45")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Paths (Metadata)
    # --------------------------------------------------------------------------
    # Using the pre-generated metadata parquet files for efficiency
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample submission for format reference
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --------------------------------------------------------------------------
    # Feature Engineering Parameters
    # --------------------------------------------------------------------------
    # Manifold Cluster Augmentation
    N_CLUSTERS = 16

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Asymmetric Parallel Vector-DCN-ResNet
    HIDDEN_SIZE = 512
    DCN_LAYERS = 3  # Shallow asymmetric branch
    RES_BLOCKS = 4  # Deep backbone
    DROPOUT_RATE = 0.2

    # Target Information
    # Cover_Type has 7 classes (1-7). We typically map these to 0-6 indices or use 7 outputs.
    # The dataset analysis confirms classes 1, 2, 3, 4, 6, 7 are present (5 is rare/missing).
    NUM_CLASSES = 7

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimization (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MODE = "max"  # We monitor validation accuracy

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # --------------------------------------------------------------------------
    # Hardware & System
    # --------------------------------------------------------------------------
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Performance vs Determinism
    # Strategy calls for disabling strict determinism to maximize kernel performance
    CUDNN_DETERMINISTIC = False
    CUDNN_BENCHMARK = True

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories for cache and submission.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Print confirmation
        print(f"Configuration Setup Complete.")
        print(f"  Cache Dir: {cls.CACHE_DIR}")
        print(f"  Device: {cls.DEVICE}")
