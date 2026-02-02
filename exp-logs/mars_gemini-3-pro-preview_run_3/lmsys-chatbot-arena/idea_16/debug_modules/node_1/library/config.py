import os
import torch


class Config:
    """
    Global configuration for the Siamese DeBERTa-v3-Base Preference Prediction pipeline.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use in debug mode

    # =========================================================================
    # File Paths
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Data Paths (using metadata as requested)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache File Paths (Parquet for dataframes/features)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.parquet")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    NUM_LABELS = 3  # Winner A, Winner B, Tie
    HIDDEN_SIZE = 768  # Hidden size for DeBERTa-v3-base
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    TRAIN_BATCH_SIZE = 8  # Tuned for A100 40GB with 512 context length
    VALID_BATCH_SIZE = 16
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1
    EARLY_STOPPING_PATIENCE = 2

    # =========================================================================
    # Optimization & Hardware
    # =========================================================================
    FP16 = True  # Mixed precision training
    USE_GRADIENT_CHECKPOINTING = True  # Save memory for larger batch/seq len

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is a safe number for dataloaders

    @classmethod
    def setup_directories(cls):
        """
        Ensures that the working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately to ensure paths exist
Config.setup_directories()
