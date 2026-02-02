import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-v3-Large preference model.
    Centralizes paths, model architecture settings, and training hyperparameters.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Toggle for debugging on a small subset
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Original Input (for reference or sample submission structure)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    # We use idea_15 to isolate this experiment's artifacts
    WORKING_DIR = "./working/idea_15"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")
    SUBMISSION_DIR = "./submission"

    # Final Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Max length for each branch (Prompt + Response)
    # 512 is a balance between context retention and memory usage for Large models
    MAX_LENGTH = 512

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Gradient Accumulation Strategy:
    # A100 40GB can typically handle physical BS=4 for DeBERTa-Large @ 512 seq len.
    # We accumulate gradients to achieve a stable Effective Batch Size of 64.
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8
    TARGET_EFFECTIVE_BATCH_SIZE = 64

    # Dynamic calculation of accumulation steps
    GRAD_ACCUM_STEPS = max(1, TARGET_EFFECTIVE_BATCH_SIZE // TRAIN_BATCH_SIZE)

    EPOCHS = 3
    LEARNING_RATE = 5e-6  # Lower LR for Large backbone to prevent divergence
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    WARMUP_RATIO = 0.1
    PATIENCE = 2  # For Early Stopping

    # =========================================================================
    # Hardware & Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    USE_FP16 = True  # Mixed Precision Training

    # =========================================================================
    # Caching Configuration
    # =========================================================================
    LOAD_CACHED_DATA = True  # Attempt to load processed data from CACHE_DIR

    @classmethod
    def setup(cls):
        """
        Initialize the workspace by creating necessary directories.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Optional: Print config status (silent for submission, useful for logs)
        # print(f"Config: {cls.MODEL_NAME} on {cls.DEVICE}")
        # print(f"Batch Size: {cls.TRAIN_BATCH_SIZE} (Physical) * {cls.GRAD_ACCUM_STEPS} (Accum) = {cls.TARGET_EFFECTIVE_BATCH_SIZE} (Effective)")


# Execute setup on import to guarantee directory existence
Config.setup()
