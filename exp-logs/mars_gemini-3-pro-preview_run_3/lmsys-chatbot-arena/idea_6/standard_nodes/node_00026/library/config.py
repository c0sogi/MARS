import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-v3-Large Chatbot Arena solution.
    Centralizes all hyperparameters, file paths, and environment settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_NAME = "idea_6"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Pre-split CSVs)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output / Working Directory
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    OUTPUT_DIR = WORKING_DIR  # Alias for compatibility

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (using Parquet for efficiency)
    CACHE_DIR = WORKING_DIR  # Store cache directly in working dir as per file structure
    TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_data.parquet")
    VAL_CACHE_PATH = os.path.join(CACHE_DIR, "val_data.parquet")
    TEST_CACHE_PATH = os.path.join(CACHE_DIR, "test_data.parquet")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Input sequence length per branch (Prompt + Response)
    # Total effective context is 2x MAX_LENGTH due to Siamese structure
    MAX_LENGTH = 512

    # Number of target classes: Winner A, Winner B, Tie
    NUM_CLASSES = 3

    # Architecture Flags
    USE_SIAMESE = True
    POOLING_TYPE = "attention_isolated"  # Options: 'mean', 'max', 'attention', 'attention_isolated'
    USE_SCALAR_FEATURES = True  # Include log-lengths of prompt/responses

    # =========================================================================
    # Data Processing
    # =========================================================================
    # Symmetric Augmentation: Flip (A, B) -> (B, A) with target (1-p)
    AUGMENT_DATA = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 2

    # Batch sizes optimized for A100 (40GB) with DeBERTa-Large
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8

    # Gradient Accumulation to simulate larger batch size (e.g., 4 * 4 = 16)
    ACCUMULATION_STEPS = 4

    # Optimizer settings
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Early Stopping
    PATIENCE = 3

    # =========================================================================
    # Hardware & Optimization
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Mixed Precision Training
    USE_FP16 = True

    # Gradient Checkpointing (Critical for Large models on limited VRAM)
    GRADIENT_CHECKPOINTING = True

    @classmethod
    def setup(cls):
        """
        Ensures all necessary output directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        # Cache dir is same as working dir, so already created


# Initialize directories upon module import
Config.setup()
