import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-v3-Base model with
    Dual-Stream Multi-Layer Aggregation.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_13"

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    SUBMISSION_DIR = "./submission"

    # Specific subdirectories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")

    # Input File Paths (using metadata as source of truth)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output File Paths
    OUTPUT_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"

    # Input constraints
    MAX_LENGTH = 512

    # Architecture specifics
    HIDDEN_SIZE = 768
    NUM_CLASSES = 3  # Winner A, Winner B, Tie
    DROPOUT = 0.1

    # Dual-Stream Aggregation settings
    # Stream 1 (Content): Concatenate last N layers
    RESPONSE_LAYERS = 4
    # Stream 2 (Context): Use last N layers (usually 1 for simple context)
    CONTEXT_LAYERS = 1

    # Feature flags
    USE_SCALARS = True  # Include log-lengths of prompt/responses

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16

    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 3
    WARMUP_RATIO = 0.1

    # Gradient accumulation to simulate larger batches if needed
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0

    # Optimization flags
    USE_FP16 = True
    GRADIENT_CHECKPOINTING = True

    # Data Augmentation
    SYMMETRIC_AUGMENTATION = True  # Train on (A,B) and (B,A)

    # =========================================================================
    # Inference Settings
    # =========================================================================
    TTA = True  # Test-Time Augmentation: Predict (A,B) and (B,A) and average

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
