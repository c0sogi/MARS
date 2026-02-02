import os
import torch


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering Task.
    Implements settings for a Multi-Task XLM-R Large architecture with
    Full-Data Seed Ensembling.
    """

    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Input Files (using metadata for correct splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # MODEL ARCHITECTURE
    # =========================================================================
    MODEL_NAME = "xlm-roberta-large"

    # =========================================================================
    # DATA PROCESSING (SLIDING WINDOW & SAMPLING)
    # =========================================================================
    # Tokenizer parameters
    MAX_LEN = 384
    DOC_STRIDE = 128

    # Negative Sampling Strategy
    # We retain 100% of positive windows (containing answer) and sample negatives
    # to maintain a specific ratio.
    # Ratio = Negatives / Positives
    NEGATIVE_POSITIVE_RATIO = 2.0

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    # Ensemble Strategy: Train 5 independent models on the full dataset
    SEEDS = [42, 43, 44, 45, 46]

    # Batch Sizes
    # Small batch size acts as implicit regularization for the large model
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8

    # Duration
    # Extended epochs to allow convergence with lower learning rates
    EPOCHS = 8

    # Optimization: Differential Learning Rates (DLR)
    # Lower rate for the pre-trained backbone to preserve multilingual features (Stability)
    LR_BACKBONE = 1e-5
    # Higher rate for the randomly initialized heads to learn the task (Plasticity)
    LR_HEADS = 5e-5

    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Multi-Task Loss Weights
    # Total Loss = Span_Loss + (RELEVANCE_LOSS_WEIGHT * Relevance_Loss)
    RELEVANCE_LOSS_WEIGHT = 1.0

    # =========================================================================
    # SYSTEM / HARDWARE
    # =========================================================================
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # DEBUGGING / DEVELOPMENT
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    @staticmethod
    def setup():
        """
        Creates necessary directories for caching, checkpoints, and submission.
        Should be called at the start of the pipeline.
        """
        dirs = [
            Config.WORKING_DIR,
            Config.CACHE_DIR,
            Config.CHECKPOINT_DIR,
            Config.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Returns configuration as a dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
