import os
import torch


class Config:
    """
    Configuration class for the Question Answering task using MuRIL.
    Centralizes all hyperparameters, paths, and settings for the Cross-Validated Ensemble.
    """

    # =========================================================================
    # 1. Experiment Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode

    # =========================================================================
    # 2. Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Input Files (using metadata as required to prevent leakage)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_CSV = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # 3. Model Architecture
    # =========================================================================
    # Using MuRIL (Multilingual Representations for Indian Languages)
    MODEL_CHECKPOINT = "google/muril-base-cased"

    # =========================================================================
    # 4. Data Preprocessing (Sliding Window)
    # =========================================================================
    # Sliding window strategy parameters to handle long contexts
    MAX_LENGTH = 384  # Max sequence length for the model input
    DOC_STRIDE = 128  # Overlap between consecutive windows

    # =========================================================================
    # 5. Training Hyperparameters
    # =========================================================================
    N_FOLDS = 3  # Number of folds for Group K-Fold Cross-Validation
    EPOCHS = 3  # Number of training epochs per fold

    # Batch sizes (Optimized for A100 40GB GPU)
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32

    # Optimizer settings
    LEARNING_RATE = 3e-5  # Typical LR for BERT-base models
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1  # Percentage of steps for linear warmup
    MAX_GRAD_NORM = 1.0  # Gradient clipping

    # Scheduler
    LR_SCHEDULER_TYPE = "linear"

    # Mixed Precision Training
    FP16 = True

    # =========================================================================
    # 6. Inference & Post-processing
    # =========================================================================
    N_BEST_SIZE = 20  # Number of top start/end logits to consider during decoding
    MAX_ANSWER_LENGTH = (
        30  # Maximum allowed length (in tokens) for the predicted answer
    )

    # =========================================================================
    # 7. System / Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        # Create working directory if it doesn't exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # Print basic configuration status (useful for log verification)
        print(f"Config initialized.")
        print(f"  Working Dir: {self.WORKING_DIR}")
        print(f"  Device: {self.DEVICE}")
        print(f"  Model: {self.MODEL_CHECKPOINT}")
        print(f"  Folds: {self.N_FOLDS}, Epochs: {self.EPOCHS}")
        print(f"  Debug Mode: {self.DEBUG}")
