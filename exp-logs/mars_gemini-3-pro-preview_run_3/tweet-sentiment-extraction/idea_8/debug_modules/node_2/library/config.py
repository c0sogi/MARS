import os
import torch


class Config:
    """
    Configuration for the Tweet Sentiment Extraction task (Idea 8).
    Implements settings for DeBERTa-v3-Large backbone, AWP training,
    and 5-Fold Cross-Validation.
    """

    # =========================================================================
    # Environment & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Metadata paths (Pre-split data provided in ./metadata)
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"

    # Sample submission for formatting reference
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working directory for caching processed data/features and saving models
    WORKING_DIR = "./working/idea_8"

    # Final submission output path
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"

    # Tokenizer settings
    MAX_LEN = 128

    # Dropout settings
    HIDDEN_DROPOUT = 0.1
    ATTENTION_DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Training schedule
    EPOCHS = 3
    N_FOLDS = 5

    # Batch Size: DeBERTa-Large is memory intensive.
    # 16 is a safe/efficient balance for A100 40GB with max_len=128.
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32

    # Optimizer settings
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Scheduler settings
    SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.1

    # Loss function settings
    LABEL_SMOOTHING = 0.1

    # =========================================================================
    # Adversarial Weight Perturbation (AWP)
    # =========================================================================
    # AWP Hyperparameters
    AWP_LR = 1e-4
    AWP_EPS = 1e-2

    # AWP Schedule: Enabled only after Epoch 1.
    # Using 0-based indexing:
    # Epoch 0: Standard Training
    # Epoch 1: AWP Training
    # Epoch 2: AWP Training
    AWP_START_EPOCH = 1

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    SENTIMENT_NEUTRAL = "neutral"

    def __init__(self, debug=False, epochs=None, train_batch_size=None):
        """
        Initialize configuration with optional overrides.

        Args:
            debug (bool): If True, enables debug mode (smaller dataset, fewer epochs).
            epochs (int, optional): Override the number of training epochs.
            train_batch_size (int, optional): Override the training batch size.
        """
        # Ensure working and output directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.SUBMISSION_PATH), exist_ok=True)

        self.DEBUG = debug

        # Adjust settings for debugging
        if self.DEBUG:
            self.EPOCHS = 2
            self.TRAIN_BATCH_SIZE = 4
            self.DEBUG_SAMPLE_SIZE = 100

        # Apply manual overrides if provided
        if epochs is not None:
            self.EPOCHS = epochs

        if train_batch_size is not None:
            self.TRAIN_BATCH_SIZE = train_batch_size
