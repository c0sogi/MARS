import os
import torch


class Config:
    """
    Configuration class for the Shared-Bottom RoBERTa Dual-Encoder experiment.
    Centralizes all hyperparameters, paths, and strategy settings.
    """

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths (Generated in previous steps)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Input Paths
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Paths (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "roberta-base"
    HIDDEN_SIZE = 768  # Hidden size for roberta-base
    NUM_TARGETS = 30

    # Shared-Bottom Split-Top Structure
    # roberta-base has 12 layers (0-11).
    # Layers 0-9 (10 layers) are shared.
    # Layers 10-11 (2 layers) are split into independent branches.
    SHARED_LAYERS = 10
    SPLIT_LAYERS = 2

    # ==========================================
    # Data & Tokenizer
    # ==========================================
    MAX_LEN = 512
    NUM_WORKERS = 4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Gradient Accumulation Strategy
    # Physical batch size 8 * Accum steps 2 = Effective batch size 16
    BATCH_SIZE = 8
    ACCUM_STEPS = 2

    # Phantom Scheduling Strategy
    # Initialize scheduler for 7 epochs, but stop training after 3.
    # This maintains a higher learning rate during the active training phase.
    PHANTOM_EPOCHS = 7
    STOP_EPOCH = 3

    # Head Warmup Strategy
    # Freeze the backbone for the first epoch to stabilize the randomly initialized head.
    FREEZE_BACKBONE_EPOCHS = 1

    # Optimization
    # Differential Learning Rates
    HEAD_LR = 1e-3
    BACKBONE_LR = 2e-5
    WEIGHT_DECAY = 0.01
    EPS = 1e-6
    MAX_GRAD_NORM = 1.0

    # Scheduler
    SCHEDULER_TYPE = "linear"
    WARMUP_RATIO = 0.1

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    SUBSET_SIZE = 100  # Number of samples to use if DEBUG=True

    def __init__(self, debug=False, batch_size=None, epochs=None):
        """
        Initialize the configuration.

        Args:
            debug (bool): If True, enables debug mode (smaller dataset).
            batch_size (int, optional): Override default batch size.
            epochs (int, optional): Override default phantom epochs.
        """
        self.DEBUG = debug

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        if epochs is not None:
            self.PHANTOM_EPOCHS = epochs
            # If epochs are overridden, we adjust stop_epoch to be the same
            # unless it was specifically meant to be a phantom schedule.
            # For flexibility, if user sets epochs, we assume they want to train for that long.
            self.STOP_EPOCH = epochs

        # Ensure necessary directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def print_config(self):
        """Prints the current configuration settings."""
        print(
            f"Config: {self.BACKBONE} | Shared: {self.SHARED_LAYERS} | Split: {self.SPLIT_LAYERS}"
        )
        print(
            f"Batch: {self.BATCH_SIZE} x {self.ACCUM_STEPS} | Epochs: {self.STOP_EPOCH}/{self.PHANTOM_EPOCHS}"
        )
        print(f"LR: Head={self.HEAD_LR}, Backbone={self.BACKBONE_LR}")
        print(f"Device: {self.device}")
        if self.DEBUG:
            print(f"DEBUG MODE ENABLED (Subset: {self.SUBSET_SIZE})")
