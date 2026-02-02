import os
import torch


class Config:
    """
    Central configuration for the Cactus Identification Task.
    Implements settings for 'Heterogeneous Trust-Gated Mixture of Experts'.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    EXPERIMENT_ID = "idea_29"

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata directories (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directories (Write Allowed)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMG_SIZE = 32
    NUM_CLASSES = 1
    NUM_FOLDS = 5

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # -------------------------------------------------------------------------
    # Compute Configuration
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8  # Optimizing for 12 vCPUs
    PIN_MEMORY = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Stage 1: Expert Training
    EPOCHS = 30
    BATCH_SIZE = 256  # Large batch size for A100 and small images
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW default

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 20
    SWA_LR = 5e-4

    # Auxiliary Task (Log-Transformed File Size Regression)
    # Lambda weight for the regression loss component
    AUX_LOSS_WEIGHT = 1.0

    # Stage 2: Gating Network Training
    GATE_EPOCHS = 20
    GATE_LR = 1e-3

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # The two expert backbones defined in the strategy
    MODEL_ARCHS = ["CactusRepVGG", "CactusResNet"]

    @classmethod
    def setup(cls):
        """
        Initializes the working directory structure.
        Must be called at the start of the pipeline.
        """
        for path in [cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(path, exist_ok=True)

    @classmethod
    def print_summary(cls):
        """Prints a summary of the current configuration."""
        print(f"\n[Config] Experiment: {cls.EXPERIMENT_ID}")
        print(f"[Config] Device: {cls.DEVICE}")
        print(f"[Config] Input: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"[Config] Batch Size: {cls.BATCH_SIZE}")
        print(f"[Config] Epochs: {cls.EPOCHS} (SWA Start: {cls.SWA_START_EPOCH})")
        print(f"[Config] Mixup Alpha: {cls.MIXUP_ALPHA}")
        print(f"[Config] Aux Loss Weight: {cls.AUX_LOSS_WEIGHT}")
        print(f"[Config] Debug Mode: {cls.DEBUG}")
        print(f"[Config] Cache Dir: {cls.CACHE_DIR}\n")
