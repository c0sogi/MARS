import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Channels: HH, HV, Synthetic Average
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # 5-Fold Cross-Validation
    NUM_FOLDS = 5

    # Dataloader
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # -------------------------------------------------------------------------
    # Model Architecture (DPSCA-CNN)
    # -------------------------------------------------------------------------
    # Backbone: Plain CNN 4 blocks
    # Width Strategy: 64 -> 128 -> 128 -> 128 (Early Expansion)
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Activation & Regularization
    LEAKY_RELU_SLOPE = 0.1
    DROPOUT_RATE = 0.5
    USE_BIAS = True  # Retain bias to preserve initialization dynamics

    # Readout
    FC_DIM = 256  # Dimension after pooling and before final classification

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 75
    PATIENCE = 12  # Early Stopping
    BATCH_SIZE = 32

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Loss
    LOSS_FN = "BCEWithLogitsLoss"

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    USE_TTA = False  # Explicitly disabled per instructions

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
