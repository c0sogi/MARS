import os
import torch


class Config:
    """
    Configuration for the Wide-Attention Isomorphic Dual-Polarity CNN (WA-IDPH-CNN) experiment.
    """

    # -------------------------------------------------------------------------
    # Experiment Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Input shape: (Channels, Height, Width)
    # Channels: 0=HH, 1=HV, 2=Avg((HH+HV)/2)
    INPUT_SHAPE = (3, 75, 75)
    NUM_WORKERS = 4  # Number of workers for DataLoader

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 75
    LEARNING_RATE = 1e-3  # Constant learning rate (AdamW)
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    PATIENCE = 12  # Early stopping patience
    N_FOLDS = 5  # 5-Fold Cross-Validation

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Backbone: Plain CNN (No Residuals)
    # Channel expansion strategy: 64 -> 128 -> 128 -> 128
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Activation: LeakyReLU with negative slope 0.1 to preserve radar shadows
    LEAKY_RELU_SLOPE = 0.1

    # Regularization: Dropout applied after activation in classification head
    DROPOUT_RATE = 0.5

    # Attention: Wide-SE Module with Reduction Ratio r=2
    SE_REDUCTION_RATIO = 2

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    USE_TTA = False  # Explicitly disable Test-Time Augmentation

    # -------------------------------------------------------------------------
    # Hardware Configuration
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Path Configuration
    # -------------------------------------------------------------------------
    # Base working directory for this specific idea
    WORKING_DIR = "./working/idea_57"

    # Writeable directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Read-only Input Data
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Read-only Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    @staticmethod
    def setup():
        """Creates necessary working directories if they don't exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
