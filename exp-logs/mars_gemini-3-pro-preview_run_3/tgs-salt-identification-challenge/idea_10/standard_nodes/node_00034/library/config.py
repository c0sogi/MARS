import os
import torch


class Config:
    """
    Global configuration for the Salt Segmentation task using
    Holistic High-Fidelity Stratified Ensemble strategy.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    EXP_NAME = "idea_10"
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # -------------------------------------------------------------------------
    # Compute Environment
    # -------------------------------------------------------------------------
    # 12 vCPUs available, so 4 workers is a safe standard
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea as requested
    WORK_DIR = os.path.join("./working", EXP_NAME)
    SUBMISSION_DIR = "./submission"

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size (divisible by 32 for ResNeXt/U-Net)
    IN_CHANNELS = 3  # [Seismic, Seismic, Depth]

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    ENCODER_NAME = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_NAME = "unetplusplus"
    DECODER_ATTENTION_TYPE = "scse"  # Spatial and Channel Squeeze & Excitation

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    FOLDS = 1
    EPOCHS = 100
    VALIDATION_STRATEGY = "fixed"
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4

    # Loss Schedule
    # Epoch to switch from BCE+Dice to Lovasz-Hinge
    LOVASZ_SWITCH_EPOCH = 15

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
