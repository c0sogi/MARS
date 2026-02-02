import os
import torch


class Config:
    """
    Global configuration for the Salt Segmentation task.
    Implements the 'Corrected High-Fidelity Stratified Ensemble' strategy.
    """

    # --------------------
    # General Configuration
    # --------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing available vCPUs (12 available)
    NUM_WORKERS = 4

    # --------------------
    # Paths
    # --------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Caching directory for deterministic processing (Idea 12)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_12")

    # Model checkpoints and submissions
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata files (Pre-generated in ./metadata)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------
    # Data Configuration
    # --------------------
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net architecture (divisible by 32)
    CHANNELS = 3  # Input Multiplexing: [Seismic, Seismic, Depth]

    # --------------------
    # Training Configuration
    # --------------------
    FOLDS = 5
    BATCH_SIZE = 64
    EPOCHS = 80
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15

    # Loss Schedule Strategy
    WARMUP_EPOCHS = 15  # Epochs for BCE+Dice convergence before Lovasz switch

    # --------------------
    # Model Configuration
    # --------------------
    # Architecture: U-Net++ with ResNeXt-50 (32x4d) Encoder and scSE Attention
    ENCODER = "seresnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # Lightweight decoder channels to prevent overfitting
    # Standard U-Net++ depth 5, channels decaying: 256 -> 16
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # --------------------
    # Debug / Development
    # --------------------
    # Flags to speed up development loop if needed
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100
    DEBUG_EPOCHS = 2

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories upon initialization.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories when module is imported
Config.setup()
