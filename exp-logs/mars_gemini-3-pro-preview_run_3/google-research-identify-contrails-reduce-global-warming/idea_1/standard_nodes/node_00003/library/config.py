import os
import torch


class Config:
    """
    Configuration parameters for the Contrail Detection pipeline.
    Centralizes paths, data normalization constants, and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata CSV Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 256

    # Ash Composite Normalization Bounds
    # Derived from standard GOES-16 ABI Ash RGB recipes for contrail detection.
    # Channel 0: Red (Band 15 - Band 13)
    ASH_RED_MIN = -4.0
    ASH_RED_MAX = 2.0

    # Channel 1: Green (Band 14 - Band 11)
    ASH_GREEN_MIN = -4.0
    ASH_GREEN_MAX = 5.0

    # Channel 2: Blue (Band 14 Temperature in Kelvin)
    ASH_BLUE_MIN = 243.0
    ASH_BLUE_MAX = 303.0

    # Input Tensor Configuration
    # 6 Channels total:
    # - 3 channels for current frame (t) Ash composite
    # - 3 channels for temporal difference (Ash(t) - Ash(t-1))
    IN_CHANNELS = 6

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # Lightweight U-Net Encoder settings
    # Starts with fewer filters to maintain high throughput
    # Increased capacity for better performance (Cite solution_lesson_node_00002)
    ENCODER_FILTERS = [32, 64, 128, 256]

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Compute Resources
    # 12 vCPUs available, setting to 8 to leave overhead for system/GPU sync
    NUM_WORKERS = 8

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    # Probability threshold for binary mask generation
    THRESHOLD = 0.5

    # -------------------------------------------------------------------------
    # Debug / Development
    # -------------------------------------------------------------------------
    # Set DEBUG to True to train on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 300

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
