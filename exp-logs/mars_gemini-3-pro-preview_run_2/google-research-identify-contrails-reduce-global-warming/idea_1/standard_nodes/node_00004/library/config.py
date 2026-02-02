import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Directory for caching preprocessed data (Idea 1 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Directory and path for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Paths to generated metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # ==========================================
    # Data & Preprocessing Configuration
    # ==========================================
    IMG_SIZE = 256

    # Frame indices for Temporal Difference
    # The dataset provides sequences; index 4 is the labeled frame.
    LABELED_FRAME_IDX = 4
    PREV_FRAME_IDX = 3

    # Input Channels:
    # 3 channels for Ash Composite at t=4
    # 3 channels for Difference (Ash t=4 - Ash t=3)
    IN_CHANNELS = 6

    # Ash False Color Composite Normalization Bounds (Kelvin)
    # These are standard bounds for GOES-16 Ash RGB recipes.
    # Band 15 - Band 14 (Red)
    ASH_RED_MIN = -4.0
    ASH_RED_MAX = 2.0

    # Band 14 - Band 11 (Green)
    ASH_GREEN_MIN = -4.0
    ASH_GREEN_MAX = 5.0

    # Band 14 (Blue)
    ASH_BLUE_MIN = 243.0
    ASH_BLUE_MAX = 303.0

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 4

    # Debugging: Set to an integer (e.g., 500) to limit dataset size
    # for rapid testing. Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model Architecture
    # ==========================================
    ENCODER_NAME = "mobilenet_v3_small"
    ENCODER_WEIGHTS = "imagenet"

    # ==========================================
    # Inference & Metrics
    # ==========================================
    THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
