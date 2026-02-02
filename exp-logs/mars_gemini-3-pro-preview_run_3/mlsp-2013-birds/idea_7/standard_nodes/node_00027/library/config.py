import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    PROJECT_NAME = "bird_species_classification"
    IDEA_NAME = "idea_7"
    SEED = 42
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    ESSENTIAL_DATA = os.path.join(INPUT_ROOT, "essential_data")
    METADATA_DIR = "./metadata"

    # Working Directories
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    SR = 16000
    DURATION = 10  # seconds

    # Spectrogram Parameters
    N_MELS = 224
    N_FFT = 1024
    # HOP_LENGTH chosen to yield approx 448 frames for 10s, allowing 3 tiles of 224
    # 16000 * 10 / 357 ~= 448
    HOP_LENGTH = 357
    FMIN = 20
    FMAX = 16000 // 2

    # -------------------------------------------------------------------------
    # Model & Tiling Parameters
    # -------------------------------------------------------------------------
    BACKBONE = "resnet18"
    PRETRAINED = True

    # Input Dimensions
    IMG_SIZE = (224, 224)  # (Height, Width) of the model input
    IN_CHANNELS = 3  # Replicating mono spectrogram to 3 channels

    # Multi-Instance Learning Strategy
    # NUM_TILES removed - using single image resizing (Cite Lesson 00019)

    NUM_CLASSES = 19
    DROPOUT = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 50
    BATCH_SIZE = 32

    # Optimizer & Scheduler
    LEARNING_RATE = (
        1e-3  # Increased for better convergence on small data (Cite Lesson 00022)
    )
    WEIGHT_DECAY = 0.0  # Reduced regularization (Cite Lesson 00022)
    MIN_LR = 1e-5

    # Early Stopping
    PATIENCE = 5

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    @classmethod
    def create_dirs(cls):
        """
        Ensures that all necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.create_dirs()
