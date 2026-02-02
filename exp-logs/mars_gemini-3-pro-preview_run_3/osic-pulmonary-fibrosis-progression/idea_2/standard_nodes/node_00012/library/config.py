import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging: Set to True to train on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20

    # -------------------------------------------------------------------------
    # Path Configuration
    # -------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Allowed)
    # Using 'idea_2' to isolate cache for this specific architecture
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing & Loading
    # -------------------------------------------------------------------------
    IMG_SIZE = 256

    # Slice positions relative to lung volume: 20% (Apical), 50% (Middle), 80% (Basal)
    SLICE_POSITIONS = [0.2, 0.5, 0.8]
    NUM_SLICES = len(SLICE_POSITIONS)

    # ImageNet Normalization Statistics
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # DICOM Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    # Dimension to project aggregated image features into before fusion
    IMG_EMBED_DIM = 128

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 35
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4

    # Scheduler settings
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Metric & Post-processing
    # -------------------------------------------------------------------------
    # Metric constants for the Modified Laplace Log Likelihood
    METRIC_MAX_ERR = 1000
    METRIC_MIN_CONF = 70

    @staticmethod
    def setup():
        """Creates necessary working and output directories."""
        dirs = [
            Config.WORKING_DIR,
            Config.CACHE_DIR,
            Config.CHECKPOINT_DIR,
            Config.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(
            f"Directories initialized at {Config.WORKING_DIR} and {Config.SUBMISSION_DIR}"
        )
