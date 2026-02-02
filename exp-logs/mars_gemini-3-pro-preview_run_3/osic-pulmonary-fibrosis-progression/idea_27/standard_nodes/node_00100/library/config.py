import os
import torch


class Config:
    """
    Global configuration for the Gated Metric-Aligned Residual Network (GMAR-Net).
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of data loading workers
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Specific to Idea 27)
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Model Save Path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    for d in [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # Image Preprocessing (Radiological Windowing & Slicing)
    # -------------------------------------------------------------------------
    # Lung Window
    HU_LEVEL = -600
    HU_WIDTH = 1500

    # EfficientNet-B2 Native Resolution
    IMG_SIZE = 260

    # Slice Selection: Anchor + 2 Boundaries
    NUM_SLICES = 3

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b2"

    # Dimensions
    CLINICAL_INPUT_DIM = 5  # Baseline FVC, Time, Age, Sex, Smoking
    LINEAR_INPUT_DIM = 2  # Baseline FVC, Time
    CLINICAL_HIDDEN_DIM = 128
    LATENT_DIM = 64
    IMAGE_PROJ_DIM = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3

    # Optimizer & Scheduler
    WEIGHT_DECAY = 1e-2
    T_MAX = 50  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Metric & Loss Constants
    # -------------------------------------------------------------------------
    # Constants for Metric-Aligned Laplace Log Likelihood
    METRIC_CLIP_SIGMA = 70.0
    METRIC_MAX_ERROR = 1000.0

    # Constants for Post-Processing
    # Note: Target FVC is Z-score standardized during training.
    # These stats should be calculated from training data, but we can store
    # placeholders here or compute them dynamically in the pipeline.
    # For this config, we assume the pipeline will compute/load them.
