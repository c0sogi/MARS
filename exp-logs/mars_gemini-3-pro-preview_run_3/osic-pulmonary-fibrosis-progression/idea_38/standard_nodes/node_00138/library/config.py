import os
import torch


class Config:
    """
    Configuration for Supervised Cascaded Output-Space Residual Network (SCOSR-Net).
    Centralizes all hyperparameters, file paths, and constants.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Write Access)
    # Using idea_38 as the specific experiment identifier
    WORKING_DIR = "./working/idea_38"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Loading
    # =========================================================================
    # Image Parameters
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices

    # DICOM Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Normalization
    # Note: FVC is Z-score standardized in the pipeline, but we define constants if needed.
    # Relative time scaling factor
    TIME_SCALE = 0.01

    # DataLoader
    BATCH_SIZE = 32

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone
    BACKBONE_ARCH = "tf_efficientnet_b2_ns"
    PRETRAINED = True

    # Dimensions
    CLINICAL_INPUT_DIM = (
        4  # Age, Sex, SmokingStatus, RelativeTime (Percent is excluded)
    )
    CLINICAL_LATENT_DIM = 64
    FUSED_DIM = 128  # Concatenation of Visual + Clinical Latent

    # Regularization
    DROPOUT_RATE = 0.2

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 50

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3

    # Optimizer
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50  # Should match EPOCHS
    ETA_MIN = 1e-6

    # Loss Function
    # Weight for the auxiliary clinical supervision (lambda)
    AUX_LOSS_WEIGHT = 0.5

    # =========================================================================
    # Post-Processing & Metrics
    # =========================================================================
    SIGMA_CLIP_MIN = 70
    MAX_ERROR_CLIP = 1000

    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Create directories immediately upon import to ensure availability
Config.setup_directories()
