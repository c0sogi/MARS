import os
import torch


class Config:
    """
    Global configuration for the Time-Conditioned Deep-Semantic Network (TCDS-Net).
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================
    # Debugging / Development
    # ==========================
    # Set DEBUG to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of patients to use in debug mode

    # ==========================
    # File Paths
    # ==========================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DCM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DCM_DIR = os.path.join(INPUT_DIR, "test")
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Access)
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Preprocessing
    # ==========================
    N_SLICES = 3  # Apical, Middle, Basal
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2

    # Normalization stats (to be computed or fixed)
    # Target (FVC) is Z-scored during training
    NORMALIZE_TARGET = True

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "tf_efficientnet_b2_ns"
    PRETRAINED = True
    IN_CHANS = 3  # Model expects RGB, we will replicate grayscale slices

    # Projection and Head
    N_FEATURES = 128  # Dimension to project flattened image features to
    HIDDEN_DIM = 512  # Hidden dimension for the mixing MLP

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 50
    BATCH_SIZE = 16  # Conservative size for A100 with 3 slices per sample

    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Slower learning for pre-trained features
    LR_HEAD = 1e-3  # Faster learning for the new MLP head

    WEIGHT_DECAY = 1e-2
    PATIENCE = 10  # Early stopping patience

    # Scheduler
    T_MAX = 50  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==========================
    # Metric / Loss
    # ==========================
    SIGMA_CLIP = 70.0
    MAX_ERROR = 1000.0

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration setup complete. Cache dir: {cls.CACHE_DIR}")

    @classmethod
    def get_debug_config(cls):
        """
        Returns a dictionary summary of the configuration for logging.
        """
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
