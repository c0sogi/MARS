import os
import torch


class Config:
    """
    Configuration for the Point-Wise Relative-Time Fusion Network (PRT-Net).
    Centralizes all file paths, hyperparameters, and constants.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_10"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Pre-generated metadata paths
    METADATA_TRAIN = "./metadata/train.csv"
    METADATA_VAL = "./metadata/val.csv"
    METADATA_TEST = "./metadata/test.csv"

    # Working directory for this specific idea/experiment
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Cache directory for deterministic data processing (e.g., resized images/arrays)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoint directory for saving model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Final submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary writable directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing & Augmentation
    # -------------------------------------------------------------------------
    IMG_SIZE = 256
    NUM_SLICES = 3  # Apical, Middle, Basal

    # Normalization Constants
    TIME_SCALE = 0.01  # Scale factor for Relative Weeks (t_rel)

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Dimension to project flattened CNN features into before concatenation
    IMG_PROJ_DIM = 128

    # MLP Hidden Layers for the fusion head
    # Note: BN/LN are excluded from this MLP in the model implementation
    FUSION_HIDDEN_DIMS = [512, 256]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 2  # Adjust based on vCPU availability (12 vCPUs available)

    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Lower LR for fine-tuning the pre-trained backbone
    LR_HEAD = 1e-3  # Higher LR for the new MLP head

    # Optimizer settings
    WEIGHT_DECAY = 0.01

    # -------------------------------------------------------------------------
    # Loss & Metric Configuration
    # -------------------------------------------------------------------------
    # Offset added to softplus(sigma) during training to prevent singularity
    SIGMA_OFFSET = 0.05

    # Metric specific clipping (used in validation/evaluation)
    METRIC_SIGMA_CLIP = 70
    METRIC_MAX_ERROR = 1000

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        """Returns the configuration as a dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
