import os
import torch


class Config:
    """
    Configuration for the Channel-Adaptive Symmetric Dual-Axis Network (Idea 23).
    Acts as a central dependency for all modules.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    IDEA_NAME = "idea_23"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, setting workers to a reasonable number
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for deterministic data processing (Idea 23 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data paths
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, f"{IDEA_NAME}_best_model.pth")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Resolution: Native 224x224 to avoid upscaling artifacts
    IMG_SIZE = 224

    # Tri-Slab Generation
    NUM_SLABS = 3
    SLAB_OVERLAP = 0.15  # 15% overlap

    # Image Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Native output dim of EfficientNet-B0
    TABULAR_HIDDEN_DIM = 1280  # Project tabular up to match visual

    # Tabular features to use
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    EPOCHS = 50
    PATIENCE = 6  # Strict patience (5-8 range)

    # Metric / Loss Constants
    SIGMA_CLIP = 70.0
    ERROR_CLIP = 1000.0

    # -------------------------------------------------------------------------
    # Debugging / Runtime Control
    # -------------------------------------------------------------------------
    # Flags to control dataset size for debugging/development
    DEBUG = False
    DEBUG_DATA_SIZE = 50  # Only use 50 samples if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for cache and submission.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Returns configuration as a dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
