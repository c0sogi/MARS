import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    IDEA_NAME = "idea_23"
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (idea_23)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Image Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices
    TIME_SCALE = 0.01  # Scaling factor for relative weeks (t_rel)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Backbone
    BACKBONE_NAME = "efficientnet_b2"

    # Architecture Dimensions
    # Clinical features: Baseline FVC, Baseline Percent, t_rel, Age, Sex, Smoking
    N_TABULAR_FEATURES = 6
    PROJECTION_DIM = 64  # Output dim for image and clinical streams
    HIDDEN_DIM = 128  # Hidden layer size for MLPs

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimizer & Scheduler
    LR_BACKBONE = 1e-4  # Lower LR for fine-tuning backbone
    LR_HEAD = 1e-3  # Higher LR for MLPs and Heads
    WEIGHT_DECAY = 1e-2
    T_MAX = 50  # For CosineAnnealingLR

    # Loss Function & Metric Constants
    MIN_UNCERTAINTY = 70.0  # ml, clipped in metric and post-processing
    MAX_ERROR = 1000.0  # ml, clipped in metric

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"Configuration for {cls.IDEA_NAME}:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Backbone: {cls.BACKBONE_NAME}")
        print(f"  Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"  Slices per Patient: {cls.NUM_SLICES}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  LR Backbone: {cls.LR_BACKBONE}")
        print(f"  LR Head: {cls.LR_HEAD}")
        print(f"  Cache Dir: {cls.CACHE_DIR}")
