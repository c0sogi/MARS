import os
import torch


class Config:
    """
    Configuration for Spatially-Aware EfficientNet-B0 CenterNet Pipeline.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory for this specific idea/experiment
    WORK_DIR = "./working/idea_6"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_meta.csv")

    # =========================================================================
    # Data & Image Processing
    # =========================================================================
    IMG_SIZE = 640  # Input resolution for the model
    IN_CHANNELS = 3  # Model expects RGB

    # Augmentation Constraints
    MIN_VISIBILITY = 0.3  # Minimum fraction of bbox area visible after transform

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "efficientnet_b0"

    # 14 Critical Findings (IDs 0-13).
    # Class 14 "No finding" is handled via the Global Classification Head.
    NUM_CLASSES = 14

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 20
    BATCH_SIZE = 8  # Tuned for 16GB VRAM

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Weights for Multi-Task Learning
    LAMBDA_HEATMAP = 1.0  # Focal Loss
    LAMBDA_SIZE = 0.1  # L1 Loss (Scale regression)
    LAMBDA_OFFSET = 1.0  # L1 Loss (Discretization error)
    LAMBDA_GLOBAL = 1.0  # BCE Loss (Finding vs No Finding)

    # =========================================================================
    # Inference & Post-Processing
    # =========================================================================
    CONF_THRESHOLD = 0.2  # Minimum confidence to propose a box
    GLOBAL_NO_FINDING_THRESH = 0.8  # If Global P(No Finding) > 0.8, output Class 14
    NMS_IOU_THRESHOLD = 0.4  # For evaluation/NMS (if used)
    MAX_DETECTIONS_PER_IMG = 100

    @staticmethod
    def setup():
        """Creates necessary directories if they don't exist."""
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure environment is ready
Config.setup()
