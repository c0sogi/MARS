import os
import torch


class Config:
    """
    Global configuration for the Salt Segmentation Task.
    Implements the FP32-Stabilized Marginalized-Distillation strategy.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = "./working/idea_40"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_PATH = os.path.join(INPUT_DIR, "depths.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Original image dimensions
    ORIG_SIZE = 101

    # Padded dimensions for model input (divisible by 32 for ResNet)
    IMG_SIZE = 128

    # Input channels (Grayscale)
    CHANNELS = 1

    # Number of workers for data loading
    NUM_WORKERS = 4

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Batch size (Fits on A100 with 128x128 images)
    BATCH_SIZE = 32

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Training duration
    EPOCHS_STAGE1 = 50  # Specialist Teacher Training
    EPOCHS_STAGE3 = 50  # Generalist Student Training

    # Cross-validation folds
    N_FOLDS = 5

    # Gating threshold for teacher models (discard if val mAP < threshold)
    TEACHER_GATING_THRESHOLD = 0.75

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Elastic Transform settings (Crucial for performance)
    AUG_ELASTIC_ALPHA = 120.0
    AUG_ELASTIC_SIGMA = 6.0
    AUG_ELASTIC_PROB = 0.2

    # Rigid Transform settings (ShiftScaleRotate)
    AUG_RIGID_PROB = 0.2

    # =========================================================================
    # Marginalization & Distillation
    # =========================================================================
    # Depth scan values (in Standard Deviations) for generating soft targets
    DEPTH_SCAN_VALUES = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet34"
    PRETRAINED = True

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for attr in dir(Config):
            if not attr.startswith("__") and not callable(getattr(Config, attr)):
                print(f"{attr}: {getattr(Config, attr)}")
        print("=" * 30)
