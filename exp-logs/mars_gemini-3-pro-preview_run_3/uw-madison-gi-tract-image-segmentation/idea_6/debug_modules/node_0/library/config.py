import os
import torch


class Config:
    """
    Configuration for Geometry-Preserving 2.5D U-Net++ with Curriculum Boundary Optimization.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ====================================================
    # Directories & Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_DF_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DF_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DF_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Idea 6
    WORKING_DIR = "./working/idea_6"

    # Output Sub-directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Automatically create necessary directories
    for d in [WORKING_DIR, CHECKPOINT_DIR, PREDICTION_DIR, LOG_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # ====================================================
    # Data Preprocessing & Geometry
    # ====================================================
    # 2.5D Input: Channels correspond to slices [t-1, t, t+1]
    IN_CHANNELS = 3

    # Target Image Size (Height, Width)
    # Strategy: Pad-to-Square to preserve aspect ratio (Geometry-Preserving)
    # 320 is chosen as a standard efficientnet-friendly resolution that covers most cases
    IMG_SIZE = (320, 320)

    # Physical properties
    SLICE_DEPTH_MM = 3.0  # Physical thickness for 3D context if needed

    # Class Definitions
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = 3

    # ====================================================
    # Model Architecture
    # ====================================================
    ARCH = "UnetPlusPlus"
    BACKBONE = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"

    # Deep Supervision: Compute loss at multiple decoder depths to prevent collapse
    DEEP_SUPERVISION = True

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    BATCH_SIZE = 32  # Optimized for A100 40GB
    EPOCHS = 15  # Total training epochs

    # Curriculum Learning Schedule
    # Phase 1: Warmup (Epochs 1 to WARMUP_EPOCHS) -> Loss: BCE + Tversky
    # Phase 2: Refinement (Epochs WARMUP_EPOCHS+1 to End) -> Loss: BCE + Tversky + Boundary
    WARMUP_EPOCHS = 5

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5

    # Learning Rate Scheduler
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # ====================================================
    # Metrics & Validation
    # ====================================================
    # Competition Metric Weights
    DICE_WEIGHT = 0.4
    HAUSDORFF_WEIGHT = 0.6

    # Validation Configuration
    # We perform full 3D volume reconstruction during validation
    VAL_BATCH_SIZE = 32

    # ====================================================
    # Inference & Post-Processing
    # ====================================================
    # Probability threshold for binary mask conversion
    MASK_THRESHOLD = 0.5

    # 3D Connected Component Analysis (CCA)
    # Minimum volume (in pixels) to retain a connected component
    MIN_COMPONENT_SIZE = 100
