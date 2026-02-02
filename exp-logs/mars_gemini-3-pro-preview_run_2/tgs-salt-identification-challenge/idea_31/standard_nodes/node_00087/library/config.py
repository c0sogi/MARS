import os
import torch


class Config:
    """
    Configuration for the Salt Segmentation Task.
    Implements the Marginalized-Distillation Multi-Task Wide-LinkNet strategy.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (Stage 1 models, pseudo-labels)
    # Requirement: Ensure this directory exists
    WORKING_DIR = "./working/idea_31"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Global Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development flags
    # Set MAX_SAMPLES to an integer (e.g., 100) to debug the pipeline with a subset of data
    DEBUG = False
    MAX_SAMPLES = None

    # -------------------------------------------------------------------------
    # Data & Image Parameters
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101
    # Pad to 128x128 to ensure divisibility by 32 (standard for ResNet/UNet)
    IMG_SIZE = 128
    CHANNELS = 1  # Grayscale input

    # Normalization (Standard ImageNet Statistics)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler: Cosine Annealing
    T_MAX = 50
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    EPOCHS = 50

    # Depth Jitter: Gaussian noise added to depth input during training
    # to prevent overfitting to discrete depth values.
    DEPTH_JITTER_STD = 0.1

    # -------------------------------------------------------------------------
    # Augmentation Parameters
    # -------------------------------------------------------------------------
    # Non-Rigid: Elastic Transform
    AUG_ELASTIC_P = 0.2
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6

    # Rigid: ShiftScaleRotate
    AUG_RIGID_P = 0.2

    # Test Time Augmentation (TTA)
    USE_TTA = True  # Horizontal Flip

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet34"
    # Wide-LinkNet: Internal decoder width
    DECODER_CHANNELS = 32

    @staticmethod
    def setup():
        """Creates necessary working and submission directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to guarantee directory existence
Config.setup()
