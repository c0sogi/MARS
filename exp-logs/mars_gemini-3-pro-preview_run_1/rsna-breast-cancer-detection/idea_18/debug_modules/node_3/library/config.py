import os
import torch


class Config:
    """
    Central configuration for the Pyramid Symmetry-Difference Siamese Network.
    Defines hyperparameters, file paths, and system settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_18_pyramid_siamese"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for artifacts (Cache, Model Checkpoints)
    # Ensure this directory exists as per requirements
    WORKING_DIR = "./working/idea_18"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"
    # Ensure submission directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Input dimensions: (3, 768, 768)
    # Channels: 1 (Image) + 1 (Age Map) + 1 (Implant Map)
    IMAGE_SIZE = (768, 768)
    IN_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    BACKBONE = "tf_efficientnet_b2_ns"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Dropout rates for the backbone
    DROP_RATE = 0.3
    DROP_PATH_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size adjusted for A100 40GB with Siamese architecture (2x images per sample)
    BATCH_SIZE = 8
    NUM_EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function Weights
    # Aggressive positive weighting to handle 2% positive class imbalance
    POS_WEIGHT = 47.0

    # Gradient Handling
    # Gradient clipping is disabled to allow large updates for the minority class
    MAX_GRAD_NORM = None

    # -------------------------------------------------------------------------
    # System / Hardware
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = False

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    # Whether to use Test Time Augmentation (TTA)
    USE_TTA = False
