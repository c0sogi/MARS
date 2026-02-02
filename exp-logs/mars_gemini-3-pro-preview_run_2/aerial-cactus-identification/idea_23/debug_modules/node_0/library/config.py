import os
import torch


class Config:
    """
    Configuration for the Custom Wide Dual-Pooling SE-ResNeXt experiment.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Pre-generated metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Batch size optimized for A100 GPU and 32x32 images
    BATCH_SIZE = 256
    NUM_WORKERS = 4

    # Debugging control
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Model Architecture Configuration
    # --------------------------------------------------------------------------
    # Architecture: Custom Wide Dual-Pooling SE-ResNeXt with Multi-Scale Aggregation
    # "Super-Wide" Channel Configuration
    MODEL_CHANNELS = [64, 128, 256]

    # Grouped Convolutions Cardinality
    MODEL_CARDINALITY = 32

    # Resolution Preservation Stages (32x32 -> 16x16 -> 8x8)
    # Note: Logic handled in model definition, but config reflects the intent.

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    EPOCHS = 20

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    OPTIMIZER = "AdamW"

    # Scheduler settings (Cosine Annealing)
    SCHEDULER = "CosineAnnealingLR"
    ETA_MIN = 1e-6

    # --------------------------------------------------------------------------
    # Ensemble Configuration
    # --------------------------------------------------------------------------
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
