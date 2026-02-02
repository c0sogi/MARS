import os
import torch


class Config:
    """
    Centralized configuration for the Stabilized Multi-Task Single-Instance Network (SMT-SIN).
    Defines hyperparameters for data processing, model architecture, and training loop.
    """

    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "SMT-SIN-Breast-Cancer-Detection"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==========================
    # Paths & Directories
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Pre-generated Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Source Directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Directories
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================
    # Data Configuration
    # ==========================
    # Resolution Strategy: Optimal geometric mean between 224 and 1024
    IMG_HEIGHT = 640
    IMG_WIDTH = 640
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # Input Engineering: Channel Expansion
    # Channel 0: Original
    # Channel 1: CLAHE (Clip Limit 2.0)
    # Channel 2: CLAHE (Clip Limit 4.0)
    IN_CHANNELS = 3
    USE_CLAHE_CHANNELS = True
    CLAHE_CLIP_LIMITS = [2.0, 4.0]

    # Dataloader Settings
    # Batch size tailored for A100 40GB with 640x640 resolution
    BATCH_SIZE = 24
    NUM_WORKERS = 12  # Matches available vCPUs
    PIN_MEMORY = True

    # ==========================
    # Model Architecture
    # ==========================
    # Backbone: Fine-Tuned EfficientNetV2-Small
    # Using 'tf_efficientnetv2_s.in1k' from timm
    MODEL_BACKBONE = "tf_efficientnetv2_s.in1k"
    PRETRAINED = True

    # Regularization
    DROP_RATE = 0.3  # Dropout rate for the classifier head
    DROP_PATH_RATE = 0.2  # Stochastic depth rate

    # Output Heads
    NUM_CLASSES = 1  # Primary: Binary Cancer Classification
    AUX_BIRADS_CLASSES = 1  # Auxiliary 1: BIRADS Regression (0-2)
    AUX_DENSITY_CLASSES = 4  # Auxiliary 2: Density Classification (A-D)

    # Metadata Fusion
    # Dimension of the metadata vector after MLP processing
    META_EMBED_DIM = 64

    # ==========================
    # Training Configuration
    # ==========================
    EPOCHS = 10

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Loss Weights & Strategy
    # Primary Head: FP32-Guarded Weighted BCE
    POS_WEIGHT = 15.0

    # Multi-Task Loss Balancing
    LOSS_WEIGHT_CANCER = 1.0
    LOSS_WEIGHT_BIRADS = 0.5
    LOSS_WEIGHT_DENSITY = 0.5

    # Learning Rate Scheduler (OneCycleLR)
    PCT_START = 0.1
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 100.0

    # Hardware & Precision
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Enable Mixed Precision (Float16)

    @classmethod
    def create_dirs(cls):
        """Creates necessary output directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon module import
Config.create_dirs()
