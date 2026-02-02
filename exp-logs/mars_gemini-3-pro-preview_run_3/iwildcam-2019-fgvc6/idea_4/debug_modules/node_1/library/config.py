import os
import torch


class Config:
    """
    Centralized configuration for the Animal Species Classification task.
    Includes file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_ROOT, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_ROOT, "test_images")

    # Sample Submission for formatting
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output Paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 256
    NUM_CLASSES = 23  # Classes 0 through 22
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # ConvNeXt V2 Tiny with MAE pretraining (ImageNet-22k -> 1k)
    MODEL_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k"

    # Pooling & Head Strategy
    USE_GEM_POOLING = True
    # Dropout rates for Multi-Sample Dropout
    DROPOUT_RATES = [0.1, 0.1, 0.1, 0.1, 0.1]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 128  # A100 40GB can handle larger batches for 256x256
    EPOCHS = 10

    # Optimizer & Scheduler
    MAX_LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
