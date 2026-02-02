import os
import torch


class Config:
    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use when DEBUG is True
    PROJECT_NAME = "idea_4"

    # ==========================
    # Hardware & System
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of data loading workers

    # ==========================
    # Directories & Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-split)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # File paths
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    LOG_PATH = os.path.join(WORKING_DIR, "train.log")

    # ==========================
    # Data Configuration
    # ==========================
    ORIGINAL_IMAGE_SIZE = 96
    CENTER_CROP_SIZE = 48  # Hard attention crop size (Center 48x48)
    INPUT_SIZE = 48  # Model input size (matches crop size)

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Augmentation Settings
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = True
    AUG_ROTATE_90 = True
    AUG_BRIGHTNESS_LIMIT = 0.1  # Mild brightness adjustment
    AUG_CONTRAST_LIMIT = 0.1  # Mild contrast adjustment

    # ==========================
    # Model Configuration
    # ==========================
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification
    DROP_PATH_RATE = 0.1  # Stochastic depth rate

    # ==========================
    # Training Hyperparameters
    # ==========================
    NUM_EPOCHS = 20
    BATCH_SIZE = 256  # Large batch size for small images
    LEARNING_RATE = 2e-4  # Initial learning rate for AdamW
    WEIGHT_DECAY = 0.05  # Weight decay for AdamW
    EARLY_STOPPING_PATIENCE = 6

    # Scheduler Settings (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    MIN_LR = 1e-6

    # ==========================
    # Inference
    # ==========================
    TTA_ENABLED = True
    TTA_STEPS = 4  # Original, HFlip, VFlip, Rotate90

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        Call this method at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Print configuration summary
        print(f"Config Setup Complete for {cls.PROJECT_NAME}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.MODEL_NAME} (Input: {cls.INPUT_SIZE}x{cls.INPUT_SIZE})")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Output Dir: {cls.WORKING_DIR}")
