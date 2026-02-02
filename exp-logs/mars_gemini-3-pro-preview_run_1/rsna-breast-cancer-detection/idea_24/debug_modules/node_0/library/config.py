import os
import torch
import random
import numpy as np


class Config:
    # Project & Experiment Identification
    PROJECT_NAME = "idea_24"
    SEED = 42

    # Hardware & System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # File Paths
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output & Working Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, PROJECT_NAME)
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Model Architecture Hyperparameters
    MODEL_BACKBONE = "tf_efficientnet_b2_ns"  # EfficientNet-B2
    IMAGE_SIZE = (768, 768)
    IN_CHANNELS = 3  # Image + Age + Implant

    # Training Hyperparameters
    BATCH_SIZE = 4  # Conservative size for 768x768 Siamese on 40GB GPU
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    POS_WEIGHT = 47.0  # Inverse class frequency approximation

    # Optimization
    WEIGHT_DECAY = 1e-2
    GRADIENT_CLIPPING = False  # Disabled as per strategy

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Performs necessary setup: creates directories and sets seeds.
        """
        # Create cache directory
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Set reproducibility seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)

        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Perform setup immediately when module is imported to ensure environment is ready
Config.setup()
