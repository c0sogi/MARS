import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # We use a new working directory for this optimized strategy to avoid
    # conflicts with previous failed ideas (e.g., idea_5).
    WORKING_DIR = "./working/idea_6"

    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 256

    # Input Channels: 6
    # 4 channels from "Ash" composite (Bands 11, 13, 14, 15)
    # + 1 channel for Temporal Difference (Ash_t - Ash_t-1) which is often derived
    #   but here we treat the input tensor construction in the dataset class.
    #   The strategy specifies a 6-channel tensor.
    N_CHANNELS = 6

    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "Unet"
    BACKBONE = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Extended optimization budget to ensure convergence with the larger backbone
    EPOCHS = 40

    # Batch size adjusted for 40GB VRAM and stability
    BATCH_SIZE = 32

    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler settings
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Checkpointing strategy
    # We average the top-k checkpoints from the latter half of training
    SAVE_TOP_K = 5
    START_SAVING_EPOCH = 20

    # =========================================================================
    # Post-Processing
    # =========================================================================
    THRESHOLD = 0.5

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flag to run on a subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def seed_everything(cls, seed=None):
        """
        Sets the random seed for reproducibility across torch, numpy, and random.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
