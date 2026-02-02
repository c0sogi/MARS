import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Normalized Shared-Latent Holistic Network (NSL-HN).
    Centralizes all file paths, hyperparameters, and constants.
    """

    # ==========================================
    # 1. File Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory paths
    WORKING_DIR = "./working"
    # Specific cache for Idea 55 (NSL-HN) processed data
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_55")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing & Image Params
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Tri-Slab configuration
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Normalization constants (ImageNet defaults)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    VISUAL_DIM = 1280  # Native output dim of EfficientNet-B0 (GAP)
    LATENT_DIM = 128  # Dimension for Shared Latent Vector (Tabular)
    HEAD_HIDDEN_DIM = 512  # Bottleneck dimension before final projection
    DROPOUT_RATE = 0.2

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Safe for A100 with B0 backbone
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict patience for Early Stopping
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # 5. Metric Constants
    # ==========================================
    SIGMA_CLIP = 70.0  # Minimum confidence value (ml)
    ERROR_CLIP = 1000.0  # Maximum absolute error penalty (ml)

    @staticmethod
    def setup():
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Environment initialized. Device: {Config.DEVICE}")
        print(f"Cache Directory: {Config.CACHE_DIR}")
