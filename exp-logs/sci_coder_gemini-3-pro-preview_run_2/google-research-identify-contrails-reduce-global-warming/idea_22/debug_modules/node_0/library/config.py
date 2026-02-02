import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_22"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMAGE_SIZE = 256
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3
    # Total frames in sequence: 4 before + 1 current + 3 after = 8

    # Input Engineering
    # 3 Channels for Ash Color Scheme + 3 Channels for Temporal Difference
    IN_CHANNELS = 6

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    ENCODER_NAME = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # Decoder specifics (Extended Kernel Strategy)
    DECODER_CHANNELS = [256, 128, 64, 32, 16]
    DECODER_KERNEL_SIZE = 11  # Extended kernel size for decoder

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Parameters
    FOCAL_GAMMA = 2.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs


def setup_system(seed=Config.SEED):
    """
    Sets up the system for reproducible training:
    1. Sets random seeds.
    2. Creates necessary directories.
    """
    # 1. Set Random Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 2. Create Directories
    directories = [
        Config.WORKING_DIR,
        Config.CACHE_DIR,
        Config.CHECKPOINT_DIR,
        Config.SUBMISSION_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"System setup complete. Device: {Config.DEVICE}, Seed: {seed}")
    print(f"Working directory: {Config.WORKING_DIR}")
