import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for caching processed tensors/dataframes
    # idea_28 corresponds to the current SIRV strategy
    CACHE_DIR = "./working/idea_28"

    # Output directory for submissions
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image dimensions
    IMAGE_SIZE = 224

    # Input channels: 3 modalities (FLAIR, T1wCE, T2w) * 3 depths = 9 channels
    IN_CHANNELS = 9

    # Relative depths for sampling (Scale-Invariant)
    # 40%, 50% (Center), 60% of the Brain ROI
    RELATIVE_DEPTHS = [0.4, 0.5, 0.6]

    # Cases to strictly exclude from training
    EXCLUDE_CASES = [109, 123, 709]

    # Debugging / Development
    DEBUG = False
    DEBUG_DATASET_SIZE = 50  # Only use 50 subjects if DEBUG is True

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"

    # Training settings
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4

    # Regularization
    WEIGHT_DECAY = 1e-2
    # Revert to default dropout (0.2) to avoid underfitting (Cite solution_lesson_node_00012)
    DROPOUT_RATE = 0.2

    # Cross-Validation
    N_FOLDS = 5
    SEED = 42

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_system(seed=Config.SEED):
    """
    Sets up the environment for reproducible training.
    1. Creates necessary directories (cache, submission).
    2. Sets random seeds for python, numpy, and torch.
    """
    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"System setup complete. Device: {Config.DEVICE}, Seed: {seed}")
    print(f"Cache Directory: {Config.CACHE_DIR}")
