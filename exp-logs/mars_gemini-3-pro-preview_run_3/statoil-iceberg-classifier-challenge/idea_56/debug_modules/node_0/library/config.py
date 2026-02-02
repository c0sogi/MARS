import os
import torch


class Config:
    """
    Centralized configuration for the Decoupled Statistical-Isomorphic CNN (DSI-CNN) pipeline.
    Contains file paths, hyperparameters, and model specifications.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working & Output Directories
    # Specific working directory for Idea 56 to avoid conflicts
    WORK_DIR = "./working/idea_56"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist (best practice to define paths that are guaranteed valid)
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache File Paths
    CACHE_TRAIN_X = os.path.join(WORK_DIR, "X_train_cache.npy")
    CACHE_TRAIN_Y = os.path.join(WORK_DIR, "y_train_cache.npy")
    CACHE_TRAIN_META = os.path.join(WORK_DIR, "meta_train_cache.npy")  # For angles/ids

    CACHE_VAL_X = os.path.join(WORK_DIR, "X_val_cache.npy")
    CACHE_VAL_Y = os.path.join(WORK_DIR, "y_val_cache.npy")
    CACHE_VAL_META = os.path.join(WORK_DIR, "meta_val_cache.npy")

    CACHE_TEST_X = os.path.join(WORK_DIR, "X_test_cache.npy")
    CACHE_TEST_META = os.path.join(WORK_DIR, "meta_test_cache.npy")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    IMAGE_SIZE = 75
    # 3 Channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    IN_CHANNELS = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 75

    # Optimization
    LEARNING_RATE = 1e-3  # Constant learning rate
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Regularization / Early Stopping
    PATIENCE = 12
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Model Architecture Specifications
    # =========================================================================
    # Plain CNN Backbone widths: 64 -> 128 -> 128 -> 128
    BLOCK_CHANNELS = [64, 128, 128, 128]

    # LeakyReLU negative slope
    LEAKY_RELU_SLOPE = 0.1

    # Feature Vector Size before classification head
    # (64_max + 64_min) from Stage 3 + (64_max + 64_min) from Stage 4 + 1 (angle)
    # This is calculated as: (64*2) + (64*2) + 1 = 257
    FC_INPUT_DIM = 257
