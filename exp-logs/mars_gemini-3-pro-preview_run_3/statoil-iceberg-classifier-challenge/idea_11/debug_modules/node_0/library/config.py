import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # System Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Input Data Paths
    # --------------------------------------------------------------------------
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Caching Paths (Numpy Arrays)
    # --------------------------------------------------------------------------
    # Versioned filenames to ensure data consistency for Idea 11
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train_v11.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train_v11.npy")
    CACHE_TRAIN_ANGLE = os.path.join(WORKING_DIR, "angle_train_v11.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "ids_train_v11.npy")

    CACHE_VAL_X = os.path.join(WORKING_DIR, "X_val_v11.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "y_val_v11.npy")
    CACHE_VAL_ANGLE = os.path.join(WORKING_DIR, "angle_val_v11.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "ids_val_v11.npy")

    CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test_v11.npy")
    CACHE_TEST_ANGLE = os.path.join(WORKING_DIR, "angle_test_v11.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "ids_test_v11.npy")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: HH, HV, and Average((HH+HV)/2)
    IN_CHANNELS = 3

    # --------------------------------------------------------------------------
    # Model Hyperparameters (SHMP-CNN)
    # --------------------------------------------------------------------------
    # Channel dimensions for the 4 Convolutional Blocks
    # Constrained to prevent overfitting
    BLOCK_CHANNELS = [64, 64, 128, 128]

    # Selective Hierarchical Pooling:
    # Apply Global Max Pooling to Block 3 (index 2) and Block 4 (index 3) only.
    POOLING_BLOCK_INDICES = [2, 3]

    # Classification Head
    FC_HIDDEN_DIM = 256
    DROPOUT_RATE = 0.5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    EARLY_STOPPING_PATIENCE = 10

    # Test-Time Augmentation (Disabled per analysis)
    USE_TTA = False
