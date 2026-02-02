import os
import torch


class Config:
    """
    Centralized configuration for the Siamese Spatially-Strided 2.5D Network (S3D-Net).
    Handles file paths, hyperparameters, and constants.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for data loading
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directory for caching processed tensors
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_38")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    # Image dimensions
    IMG_SIZE = 224

    # Volumetric Sampling
    NUM_SLICES_TOTAL = 32  # Total slices sampled from the volume
    NUM_SLICES_PER_VIEW = 16  # Slices per Siamese view (Even/Odd)
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # --------------------------------------------------------------------------
    # Caching Filenames
    # --------------------------------------------------------------------------
    # Training Cache
    CACHE_TRAIN_X = os.path.join(CACHE_DIR, "X_train.npy")
    CACHE_TRAIN_IDS = os.path.join(CACHE_DIR, "ids_train.npy")
    CACHE_TRAIN_Y = os.path.join(CACHE_DIR, "y_train.npy")

    # Validation Cache
    CACHE_VAL_X = os.path.join(CACHE_DIR, "X_val.npy")
    CACHE_VAL_IDS = os.path.join(CACHE_DIR, "ids_val.npy")
    CACHE_VAL_Y = os.path.join(CACHE_DIR, "y_val.npy")

    # Test Cache
    CACHE_TEST_X = os.path.join(CACHE_DIR, "X_test.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "ids_test.npy")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"

    # Input channels for the backbone
    # Calculation: 16 slices (per view) * 4 modalities = 64 channels
    IN_CHANS = NUM_SLICES_PER_VIEW * NUM_MODALITIES

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 5

    # Regularization
    DROP_PATH_RATE = 0.2  # Stochastic depth rate

    # --------------------------------------------------------------------------
    # Debugging / Development
    # --------------------------------------------------------------------------
    # If True, runs the pipeline on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 32
