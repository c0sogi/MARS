import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Project & Directory Structure
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_5"
    INPUT_ROOT = "./input"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    # Using .npy for efficient storage of processed tensors
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    TRAIN_LABELS_CACHE = os.path.join(WORKING_DIR, "train_labels.npy")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    VAL_LABELS_CACHE = os.path.join(WORKING_DIR, "val_labels.npy")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npy")
    TEST_CLIPS_CACHE = os.path.join(WORKING_DIR, "test_clips.npy")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    SR = 2000  # Sample rate (Hz)
    DURATION = 2.0  # Clip duration (seconds)
    N_FFT = 1024  # FFT window size (High frequency resolution)
    HOP_LENGTH = 64  # Hop length (High temporal resolution)
    N_MELS = 128  # Number of Mel bands
    FMIN = 0  # Minimum frequency
    FMAX = None  # Maximum frequency (defaults to SR/2)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b2"
    NUM_CLASSES = 1
    IN_CHANNELS = 3  # 3 Channels: Log-Mel, Delta, Delta-Delta
    USE_GEM = True  # Generalized Mean Pooling
    PRETRAINED = True  # Use ImageNet weights

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 128  # Large batch size for stability
    EPOCHS = 15  # Max epochs
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # Regularization
    PATIENCE = 5  # Early stopping patience

    # -------------------------------------------------------------------------
    # Compute & Debugging
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4  # Number of DataLoader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLES = 500  # Number of samples to use in debug mode
