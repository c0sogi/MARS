import os
import torch


class Config:
    """
    Configuration class for the Dual-Axis Tri-Slab Network pipeline.
    Centralizes hyperparameters, file paths, and runtime settings.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLES = 32  # Number of samples to use in debug mode

    # ==========================================
    # Compute Environment
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # ==========================================
    # File System Paths
    # ==========================================
    # Input Data
    INPUT_ROOT = "./input"
    DICOM_TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
    DICOM_TEST_DIR = os.path.join(INPUT_ROOT, "test")

    # Metadata (Generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working & Cache Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Preprocessing
    # ==========================================
    IMG_SIZE = 224  # Input resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Number of slabs for Tri-Slab generation

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BATCH_SIZE = 16  # Batch size (adjusted for dual-branch architecture)
    LEARNING_RATE = 1e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    EPOCHS = 30  # Maximum training epochs

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # Metric Constraints
    MAX_ERROR = 1000  # Clipping threshold for error
    MIN_CONFIDENCE = 70  # Clipping threshold for confidence
