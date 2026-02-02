import os
import torch


class Config:
    """
    Central configuration for the Wide-Field Stratified Instance Learning (WSIL) Network.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)

    # ==========================================
    # Data Pipeline
    # ==========================================
    IMG_SIZE = 224
    INPUT_CHANNELS = 3  # Channel 1: FLAIR, Channel 2: T1wCE, Channel 3: T2w

    # Wide-Field Strategy
    STRIDE = 10  # Delta for wide-field expansion (M-10, M, M+10)

    # Debugging / Development
    DEBUG = False
    MAX_SAMPLES = (
        None  # Set to an integer (e.g., 50) to limit dataset size for quick testing
    )

    # ==========================================
    # File Paths
    # ==========================================
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Caching
    # We use idea_13 as the specific working directory for this iteration
    CACHE_DIR = "./working/idea_13"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache File Names
    CACHE_TRAIN_IMAGES = os.path.join(CACHE_DIR, "train_images.npy")
    CACHE_TRAIN_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(CACHE_DIR, "train_ids.npy")

    CACHE_VAL_IMAGES = os.path.join(CACHE_DIR, "val_images.npy")
    CACHE_VAL_LABELS = os.path.join(CACHE_DIR, "val_labels.npy")
    CACHE_VAL_IDS = os.path.join(CACHE_DIR, "val_ids.npy")

    CACHE_TEST_IMAGES = os.path.join(CACHE_DIR, "test_images.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.2  # Default for B0

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Aggressive regularization for small data
    LABEL_SMOOTHING = 0.1  # Mitigates noise from peripheral slices

    # Optimization
    EARLY_STOPPING_PATIENCE = 5

    # Validation
    N_FOLDS = 5  # For GroupKFold
