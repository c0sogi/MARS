import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Caching & Outputs
    # We use a specific subdirectory for this idea to avoid conflicts
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
    os.makedirs(CACHE_DIR, exist_ok=True)

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # High resolution for microcalcification detection
    IMG_SIZE = (512, 512)

    # 3 Channels: 1 Image + 1 Age (Broadcasted) + 1 Implant (Broadcasted)
    IN_CHANNELS = 3

    # Mapping for Auxiliary Task (Density)
    DENSITY_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
    NUM_AUX_CLASSES = 4

    NUM_WORKERS = 12  # Utilizing available vCPUs

    # =========================================================================
    # Model Configuration
    # =========================================================================
    BACKBONE = "tf_efficientnet_b2_ns"  # Noisy Student weights, B2 variant
    NUM_CLASSES = 1  # Binary Cancer Detection

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE = 16  # Adjusted for A100 40GB RAM with 768x768 resolution
    EPOCHS = 15  # Sufficient for convergence with early stopping

    # Optimization
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-5

    # Loss Weights
    # High positive weight to counter 1:47 imbalance
    POS_WEIGHT = 45.0
    # Weight for the auxiliary density classification task
    AUX_WEIGHT = 0.5

    # Gradient Handling
    MAX_GRAD_NORM = None  # Gradient clipping disabled as per lesson 00009

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
