import os
import torch


class Config:
    """
    Configuration module for the EfficientNet-B4 U-Net with Deep Supervision pipeline.
    Centralizes all hyperparameters, file paths, and constants.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = os.cpu_count()

    # =========================================================================
    # Directories & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Optimized Solution (Intermediate files & Checkpoints)
    WORKING_DIR = "./working/optimized"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Source Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths (for deterministic data loading)
    # Training Data Cache
    CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
    CACHE_TRAIN_MASKS = os.path.join(WORKING_DIR, "train_masks.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    # Validation Data Cache
    CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
    CACHE_VAL_MASKS = os.path.join(WORKING_DIR, "val_masks.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    # Test Data Cache
    CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
    CACHE_TEST_DIMS = os.path.join(WORKING_DIR, "test_dims.parquet")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data & Preprocessing
    # =========================================================================
    IMG_SIZE = 512
    NUM_CLASSES = 4

    # Study Labels (Order matters for one-hot encoding/decoding)
    CLASS_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "efficientnet_b4"
    DEEP_SUPERVISION = True  # Enable auxiliary heads at 1/2 and 1/4 resolution

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size adjusted for EfficientNet-B4 @ 512x512 on A100
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Loss Function Weights (1:10 ratio)
    # Prioritize segmentation to force feature learning in the encoder
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_SEG = 10.0

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    # Metric Calculation
    IOU_THRESHOLD = 0.5

    # Prediction Strings
    NONE_PREDICTION = "none 1 0 0 1 1"

    # Gating Logic
    # If study prediction is "Negative", override image prediction to NONE_PREDICTION
    GATING_ENABLED = True
