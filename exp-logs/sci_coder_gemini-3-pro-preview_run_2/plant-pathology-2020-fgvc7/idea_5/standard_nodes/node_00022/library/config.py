import os
import torch


class Config:
    """
    Configuration for Apple Disease Detection (Idea 5).
    Implements strategy using EfficientNet-B5 with Multi-Label Decomposition.
    """

    # ==========================================
    # System & Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers for data loading (safe given 12 vCPUs)
    NUM_WORKERS = 4

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working Directory for Caching and Checkpoints (Idea 5)
    WORKING_DIR = "./working/idea_5"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Using EfficientNet-B5 with Noisy Student weights
    # Native resolution is 456x456
    MODEL_NAME = "tf_efficientnet_b5_ns"
    IMG_SIZE = 456

    # Multi-label decomposition: 2 output nodes (Rust, Scab)
    # Healthy = [0, 0], Rust = [1, 0], Scab = [0, 1], Multiple = [1, 1]
    NUM_CLASSES = 2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size reduced to 8 to fit within ~16GB VRAM
    BATCH_SIZE = 8

    EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # Regularization & Augmentation
    # ==========================================
    DROPOUT_RATE = 0.4

    # Label smoothing applied manually to binary targets
    # e.g., 0 -> 0.05, 1 -> 0.95
    LABEL_SMOOTHING = 0.05

    # CoarseDropout settings (Albumentations)
    COARSE_DROPOUT_MAX_HOLES = 8
    COARSE_DROPOUT_MAX_HEIGHT = 100
    COARSE_DROPOUT_MAX_WIDTH = 100
    COARSE_DROPOUT_MIN_HEIGHT = 16
    COARSE_DROPOUT_MIN_WIDTH = 16

    # ==========================================
    # Inference
    # ==========================================
    # Test Time Augmentation: Original + Horizontal Flip
    TTA_STEPS = 2
