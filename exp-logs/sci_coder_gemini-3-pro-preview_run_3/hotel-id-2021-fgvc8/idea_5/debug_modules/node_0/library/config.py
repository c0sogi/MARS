import os
import torch


class Config:
    """
    Configuration class for Hotel Identification Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ---------------------------------------------------------
    # General Settings
    # ---------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples if DEBUG is True

    # ---------------------------------------------------------
    # Directory Paths
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # Idea 5: EfficientNet-B1 + GeM + BN + ArcFace @ 384x384
    WORKING_DIR = "./working/idea_5"
    OUTPUT_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Data Configuration
    # ---------------------------------------------------------
    IMAGE_SIZE = 384
    NUM_CLASSES = 7770  # Based on dataset analysis
    NUM_WORKERS = 4  # Number of DataLoader workers
    PIN_MEMORY = True

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    MODEL_NAME = "efficientnet_b1"  # timm backbone name
    EMBEDDING_SIZE = 512  # Dimension of embedding before ArcFace head
    PRETRAINED = True
    USE_GEM_POOLING = True
    USE_BN_NECK = True

    # ---------------------------------------------------------
    # ArcFace Hyperparameters
    # ---------------------------------------------------------
    ARCFACE_SCALE = 30.0
    ARCFACE_MARGIN = 0.50
    ARCFACE_LS_EPS = 0.0  # Label smoothing epsilon (0.0 for standard ArcFace)

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    BATCH_SIZE = 32  # Adjusted for 384x384 resolution on A100
    EPOCHS = 12  # Fixed epochs for convergence
    LEARNING_RATE = 1e-3  # Max LR for scheduler
    MIN_LR = 1e-6  # Min LR for scheduler
    WEIGHT_DECAY = 1e-2  # For AdamW
    GRADIENT_CLIP = 1.0  # Gradient clipping value

    # Scheduler
    SCHEDULER_TYPE = "CosineAnnealingLR"
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Inference / Evaluation
    # ---------------------------------------------------------
    TTA = True  # Test Time Augmentation (Horizontal Flip)
    TOP_K = 5  # MAP@5

    # Model Checkpoint Path
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
