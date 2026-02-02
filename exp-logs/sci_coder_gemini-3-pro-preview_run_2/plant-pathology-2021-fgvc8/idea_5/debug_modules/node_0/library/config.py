import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection Task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # -------------------------------------------------------------------------
    # Directory Structure
    # -------------------------------------------------------------------------
    # Base Input Directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata Directory (Pre-generated splits)
    METADATA_DIR = "./metadata"

    # Working Directory (For model checkpoints, logs, cache)
    WORKING_DIR = "./working/idea_5"

    # Submission Directory
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories (Relative to INPUT_DIR in metadata, but defined here for ref)
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMG_SIZE = 384
    NUM_CLASSES = 6

    # Class mapping (Alphabetical order assumed for MultiLabelBinarizer)
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Ensemble Strategy: Heterogeneous Large-Scale Ensemble

    # Model 1: ConvNeXt-Large (OpenCLIP backbone)
    # Selected for robust texture feature extraction and batch stability
    MODEL_1_NAME = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"

    # Model 2: Swin Transformer V2-Large
    # Selected for global attention and numerical stability at scale
    MODEL_2_NAME = "swinv2_large_window12to24_192to384"

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 10

    # Gradient Accumulation Strategy for Effective Batch Size of 32
    # Physical Batch Size: 16 (Fits on A100 40GB with Mixed Precision)
    # Accumulation Steps: 2 (16 * 2 = 32)
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Scheduler Settings (Cosine Annealing)
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 3

    # -------------------------------------------------------------------------
    # Compute & Environment
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    USE_TTA = True  # Enable Test Time Augmentation (Horizontal Flip)
