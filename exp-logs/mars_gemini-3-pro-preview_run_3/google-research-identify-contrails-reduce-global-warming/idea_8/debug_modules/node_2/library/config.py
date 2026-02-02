import os
import torch


class Config:
    # ==============================
    # General Configuration
    # ==============================
    PROJECT_NAME = "contrail_segmentation"
    IDEA_NAME = "idea_8"
    SEED = 42

    # ==============================
    # Directories & Paths
    # ==============================
    # Root directory is the current working directory
    ROOT_DIR = os.getcwd()

    # Input Data
    INPUT_DIR = os.path.join(ROOT_DIR, "input")

    # Metadata (Pre-generated)
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Cache, Checkpoints, Predictions)
    WORKING_DIR = os.path.join(ROOT_DIR, "working", IDEA_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==============================
    # Data Parameters
    # ==============================
    IMG_SIZE = 256

    # Temporal sequence details
    # Sequence: [t-4, t-3, t-2, t-1, t, t+1, t+2, t+3] (Total 8 frames usually provided in band arrays)
    # Labeled frame is at index 4 (0-based) -> Time t
    # We need t, t-1, t-2 for the 2nd order difference
    TARGET_FRAME_IDX = 4

    # Input Channels:
    # 3 (Ash t) + 3 (Ash t - Ash t-1) + 3 (Ash t-1 - Ash t-2) = 9 Channels
    INPUT_CHANNELS = 9

    # ==============================
    # Model Architecture
    # ==============================
    BACKBONE = "convnext_base"  # Scaled up from small to base
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES = 1  # Binary segmentation

    # ==============================
    # Training Hyperparameters
    # ==============================
    EPOCHS = 40

    # Batch Size & Gradient Accumulation
    # ConvNeXt-Base is memory intensive.
    # A100 40GB can handle ~24-32 images of size 256x256 with AMP.
    BATCH_SIZE = 24
    ACCUM_ITER = 1  # Effective batch size = BATCH_SIZE * ACCUM_ITER

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Weights (if using hybrid loss)
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # ==============================
    # Hardware & System
    # ==============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs, capped at 12 as per compute spec
    NUM_WORKERS = min(os.cpu_count(), 12)

    # ==============================
    # Checkpointing & Validation
    # ==============================
    # Save Top-5 models based on Dice score
    TOP_K_CHECKPOINTS = 5

    # Validation strategy
    VAL_CHECK_INTERVAL = 1.0  # Check every epoch

    # ==============================
    # Submission
    # ==============================
    SUBMISSION_PATH = os.path.join(ROOT_DIR, "submission", "submission.csv")
