import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Input data paths (using metadata as requested)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directory for the specific idea
    WORKING_DIR = "./working/idea_9"
    OUTPUT_DIR = WORKING_DIR

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_CHECKPOINT = "xlm-roberta-large"

    # Weighted Layer Pooling Settings
    # Use the last N hidden layers for the weighted average
    N_LAST_HIDDEN = 4

    # =========================================================================
    # Data Processing
    # =========================================================================
    MAX_LENGTH = 384
    DOC_STRIDE = 128

    # Negative sampling strategy: Retain all windows (no downsampling)
    DOWN_SAMPLE_NEGATIVES = False

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Ensemble Seeds
    SEEDS = [42, 43, 44]

    # Training duration
    EPOCHS = 3

    # Batch sizes
    # A100 40GB can handle roughly 4-8 samples of XLM-R Large with max_len 384.
    # We use 4 here and accumulate gradients.
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 8

    # Gradient Accumulation
    # Effective batch size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS = 4 * 4 = 16
    GRAD_ACCUM_STEPS = 4

    # Optimization
    LEARNING_RATE = 1.5e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Layer-wise Learning Rate Decay (LLRD)
    # Decay rate for learning rates from top layer to bottom layer
    LLRD_DECAY = 0.95

    # =========================================================================
    # System & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Debug mode to run quickly on a subset
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100
