import os
import torch


class Config:
    # ---------------------------------------------------------
    # General Configuration
    # ---------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ---------------------------------------------------------
    # Directory & File Paths
    # ---------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths (Parquet files)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output & Working Directories
    # Using idea_2 as specified in requirements
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")

    # Checkpoint Paths
    LOCATOR_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "locator_checkpoints")
    FILLER_CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "filler_checkpoints")

    BEST_LOCATOR_PATH = os.path.join(LOCATOR_CHECKPOINT_DIR, "best_locator.pth")
    BEST_FILLER_PATH = os.path.join(FILLER_CHECKPOINT_DIR, "best_filler.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCATOR_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FILLER_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Model Hyperparameters
    # ---------------------------------------------------------
    MODEL_NAME = "distilroberta-base"

    # Sequence Length
    # EDA shows mean word count ~25, but max is high.
    # 128 covers the vast majority of sentences while keeping compute efficient.
    MAX_LEN = 128

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    # Data Streaming
    # We train on a large subset of the 24M available sentences for 1 epoch
    MAX_TRAIN_SAMPLES = 5_000_000
    VAL_SAMPLES = 50_000  # Subset for faster validation

    # Batch Sizes (A100 40GB allows for larger batches)
    TRAIN_BATCH_SIZE = 128
    VAL_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256

    # Optimization
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 1  # Single pass over the massive subset
    WARMUP_RATIO = 0.1

    # Early Stopping
    PATIENCE = 3

    # Logging
    LOG_INTERVAL = 100  # Print metrics every N steps

    # ---------------------------------------------------------
    # Inference Configuration
    # ---------------------------------------------------------
    # Mask token for RoBERTa
    MASK_TOKEN = "<mask>"
