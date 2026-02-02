import os
import torch


class Config:
    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 200  # Number of samples if DEBUG is True

    # --- Directories ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Output Subdirectories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Create directories immediately
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Metadata Paths ---
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --- Data Parameters ---
    IMG_SIZE = 256
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3
    # Total frames in the sequence provided in the numpy arrays (4 before + 1 current + 3 after = 8)
    SEQ_LENGTH = N_TIMES_BEFORE + N_TIMES_AFTER + 1

    # Band Indices (0-based index relative to the list [band_08, ..., band_16])
    # Band 08 is index 0, Band 16 is index 8
    BAND_11_IDX = 3
    BAND_13_IDX = 5
    BAND_14_IDX = 6
    BAND_15_IDX = 7

    # --- Model Parameters ---
    BACKBONE = "convnext_small"
    # Input Channels: 6
    # 3 channels for Ash Color Scheme at time t
    # 3 channels for Temporal Difference (Ash_t - Ash_{t-1})
    IN_CHANNELS = 6
    ENCODER_WEIGHTS = "imagenet"

    # --- Training Hyperparameters ---
    EPOCHS = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-6
    MAX_GRAD_NORM = 10.0

    # Scheduler
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss
    SMOOTH = 1e-6  # For Dice Loss stability

    # --- Hardware ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Checkpointing & Optimization ---
    TOP_K_CHECKPOINTS = 5  # Keep only the best 5 checkpoints based on validation Dice
    START_AVERAGING_EPOCH = 20  # Only average checkpoints after this epoch

    # --- Inference ---
    USE_TTA = True  # Use Test Time Augmentation (Horizontal/Vertical Flips)
