import os
import torch


class Config:
    # ==========================
    # Path Configuration
    # ==========================
    ROOT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "test_images")

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(ROOT_DIR, "sample_submission.csv")

    # Output & Cache Paths
    OUTPUT_DIR = WORKING_DIR
    CACHE_DIR = WORKING_DIR
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================
    # Data Configuration
    # ==========================
    # 2.5D Input: Stacking 3 slices (z-1, z, z+1)
    IN_CHANS = 3

    # Image resolution
    # Reduced from 512 to 384 to fit seq_len=96 and backbone=B4 in GPU memory
    IMAGE_SIZE = (384, 384)

    # Sequence length for LSTM (High density sampling)
    SEQ_LEN = 96

    # Targets
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    NUM_CLASSES = len(TARGET_COLS)

    # ==========================
    # Model Configuration
    # ==========================
    BACKBONE = "tf_efficientnet_b4"
    PRETRAINED = True
    HIDDEN_DIM = 256  # LSTM hidden dimension
    DROPOUT = 0.2

    # ==========================
    # Training Configuration
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    EPOCHS = 15
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1000.0

    # Batch Size & Accumulation
    # Reduced from 4 to 2 to prevent OOM with SEQ_LEN=48/96.
    # Effective Batch Size for CNN = 2 * 48 = 96 (fits with checkpointing).
    # Optimization Batch Size = 2 * 8 = 16 (same as before).
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8

    # Early Stopping
    PATIENCE = 5
    MIN_DELTA = 1e-4

    # Workers
    NUM_WORKERS = 12

    # Debugging
    DEBUG = False  # Set to True to train on a small subset for rapid iteration

    # ==========================
    # Loss Configuration
    # ==========================
    # Weighted Multi-Label Logarithmic Loss
    # Strategy: "The any label is weighted more highly than specific fracture level sub-types."
    # We assign weight 1.0 to C1-C7 and 7.0 to patient_overall.
    # We normalize these weights to sum to NUM_CLASSES (8.0) to keep loss magnitude consistent.
    _raw_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    LOSS_WEIGHTS = _raw_weights / _raw_weights.sum() * NUM_CLASSES

    # Calibration: No positive class weighting to ensure probabilistic accuracy
    POS_WEIGHT = 1.0
