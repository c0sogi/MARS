import os
import torch


class Config:
    """
    Configuration for Dog Breed Classification Task.
    Implements parameters for ConvNeXt-Tiny model, progressive fine-tuning,
    and geometric-preserving data processing.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 8 workers to balance between vCPU usage and GPU throughput
    NUM_WORKERS = 8

    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for the current idea iteration
    WORKING_DIR = "./working/idea_2"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Source Paths (Read-only)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths for Processed Data (Parquet format)
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "convnext_tiny_best.pth")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Resize to 256 first, then center crop to 224 to preserve aspect ratio
    RESIZE_SIZE = 256
    CROP_SIZE = 224

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_NAME = "convnext_tiny"
    NUM_CLASSES = 120
    PRETRAINED = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    DROPOUT_RATE = 0.2

    # Optimization
    OPTIMIZER_NAME = "AdamW"
    WEIGHT_DECAY = 0.01

    # Phase 1: Head Adaptation (Backbone Frozen)
    # Short phase to align the random head with pretrained features
    PHASE1_EPOCHS = 3
    PHASE1_LR = 1e-3

    # Phase 2: Fine-Tuning (Stage 4 + Head Unfrozen)
    # Longer phase with discriminative learning rates
    PHASE2_EPOCHS = 20
    PHASE2_BACKBONE_LR = 1e-5  # Low LR to preserve pretrained features
    PHASE2_HEAD_LR = 1e-3  # Higher LR for the task-specific head

    # Regularization & Early Stopping
    PATIENCE = 5  # Stop if validation loss doesn't improve for 5 epochs

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
