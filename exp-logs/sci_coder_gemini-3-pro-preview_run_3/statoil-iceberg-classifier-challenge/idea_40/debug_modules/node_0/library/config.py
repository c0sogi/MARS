import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General & Compute
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    BATCH_SIZE = 32

    # -------------------------------------------------------------------------
    # Model Architecture: Projected Dual-Polarity DropBlock CNN
    # -------------------------------------------------------------------------
    MODEL_NAME = "PDP_D_CNN"

    # Backbone: Custom 4-Stage Plain CNN
    # Filters: 64 -> 128 -> 128 -> 128
    BACKBONE_FILTERS = [64, 128, 128, 128]
    USE_BIAS = True
    LEAKY_RELU_SLOPE = 0.1

    # Regularization: DropBlock
    # Applied to Stage 3 and Stage 4 (indices 2 and 3)
    DROPBLOCK_STAGES = [2, 3]
    DROPBLOCK_START_PROB = 0.0
    DROPBLOCK_MAX_PROB = 0.1
    DROPBLOCK_BLOCK_SIZE = 5

    # Projection & Readout (Innovation)
    # Project channels 128 -> 64 before pooling to control capacity
    PROJECTION_DIM = 64
    # Use both Global Max Pooling (Peak) and Global Min Pooling (Shadow)
    USE_DUAL_POLARITY = True

    # Classification Head
    # Input: (64_max + 64_min) * 2_stages + 1_angle = 257 features
    # (or if concatenating stages differently, defined by model logic)
    FC_DROPOUT = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 75
    PATIENCE = 12

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3  # Constant learning rate
    WEIGHT_DECAY = 1e-2  # L2 Regularization

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    USE_TTA = False  # Test-Time Augmentation disabled per strategy
