import os
import torch


class Config:
    """
    Configuration class for the Unified Translation-Invariant SegFormer.
    Centralizes all file paths, hyperparameters, and strategy-specific constants.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"

    # Ensure the working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Output Path
    SUBMISSION_PATH = "./submission.csv"

    # Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Spatial dimensions of the crops
    TILE_SIZE = 512

    # Volumetric Slab Configuration
    SLAB_DEPTH = 12  # Number of Z-slices in a single input slab
    IN_CHANNELS = 3  # Number of channels for the model input (RGB interface)

    # Constrained Volumetric Translation Strategy
    # Training: Dynamically sample Z-start index from this inclusive range.
    # This forces the model to learn invariance to the ink's position within the channels.
    TRAIN_Z_MIN = 16
    TRAIN_Z_MAX = 24

    # Inference: Deterministic Z-scanning steps.
    # We generate predictions for these Z-starts and Max-Fuse them.
    INFERENCE_Z_STARTS = [16, 20, 24]

    # Debugging / Development Flags
    # Set MAX_SAMPLES to an integer (e.g., 50) to train on a subset for debugging.
    DEBUG = False
    MAX_SAMPLES = None

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # SegFormer Backbone
    BACKBONE = "nvidia/mit-b2"
    NUM_CLASSES = 1
    PRETRAINED = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Micro-Dataset Protocol settings
    BATCH_SIZE = 8
    LEARNING_RATE = 6e-5
    EPOCHS = 20

    # Optimizer settings (AdamW)
    WEIGHT_DECAY = 0.01

    # =========================================================================
    # Compute & Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, 4 workers is a safe balance to avoid overhead
    NUM_WORKERS = 4

    # =========================================================================
    # Evaluation & Metrics
    # =========================================================================
    # Threshold for converting probability map to binary mask (0-1)
    BINARIZATION_THRESHOLD = 0.5

    # Validation Gating Threshold (F0.5 Score)
    # The submission file is only generated if the model beats this score.
    BASELINE_SCORE = 0.598
