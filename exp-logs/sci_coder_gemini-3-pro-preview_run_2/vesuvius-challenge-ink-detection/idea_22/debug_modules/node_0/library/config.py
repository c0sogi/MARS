import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    EXPERIMENT_NAME = "idea_22"
    DEBUG = False

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VALID_METADATA = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Data Generation & Preprocessing
    # =========================================================================
    # Patch extraction
    TILE_SIZE = 512
    TRAIN_STRIDE = 512  # Non-overlapping for training
    INFERENCE_STRIDE = 256  # Overlapping for inference

    # Z-Axis Projection Strategy (Overlapping Thick Slab)
    # We map a specific Z-range to 3 channels using thick slabs.
    # Total Range Required: 24 slices.
    # Channel 0: [start, start + 12)
    # Channel 1: [start + 6, start + 18)
    # Channel 2: [start + 12, start + 24)
    SLAB_THICKNESS = 12
    SLAB_OVERLAP = 6  # 50% of thickness
    IN_CHANNELS = 3

    # Normalization
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet defaults
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "mit_b2"
    PRETRAINED = True
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8
    LEARNING_RATE = 6e-5
    EPOCHS = 15
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # For early stopping

    # Validation & Metrics
    VALID_THRESHOLD = 0.5
    BETA = 0.5  # For F0.5 Score

    # =========================================================================
    # Specialist Ensemble Definition
    # =========================================================================
    # Defines the three specialist models and their specific Z-ranges.
    # Ranges are [start, end).
    SPECIALISTS = [
        {
            "name": "high",
            "z_start": 16,
            "z_end": 40,  # Covers 16-39 (24 slices)
            "checkpoint_path": os.path.join(WORKING_DIR, "model_high.pth"),
        },
        {
            "name": "mid",
            "z_start": 20,
            "z_end": 44,  # Covers 20-43 (24 slices)
            "checkpoint_path": os.path.join(WORKING_DIR, "model_mid.pth"),
        },
        {
            "name": "low",
            "z_start": 24,
            "z_end": 48,  # Covers 24-47 (24 slices)
            "checkpoint_path": os.path.join(WORKING_DIR, "model_low.pth"),
        },
    ]

    # =========================================================================
    # Submission
    # =========================================================================
    SUBMISSION_PATH = "./submission.csv"
