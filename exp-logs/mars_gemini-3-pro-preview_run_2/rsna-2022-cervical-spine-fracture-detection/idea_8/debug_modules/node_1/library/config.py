import os
import torch


class Config:
    """
    Configuration for the 2.5D Dual-Attention Network with Spatially-Guided Feature Aggregation.
    Idea 8: Hierarchical attention with spatial supervision.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_8"

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 8)
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Annotations
    BOUNDING_BOXES = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SEGMENTATIONS_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing & Caching
    # =========================================================================
    # Image resolution: 384x384 is a good balance for B4 backbone + Seq Len 96 on A100
    IMAGE_SIZE = (384, 384)

    # 2.5D Stacking: Input channels = 3 (Slice z-1, z, z+1)
    IN_CHANNELS = 3

    # Sequence Length: High density sampling as per strategy
    SEQ_LEN = 96

    # Data Loading
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    # Caching Flag: Used by dataset class to determine whether to load pre-processed arrays
    LOAD_CACHED_DATA = True
    CACHE_DIR = WORKING_DIR  # Store parquet/npy caches here

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: EfficientNet-B4 (timm implementation)
    BACKBONE = "tf_efficientnet_b4_ns"

    # Feature Extraction
    # Extract features from P4 and P5 blocks
    BACKBONE_OUT_CHANNELS = (
        1792  # Approx for B4 concatenated features, adjusted dynamically in model
    )

    # Sequence Model
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.2

    # Architecture Flags
    USE_SPATIAL_ATTENTION = True  # Enable Spatially-Guided Feature Aggregation
    USE_SLICE_AUX = True  # Enable Auxiliary Slice Classification Head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 12

    # Batch Size: Small per-step batch size due to memory usage of 3D/Sequence data
    BATCH_SIZE = 2

    # Gradient Accumulation: Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS
    # Target effective batch size ~16
    ACCUMULATION_STEPS = 8

    # Optimization
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 1000

    # Scheduler
    SCHEDULER_T_MAX = EPOCHS  # For CosineAnnealingLR
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MODE = "min"  # Monitor validation loss

    # =========================================================================
    # Loss Function Configuration (Tri-Level Loss)
    # =========================================================================
    # 1. Study Level Loss (Weighted Multi-Label Log Loss)
    # Positive class weight > 2.0 to handle class imbalance
    POS_WEIGHT_STUDY = 2.0

    # 2. Auxiliary Slice Loss Weight (Lambda 1)
    LAMBDA_SLICE = 1.0

    # 3. Spatial Attention Loss Weight (Lambda 2)
    # Applied to the spatial attention map using Masked Dice Loss
    LAMBDA_SPATIAL = 1.0

    # =========================================================================
    # Targets & Hardware
    # =========================================================================
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    NUM_CLASSES = len(TARGET_COLS)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __str__(self):
        """Prints the configuration."""
        attributes = [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]
        config_str = "Configuration:\n"
        for attr in attributes:
            config_str += f"{attr}: {getattr(self, attr)}\n"
        return config_str
