import os
import torch


class Config:
    """
    Configuration for the Overlapping Stratified SegFormer (OSS-Net).
    """

    # ==============================
    # General Settings
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset for debugging

    # ==============================
    # Directories and Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "submission.csv"

    # ==============================
    # Data Configuration
    # ==============================
    TILE_SIZE = 512
    STRIDE = 512  # Stride for tiling test images

    # Overlapping Stratified Depth Projection Settings
    # We use 3 channels, each representing a Maximum Intensity Projection (MIP)
    # over a "thick" slab of 12 slices. The slabs overlap by 50% (6 slices).
    # Channel 1: Slices 20 to 32 (Top)
    # Channel 2: Slices 26 to 38 (Middle)
    # Channel 3: Slices 32 to 44 (Bottom)
    Z_RANGES = [(20, 32), (26, 38), (32, 44)]

    # Normalization Parameters (Uint16 data)
    PIXEL_MIN = 0.0
    PIXEL_MAX = 65535.0

    # ==============================
    # Model Configuration
    # ==============================
    # SegFormer MiT-B2 backbone
    MODEL_ENCODER = "nvidia/mit-b2"
    PRETRAINED = True
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # ==============================
    # Training Configuration
    # ==============================
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 1e-2
    EPOCHS = 20

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Learning Rate Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MODE = "max"  # We monitor F0.5 score

    # ==============================
    # Inference Configuration
    # ==============================
    THRESHOLD = 0.5
    USE_TTA = True  # Enable Test Time Augmentation (Flips/Rotations)

    # ==============================
    # Compute
    # ==============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("OSS-Net Configuration:")
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
