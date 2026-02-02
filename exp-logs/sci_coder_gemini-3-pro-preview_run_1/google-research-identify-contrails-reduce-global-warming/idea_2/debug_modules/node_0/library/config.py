import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, cache)
    # Using 'idea_2' as the current iteration
    WORKING_DIR = "./working/idea_2"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 256
    NUM_FRAMES = 8  # Total frames in npy files (4 before + 1 current + 3 after)
    LABELED_FRAME_IDX = 4  # The 5th frame (index 4) is the labeled one

    # Band Indices in the provided npy files (08, 09, 10, 11, 12, 13, 14, 15, 16)
    # Mapping: 08->0, 09->1, 10->2, 11->3, 12->4, 13->5, 14->6, 15->7, 16->8
    BAND_11_IDX = 3
    BAND_14_IDX = 6
    BAND_15_IDX = 7

    # Ash Color Composite Normalization Constants
    # Based on standard physical properties of contrails
    # Red: Optical Depth (Band 15 - Band 14)
    ASH_RED_MIN = -4.0
    ASH_RED_MAX = 2.0

    # Green: Particle Phase (Band 14 - Band 11)
    ASH_GREEN_MIN = -4.0
    ASH_GREEN_MAX = 5.0

    # Blue: Temperature (Band 14)
    ASH_BLUE_MIN = 243.0
    ASH_BLUE_MAX = 303.0

    # --------------------------------------------------------------------------
    # Model Parameters
    # --------------------------------------------------------------------------
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3  # Ash composite has 3 channels
    CLASSES = 1  # Binary segmentation

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30

    # Loss Function Weights
    FOCAL_LOSS_WEIGHT = 0.75
    DICE_LOSS_WEIGHT = 0.25

    # Focal Loss Parameters
    FOCAL_ALPHA = 0.5
    FOCAL_GAMMA = 2.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-7

    # --------------------------------------------------------------------------
    # Debug / Development
    # --------------------------------------------------------------------------
    # Set to a small integer (e.g., 100) to train on a subset for debugging
    DEBUG_SUBSET_SIZE = None

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
