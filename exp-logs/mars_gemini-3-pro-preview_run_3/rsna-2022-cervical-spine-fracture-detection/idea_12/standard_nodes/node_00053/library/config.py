import os
import torch


class Config:
    """
    Configuration class for the LayerNorm-Native Multi-Task MIL Network.
    Encapsulates all file paths, hyperparameters, and constants.
    """

    # =========================================================================
    # File System & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    WORKING_DIR = "./working"

    # Metadata Paths (Pre-generated in ./metadata)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Submission Format
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output & Caching
    # Caching directory for preprocessed 2.5D stacks (uint8)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_12")
    # Path to save the best model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    # Final submission file
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Compute & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # Windowing (Standard Bone Window)
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Input Dimensions
    IMAGE_SIZE = 224  # Resolution for ConvNeXt
    IN_CHANS = 3  # 2.5D Stacking: [z-1, z, z+1]
    NUM_SLICES = 64  # Sequence length (Bag size) for MIL

    # Caching Control
    USE_CACHE = True  # Enable loading/saving to CACHE_DIR

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    NUM_CLASSES = 8  # 7 Vertebrae (C1-C7) + 1 Patient Overall

    # Context Module (1D Convolution)
    CONTEXT_KERNEL_SIZE = 3
    CONTEXT_PADDING = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Small batch size as regularizer
    EPOCHS = 10  # Default epochs, adjustable

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Decoupled Cosine Annealing)
    # T_max will be calculated as EPOCHS * T_MAX_MULTIPLIER
    T_MAX_MULTIPLIER = 1.5
    MIN_LR = 1e-6

    # Loss Weights
    # Implicitly handled by BCE sum (1 patient head vs 7 vertebrae heads)
    # No explicit scalar weights needed per the design.

    # =========================================================================
    # Augmentation
    # =========================================================================
    # Applied consistently across the volume (Volumetric-Consistent)
    AUG_ROTATION = 15  # Degrees
    AUG_SCALE = (0.8, 1.2)  # Scaling factor range
    AUG_SHIFT = 0.1  # Translation fraction

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 20  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Ensure necessary directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
