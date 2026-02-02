import os
import torch


class Config:
    """
    Configuration class for the Anatomically ROI-Focused ConvNeXt MIL Network.
    Contains all file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # System & Hardware
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Matches available vCPUs

    # =========================================================================
    # File Paths
    # =========================================================================
    # Root directories
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    WORKING_DIR = os.path.join(ROOT_DIR, "working")

    # Metadata (Pre-generated)
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Data
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching (Optimized)
    # Ensures the cache directory exists at import time
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_optimized")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission Output
    # Ensures the submission directory exists at import time
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpointing
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # Windowing: Standard Bone Window
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Input Geometry
    # Center Crop to 224x224 (Zooming in on vertebrae)
    IMAGE_SIZE = 224

    # MIL Sequence Length
    SEQ_LEN = 64

    # Input Channels (Replicated for RGB backbone)
    IN_CHANNELS = 3

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True

    # Targets: C1-C7 (7) + Patient_Overall (1)
    NUM_CLASSES = 8

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch Size 8 (Stable with LayerNorm backbone)
    BATCH_SIZE = 8

    # Epochs (Adjustable via argument in training loop, default here)
    EPOCHS = 10

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0  # Gradient clipping

    # Scheduler: Decoupled Cosine Annealing
    # T_max = T_MAX_MULT * EPOCHS
    T_MAX_MULT = 1.5
    MIN_LR = 1e-6

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data for quick verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 32

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print("CONFIGURATION")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key:<25}: {value}")
        print("=" * 40 + "\n")
