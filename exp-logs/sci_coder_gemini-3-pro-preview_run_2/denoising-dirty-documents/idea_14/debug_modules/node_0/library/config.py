import os
import torch


class Config:
    """
    Configuration for the Two-Stage Progressive Coordinate ResUNet (TSP-CResUNet) experiment.
    Defines hyperparameters, file paths, and system settings.
    """

    # -------------------------------------------------------------------------
    # Experiment Identifiers
    # -------------------------------------------------------------------------
    EXPERIMENT_NAME = "idea_14"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for quick debugging

    # -------------------------------------------------------------------------
    # Hardware & System
    # -------------------------------------------------------------------------
    # Use CUDA if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use all available vCPUs
    NUM_WORKERS = 12
    PIN_MEMORY = True

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Sub-directories for caching and checkpoints
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Model Save Path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "tsp_cresunet_best.pth")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Patch size for training (128x128)
    PATCH_SIZE = 128
    # Number of random patches to extract per image per epoch
    PATCHES_PER_IMAGE = 100

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE_FILTERS = 64  # Base filter count for U-Net

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 100
    BATCH_SIZE = 32  # Batch size for A100 GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay for regularization
    EARLY_STOPPING_PATIENCE = 15

    # Scheduler Settings (Cosine Annealing)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Inference Hyperparameters
    # -------------------------------------------------------------------------
    # Overlap ratio for tiled inference (50%)
    OVERLAP_RATIO = 0.5
    # Enable Test-Time Augmentation (Geometric flips/rotations)
    USE_TTA = True

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        This is called automatically when the module is imported.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize the directory structure immediately
Config.setup()
