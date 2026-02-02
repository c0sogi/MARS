import os
import torch


class Config:
    """
    Centralized configuration for the Context-Aware Coordinate ResUNet (CAC-ResUNet) pipeline.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific idea (Idea 5)
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "cac_resunet_best.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --- Data Parameters ---
    # Patching strategy: High-density contextual patching
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100

    # Data Loading
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Utilizing available vCPUs

    # --- Model Architecture ---
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE_FILTERS = 64

    # --- Training Hyperparameters ---
    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Aggressive regularization as requested

    # Training Loop
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # --- Inference ---
    # Tiled inference settings
    TILE_OVERLAP = 0.5  # 50% overlap

    # --- Hardware ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Debugging ---
    # Flags to run on a subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 5

    @classmethod
    def setup(cls):
        """
        Ensures that the working and cache directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
