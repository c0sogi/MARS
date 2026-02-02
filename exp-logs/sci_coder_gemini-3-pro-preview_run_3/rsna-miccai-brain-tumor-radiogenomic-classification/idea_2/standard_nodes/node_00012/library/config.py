import os
import torch


class Config:
    # ==========================================
    # Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # File Paths (Metadata)
    # ==========================================
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # ==========================================
    # File Paths (Cache & Outputs)
    # ==========================================
    # Caching processed tensors to speed up subsequent runs
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "cached_train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "cached_train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "cached_val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "cached_val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "cached_test_X.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cached_test_ids.npy")

    # Model checkpoints and final submission
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    NUM_SLICES = 32  # Number of slices per patient bag
    IMG_SIZE = 256  # Spatial resolution (H, W)
    IN_CHANNELS = 4  # Modalities: FLAIR, T1w, T1wCE, T2w

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 8  # Adjusted for 32 slices * 256x256 * 4 channels
    EPOCHS = 20  # Sufficient for convergence
    LEARNING_RATE = 1e-4  # Standard Adam LR

    # ==========================================
    # Compute & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debug flags to control dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20

    @classmethod
    def setup(cls):
        """Ensures necessary writable directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
