import os
import torch


class Config:
    """
    Central configuration for the Stabilized Slice-Grouped High-Density 2.5D Network pipeline.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for testing pipeline flow
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"

    # Metadata Paths (Pre-generated)
    TRAIN_META_PATH = "./metadata/train.parquet"
    VAL_META_PATH = "./metadata/val.parquet"
    TEST_META_PATH = "./metadata/test.parquet"

    # Working Directory for Caching and Outputs
    WORKING_DIR = "./working/idea_14"

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    NUM_SLICES = 32
    NUM_MODALITIES = 4

    # Input channels = NUM_SLICES * NUM_MODALITIES
    # We stack slices in an interleaved manner: [Slice0_FLAIR, Slice0_T1w, ..., Slice1_FLAIR, ...]
    # Total channels: 32 * 4 = 128
    IN_CHANNELS = NUM_SLICES * NUM_MODALITIES

    # ==========================================
    # Model Configuration
    # ==========================================
    BACKBONE = "efficientnet_b0"
    STEM_OUT_CHANNELS = 64
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 16  # A100 40GB can handle this density
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0  # Explicitly 0.0 as per instructions (Adam, no AdamW)
    PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Caching Configuration
    # ==========================================
    # Paths for cached numpy arrays to avoid reprocessing DICOMs every run
    TRAIN_CACHE_X = os.path.join(WORKING_DIR, "cached_train_X.npy")
    TRAIN_CACHE_Y = os.path.join(WORKING_DIR, "cached_train_y.npy")

    VAL_CACHE_X = os.path.join(WORKING_DIR, "cached_val_X.npy")
    VAL_CACHE_Y = os.path.join(WORKING_DIR, "cached_val_y.npy")

    TEST_CACHE_X = os.path.join(WORKING_DIR, "cached_test_X.npy")
    TEST_CACHE_IDS = os.path.join(WORKING_DIR, "cached_test_ids.npy")

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
