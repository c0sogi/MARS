import os
import torch


class Config:
    """
    Configuration for the 2.5D Siamese Multi-Slice Network pipeline.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Generated previously)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for Caching and Models
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = WORKING_DIR  # Directory to store parquet/npy caches
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Image Settings
    # ==========================================
    IMG_SIZE = 256

    # Deterministic Slice Sampling
    # We select slices at 45%, 50%, and 55% depth to capture the tumor volume
    # while handling potential anatomical offsets.
    SLICE_DEPTHS = [0.45, 0.50, 0.55]

    # Modalities to use for the 3-channel composite image
    # Note: T1w is excluded based on the architecture design.
    # Order: Channel 0, Channel 1, Channel 2
    SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1

    # Regularization
    DROPOUT_RATE = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Batch size adjusted for 12 vCPUs and A100 GPU
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay as per design

    # Training Loop
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for outputs and caching.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure environment is ready
Config.setup()
