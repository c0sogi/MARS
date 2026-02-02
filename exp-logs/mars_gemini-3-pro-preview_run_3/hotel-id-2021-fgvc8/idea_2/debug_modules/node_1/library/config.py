import os
import torch


class Config:
    """
    Global configuration for the Hotel ID identification task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLE_SIZE = 2000  # Number of samples to use when DEBUG is True

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data & Preprocessing
    # -------------------------------------------------------------------------
    IMG_SIZE = 256
    NUM_CLASSES = 7770  # Based on metadata analysis (train + singletons)

    # Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # DataLoader
    BATCH_SIZE = 64
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Class-Balanced Sampler Settings
    SAMPLES_PER_CLASS = 4  # K instances per class for M-per-class sampling
    # Note: Effective batch size logic is handled by the sampler.
    # If BATCH_SIZE is 64, we select 64 // SAMPLES_PER_CLASS = 16 unique classes per batch.

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    EMBEDDING_SIZE = 512
    DROPOUT = 0.2

    # ArcFace Head Parameters
    ARC_MARGIN = 0.50
    ARC_SCALE = 30.0

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW
    MIN_LR = 1e-6  # For Cosine Annealing Scheduler

    # Early Stopping
    PATIENCE = 4

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    USE_TTA = True  # Use Test-Time Augmentation (Horizontal Flip)
    TOP_K = 5  # Number of predictions to output per image

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
