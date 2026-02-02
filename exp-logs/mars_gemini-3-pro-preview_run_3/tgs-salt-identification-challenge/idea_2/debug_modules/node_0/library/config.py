import os
import torch


class Config:
    """
    Configuration class for Salt Segmentation Task.
    Centralizes all file paths, hyperparameters, and model settings.
    """

    # =========================================================================
    # Directory and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Directory for caching deterministic data processing (e.g., resized images/masks)
    # as required by the "Idea" section.
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")

    # Directory for saving model checkpoints during training
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Directory for storing the final submission file
    SUBMISSION_DIR = "./submission"

    # Metadata CSV paths (generated previously)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Final submission file path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specification
    # =========================================================================
    # Original image dimensions
    ORIG_SIZE = 101

    # Target image size for the network.
    # 128 is chosen because it is divisible by 32 (required for ResNeXt encoder)
    # and close to the original 101 size.
    IMG_SIZE = 128

    # Input Channels: 3 (Image converted to RGB) + 1 (Depth Map) = 4
    # This aligns with the strategy of fusing depth as a spatial channel.
    IN_CHANNELS = 4

    # Number of workers for DataLoader (12 vCPUs available)
    NUM_WORKERS = 8

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Encoder backbone
    ENCODER = "resnext50_32x4d"

    # Pretrained weights to initialize the encoder
    ENCODER_WEIGHTS = "imagenet"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # Training Stages
    # Stage 1: Warm-up with BCE + Dice Loss to establish convergence
    EPOCHS_STAGE1 = 50

    # Stage 2: Fine-tuning with Lovász-Hinge Loss to optimize IoU metric directly
    EPOCHS_STAGE2 = 50

    # Early Stopping Patience (number of epochs with no improvement)
    EARLY_STOPPING_PATIENCE = 15

    # Compute Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging: Limit dataset size (Set to None for full training)
    # Useful for quick pipeline verification.
    MAX_SAMPLES = None

    # =========================================================================
    # Setup Logic
    # =========================================================================
    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories if they do not exist."""
        for path in [cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]:
            os.makedirs(path, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup_directories()
