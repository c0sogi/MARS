import os
import torch


class Config:
    # --------------------
    # General Configuration
    # --------------------
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for debugging
    MAX_DEBUG_SAMPLES = 50  # Number of samples to use when DEBUG is True

    # --------------------
    # Compute Configuration
    # --------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --------------------
    # Directory & File Paths
    # --------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea specific directory for caching and outputs
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_3")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------
    # Data Configuration
    # --------------------
    ORIG_IMAGE_SIZE = 101
    IMAGE_SIZE = 128  # Padded size (multiple of 32 for ResNeXt)
    IN_CHANNELS = 3  # Input channels (e.g., Image, Depth, Coord or duplicated)

    # --------------------
    # Model Configuration
    # --------------------
    MODEL_NAME = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # --------------------
    # Training Configuration
    # --------------------
    BATCH_SIZE = 32
    EPOCHS = 80
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Loss Schedule
    # Start fine-tuning with Lovasz loss after this epoch
    LOVASZ_EPOCH_START = 30

    # --------------------
    # Inference Configuration
    # --------------------
    USE_TTA = True  # Use Test Time Augmentation (Horizontal Flip)
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
