import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    IMAGE_SIZE = 224  # Upsampling to 224x224 for ResNet
    CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet34"
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1  # Binary classification
    USE_INC_ANGLE = True  # Whether to fuse incidence angle

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    NUM_EPOCHS = 30
    BATCH_SIZE = 32

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01

    # Loss Function
    LABEL_SMOOTHING = 0.05

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 3
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 7

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures that working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
