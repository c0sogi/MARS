import os
import torch


class Config:
    # ==============================
    # General Settings
    # ==============================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True

    # ==============================
    # Compute
    # ==============================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Data Paths
    # ==============================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==============================
    # Output Paths
    # ==============================
    WORKING_DIR = "./working/idea_3"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Model Architecture
    # ==============================
    BACKBONE = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    IMG_SIZE = 352
    NUM_SLICES = 1  # 2D Input: Current slice only (replicated to 3 channels)
    IN_CHANNELS = 3
    NUM_CLASSES = 3
    CLASSES = ["large_bowel", "small_bowel", "stomach"]

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 2e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # Loss Configuration
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # ==============================
    # Inference / Post-processing
    # ==============================
    # Threshold for converting probability to binary mask
    MASK_THRESHOLD = 0.5
    # Minimum size for connected components (to remove noise)
    MIN_COMPONENT_SIZE = 50

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories for checkpoints, predictions, and submissions.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
