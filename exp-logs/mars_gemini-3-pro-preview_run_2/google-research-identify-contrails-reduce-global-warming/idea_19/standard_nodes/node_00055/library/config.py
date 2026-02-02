import os
import torch


class Config:
    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    IMAGE_SIZE = 256
    N_CHANNELS = 6  # 3 (Ash Composite) + 3 (Temporal Differences)

    # Bands required for Ash Vector (11, 14, 15) and Temporal Diffs
    USED_BANDS = [11, 14, 15]

    # Temporal offsets defined in dataset description
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "Unet"
    BACKBONE = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # Architecture Specifics
    USE_ATTENTION_GATES = True
    USE_ISOTROPIC_DECODER = True  # Use ConvNeXt blocks in decoder
    USE_SCSE = True
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Loss
    LOSS_FUNCTION = "BCE_BatchDice"

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Augmentation & Regularization
    # ==========================================
    AUG_PROB = 0.5
    # Strict affine only: Rotation, Scale, Shift, Flip
    AUG_TYPES = ["HorizontalFlip", "VerticalFlip", "ShiftScaleRotate"]

    # ==========================================
    # Inference & Post-Processing
    # ==========================================
    THRESHOLD = 0.5
    USE_TTA = True  # Horizontal, Vertical, 180 Rotation

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
