import os
import torch


class Config:
    """
    Configuration for SE-ResNet18 Multi-Task U-Net with MixUp Regularization.
    """

    # ==============================
    # General Configuration
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # ==============================
    # Data Configuration
    # ==============================
    IMAGE_SIZE = 512
    NUM_CLASSES = 4
    CLASS_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # MixUp Regularization
    MIXUP_ALPHA = 0.4

    # ==============================
    # Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata paths (pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    WORKING_DIR = "./working/idea_9"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Training Hyperparameters
    # ==============================
    EPOCHS = 25
    BATCH_SIZE = 32

    # Optimizer & Scheduler
    # Linear Scaling Rule: LR = Base_LR * (Batch_Size / 256)
    # Using a Base_LR of 4e-3 for Batch Size 256 (standard for ResNet-like backbones)
    # For BS=32: 4e-3 * (32/256) = 5e-4
    BASE_LR_256 = 4e-3
    LEARNING_RATE = BASE_LR_256 * (BATCH_SIZE / 256.0)

    WEIGHT_DECAY = 1e-2

    # Loss Weights (1:10 ratio for Classification:Segmentation)
    LOSS_WEIGHTS = {"class": 1.0, "seg": 10.0}

    # ==============================
    # Compute
    # ==============================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon import
Config.setup()
