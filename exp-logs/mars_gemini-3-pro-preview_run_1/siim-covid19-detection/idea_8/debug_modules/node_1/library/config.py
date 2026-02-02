import os
import torch


class Config:
    """
    Configuration class for the ASPP-Enhanced Multi-Task ResNet18 U-Net.
    Centralizes all hyperparameters, file paths, and environment settings.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (idea_8 specific)
    WORKING_DIR = "./working/idea_8"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 512

    # Study-level labels (Class ID mapping matches the list order)
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_CLASSES = len(STUDY_LABELS)

    # Image-level label
    DETECTION_LABEL = "opacity"

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"

    # Loss Weights (Prioritizing segmentation as per strategy)
    CLS_LOSS_WEIGHT = 1.0
    SEG_LOSS_WEIGHT = 10.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size maximized for A100-40GB with ResNet18+ASPP @ 512x512
    BATCH_SIZE = 32

    # Epochs for full convergence
    EPOCHS = 20

    # Optimizer settings
    # Using AdamW. LR is set based on linear scaling principles for BS=32
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler settings
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_MIN_LR = 1e-6

    # =========================================================================
    # System & Environment
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def get_transforms(data="train"):
        """
        Placeholder for transform definitions if needed by external modules,
        though typically handled in the dataset module.
        """
        pass
