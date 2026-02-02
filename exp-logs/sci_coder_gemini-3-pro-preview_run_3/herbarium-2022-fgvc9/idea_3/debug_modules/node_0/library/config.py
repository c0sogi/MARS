import os
import torch


class Config:
    """
    Configuration class for Plant Image Classification (Idea 3).
    Centralizes all hyperparameters, file paths, and constants.
    """

    # ---------------------------------------------------------
    # General Settings
    # ---------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # ---------------------------------------------------------
    # Directory Paths
    # ---------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories for Idea 3
    WORK_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(WORK_DIR, "last_model.pth")

    # ---------------------------------------------------------
    # Data Configuration
    # ---------------------------------------------------------
    IMAGE_SIZE = 384  # High resolution for fine-grained details
    NUM_CLASSES = 15501
    NUM_WORKERS = 12  # Matches available vCPUs

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ---------------------------------------------------------
    # Model Configuration
    # ---------------------------------------------------------
    MODEL_NAME = "convnext_tiny"  # Timm model name
    PRETRAINED = True
    DROP_PATH_RATE = 0.1  # Stochastic depth rate
    USE_AMP = True  # Automatic Mixed Precision

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    EPOCHS = 20
    BATCH_SIZE = 32  # Adjusted for 384x384 on A100 (40GB)

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3  # Max LR for OneCycle
    WEIGHT_DECAY = 0.05

    # Scheduler (OneCycleLR)
    PCT_START = 0.1  # Percentage of training to increase LR
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 100.0

    # Loss (CrossEntropy)
    LABEL_SMOOTHING = 0.1

    # Regularization (Mixup & CutMix)
    # Applied with probability during training
    USE_MIXUP_CUTMIX = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0  # Probability of applying either Mixup or CutMix
    SWITCH_PROB = 0.5  # Probability of switching between Mixup and CutMix

    # ---------------------------------------------------------
    # Inference Configuration
    # ---------------------------------------------------------
    TTA = True  # Test Time Augmentation (Horizontal Flip)

    # ---------------------------------------------------------
    # Compute
    # ---------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_transforms_config(cls):
        """Returns a dictionary of transform configurations."""
        return {"image_size": cls.IMAGE_SIZE, "mean": cls.MEAN, "std": cls.STD}
